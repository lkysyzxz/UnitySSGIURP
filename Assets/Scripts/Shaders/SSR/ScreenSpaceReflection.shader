Shader "Hidden/SSR/ScreenSpaceReflection"
{
    Properties
    {
        _MainTex ("Source", 2D) = "white" {}
    }

    SubShader
    {
        Tags { "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" }

        Pass
        {
            Name "ScreenSpaceReflection"

            Cull Off
            ZWrite Off
            ZTest Always

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment ScreenSpaceReflectionFrag

            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareOpaqueTexture.hlsl"
            #include "../GI/Commond.hlsl"

            float _SSRMaxDistance;
            int   _SSRMaxSteps;
            float _SSRThickness;

            #define SSR_BINARY_STEPS 10

            // ===== SSR 命中结果 =====
            struct SSRHit
            {
                bool   hit;
                float2 hitUV;
                float3 color;
            };

            // ===== 辅助：view space 位置（正 Z 约定）投影到屏幕 UV =====
            // UNITY_MATRIX_P 期望负 Z，这里取反后投影
            float2 ProjectVStoUV(float3 vsPos)
            {
                float4 clip = mul(UNITY_MATRIX_P, float4(vsPos.xy, -vsPos.z, 1.0));
                return (clip.xy / clip.w) * 0.5 + 0.5;
            }

            // ===== 方法一：DDA 直接命中 =====
            // 屏幕 UV 空间均匀步进（每步约 1 像素），每步做厚度测试。
            SSRHit SSRMarchDDA(float3 rayStartVS, float3 rayEndVS, int maxSteps, float thickness)
            {
                SSRHit result = (SSRHit)0;

                float2 startUV = ProjectVStoUV(rayStartVS);
                float2 endUV   = ProjectVStoUV(rayEndVS);

                // DDA 步数：各轴按对应分辨率转像素，取主轴跨度
                float2 deltaUV      = endUV - startUV;
                float2 deltaPixel   = deltaUV * _ScreenParams.xy;
                float  maxPixelSpan = max(abs(deltaPixel.x), abs(deltaPixel.y));
                int    numSteps     = (int)clamp(maxPixelSpan, 1.0, (float)maxSteps);

                // 1/z 透视正确插值（屏幕空间步进下 z 非线性，1/z 才线性）
                float  invZ0    = 1.0 / rayStartVS.z;
                float  invZ1    = 1.0 / rayEndVS.z;
                float2 stepUV   = deltaUV / (float)numSteps;
                float  invZStep = (invZ1 - invZ0) / (float)numSteps;

                float2 currentUV   = startUV;
                float  currentInvZ = invZ0;

                [loop]
                for (int i = 0; i < numSteps; i++)
                {
                    currentUV   += stepUV;
                    currentInvZ += invZStep;

                    // 出屏停止
                    if (currentUV.x < 0.0 || currentUV.x > 1.0 ||
                        currentUV.y < 0.0 || currentUV.y > 1.0)
                        break;

                    float currentRayZ = 1.0 / currentInvZ;
                    float sceneZ = LinearEyeDepth(SampleSceneDepth(currentUV), _ZBufferParams);

                    // 厚度测试：射线穿到几何体后方，且在容差内
                    float depthDiff = currentRayZ - sceneZ;
                    if (depthDiff > 0.0 && abs(depthDiff) < thickness)
                    {
                        result.hit   = true;
                        result.hitUV = currentUV;
                        result.color = SampleSceneColor(currentUV);
                        break;
                    }
                }

                return result;
            }

            // ===== 方法二：DDA 粗步进 + 二分细化 =====
            // Phase 1：粗步进找射线第一次穿到几何体后方（rayZ > sceneZ）的跨越点
            // Phase 2：在跨越区间 [lo(前方), hi(后方)] 内二分缩窄，精确锁定命中点
            SSRHit SSRMarchBinary(float3 rayStartVS, float3 rayEndVS,
                                  int maxSteps, float thickness, int binarySteps)
            {
                SSRHit result = (SSRHit)0;

                float2 startUV = ProjectVStoUV(rayStartVS);
                float2 endUV   = ProjectVStoUV(rayEndVS);

                float2 deltaUV      = endUV - startUV;
                float2 deltaPixel   = deltaUV * _ScreenParams.xy;
                float  maxPixelSpan = max(abs(deltaPixel.x), abs(deltaPixel.y));
                int    numSteps     = (int)clamp(maxPixelSpan, 1.0, (float)maxSteps);

                float  invZ0    = 1.0 / rayStartVS.z;
                float  invZ1    = 1.0 / rayEndVS.z;
                float2 stepUV   = deltaUV / (float)numSteps;
                float  invZStep = (invZ1 - invZ0) / (float)numSteps;

                // --- Phase 1: 粗步进，找射线第一次穿到几何体后方 ---
                float2 prevUV      = startUV;
                float  prevInvZ    = invZ0;
                float2 currentUV   = startUV;
                float  currentInvZ = invZ0;
                bool   crossed     = false;

                [loop]
                for (int i = 0; i < numSteps; i++)
                {
                    prevUV       = currentUV;
                    prevInvZ     = currentInvZ;
                    currentUV   += stepUV;
                    currentInvZ += invZStep;

                    if (currentUV.x < 0.0 || currentUV.x > 1.0 ||
                        currentUV.y < 0.0 || currentUV.y > 1.0)
                        break;

                    float currentRayZ = 1.0 / currentInvZ;
                    float sceneZ = LinearEyeDepth(SampleSceneDepth(currentUV), _ZBufferParams);

                    if (currentRayZ > sceneZ)   // 射线穿到后方
                    {
                        crossed = true;
                        break;
                    }
                }

                if (!crossed)
                    return result;   // 全程没碰到几何体

                // --- Phase 2: 二分细化 ---
                // lo: 射线在表面前方 (rayZ <= sceneZ) —— 跨越前的最后位置
                // hi: 射线在表面后方 (rayZ >  sceneZ) —— 跨越后的第一个位置
                float2 loUV   = prevUV;
                float  loInvZ = prevInvZ;
                float2 hiUV   = currentUV;
                float  hiInvZ = currentInvZ;

                [loop]
                for (int j = 0; j < binarySteps; j++)
                {
                    float2 midUV     = (loUV + hiUV) * 0.5;
                    float  midInvZ   = (loInvZ + hiInvZ) * 0.5;
                    float  midRayZ   = 1.0 / midInvZ;
                    float  midSceneZ = LinearEyeDepth(SampleSceneDepth(midUV), _ZBufferParams);

                    if (midRayZ > midSceneZ)
                    {
                        // 中点在后方 → 命中点在 [lo, mid]
                        hiUV   = midUV;
                        hiInvZ = midInvZ;
                    }
                    else
                    {
                        // 中点在前方 → 命中点在 [mid, hi]
                        loUV   = midUV;
                        loInvZ = midInvZ;
                    }
                }

                // 二分收敛，hi 是射线从后方逼近表面的位置；做最终厚度确认
                // （排除射线跨过缝隙/厚墙的情况：收敛后若穿透仍很深，判为未命中）
                float finalRayZ   = 1.0 / hiInvZ;
                float finalSceneZ = LinearEyeDepth(SampleSceneDepth(hiUV), _ZBufferParams);
                float depthDiff   = finalRayZ - finalSceneZ;

                if (depthDiff > 0.0 && abs(depthDiff) < thickness)
                {
                    result.hit   = true;
                    result.hitUV = hiUV;
                    result.color = SampleSceneColor(hiUV);
                }

                return result;
            }

            // ===== Fragment =====
            float4 ScreenSpaceReflectionFrag(Varyings input) : SV_Target
            {
                float2 uv = input.texcoord;

                // 1. 采样深度，重建观察空间位置
                float rawDepth = SampleSceneDepth(uv);
                #if UNITY_REVERSED_Z
                    if (rawDepth == 1.0) return SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv); // sky
                #else
                    if (rawDepth == 0.0) return SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv);
                #endif

                float3 viewPos = ComputeViewSpacePosition(uv, rawDepth);

                // 2. 法线 & 反射方向
                float3 normal  = GetNormalFromPosition(viewPos);
                float3 viewDir = normalize(viewPos);
                float3 rayDir  = reflect(viewDir, normal);

                // 正 Z 约定：射线 z<0 朝相机，无可反射内容
                if (rayDir.z < 0.0) return SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv);

                // 3. 射线起终点
                float3 rayStartVS = viewPos;
                float3 rayEndVS   = viewPos + rayDir * _SSRMaxDistance;

                // 4. Ray March —— 切换方法只需换这一行
                SSRHit hit = SSRMarchBinary(rayStartVS, rayEndVS, _SSRMaxSteps, _SSRThickness, SSR_BINARY_STEPS);
                // SSRHit hit = SSRMarchDDA(rayStartVS, rayEndVS, _SSRMaxSteps, _SSRThickness);

                // 5. 合成
                float4 originalColor = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv);
                if (hit.hit)
                {
                    return float4(lerp(originalColor.rgb, hit.color, 0.5f), 1.0);
                }
                return originalColor;
            }
            ENDHLSL
        }
    }
}
