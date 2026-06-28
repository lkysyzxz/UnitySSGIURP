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

            float _SSRStepSize;
            float _SSRMaxDistance;
            int _SSRMaxSteps;
            float _SSRThickness;

            float4 ScreenSpaceReflectionFrag(Varyings input) : SV_Target
            {
                float2 uv = input.texcoord;

                // 1. Sample depth and reconstruct view-space position
                float rawDepth = SampleSceneDepth(uv);
                #if UNITY_REVERSED_Z
                    if (rawDepth == 1.0) return SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv); // sky pixel
                #else
                    if (rawDepth == 0.0) return SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv);
                #endif

                // ComputeViewSpacePosition returns Z = +linearEyeDepth (positive).
                // We KEEP this positive-Z convention internally for ray marching (it is
                // self-consistent). Only the projection step needs special handling,
                // because UNITY_MATRIX_P expects negative Z for points in front of camera.
                float3 viewPos = ComputeViewSpacePosition(uv, rawDepth);

                // 2. Compute view-space normal from position derivatives
                float3 normal = GetNormalFromPosition(viewPos);

                // 3. Compute reflected ray direction in view space
                float3 viewDir = normalize(viewPos);
                float3 rayDir = reflect(viewDir, normal);

                // Positive-Z convention: scene is at z>0, camera at z=0.
                // Reflected ray heading to z<0 goes back towards the camera — skip it.
                if (rayDir.z < 0.0) return SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv);

                // 4. 屏幕空间 DDA Ray March
                //    在 view space 算起点/终点，投影到屏幕 UV，再用 DDA 在 UV 空间均匀划线步进。
                //    步数由屏幕跨度决定（而非固定步长），屏幕采样分布更均匀。

                // 4.1 起点和终点（view space，正 Z 约定）
                float3 rayStartVS = viewPos;
                float3 rayEndVS   = viewPos + rayDir * _SSRMaxDistance;

                // 4.2 投影到屏幕 UV（-Z 因为 UNITY_MATRIX_P 期望负 Z）
                float4 startClip = mul(UNITY_MATRIX_P, float4(rayStartVS.xy, -rayStartVS.z, 1.0));
                float2 startUV = (startClip.xy / startClip.w) * 0.5 + 0.5;
               

                float4 endClip = mul(UNITY_MATRIX_P, float4(rayEndVS.xy, -rayEndVS.z, 1.0));
                float2 endUV   = (endClip.xy / endClip.w) * 0.5 + 0.5;

                // 4.3 DDA 步数：UV 跨度按各轴对应的屏幕分辨率转成像素，取主轴像素跨度
                float2 deltaUV      = endUV - startUV;
                float2 deltaPixel   = deltaUV * _ScreenParams.xy;  // x 乘宽、y 乘高，各自对齐
                float  maxPixelSpan = max(abs(deltaPixel.x), abs(deltaPixel.y));
                int    numSteps     = (int)clamp(maxPixelSpan, 1.0, (float)_SSRMaxSteps);

                // 4.4 每步增量：UV 线性步进 + 1/z 线性插值（透视正确）
                //     3D 直线投影后仍是屏幕直线，沿该直线 1/z 线性变化。
                //     所以 UV 线性步进时插值 1/z，再取倒数得到射线深度，和 currentUV 精确对齐。
                float  invZ0     = 1.0 / rayStartVS.z;
                float  invZ1     = 1.0 / rayEndVS.z;
                float2 stepUV    = deltaUV / (float)numSteps;
                float  invZStep  = (invZ1 - invZ0) / (float)numSteps;

                // 4.5 DDA 步进
                float2 currentUV   = startUV;
                float  currentInvZ = invZ0;
                float3 hitColor    = float3(0, 0, 0);
                bool    hit        = false;

                [loop]
                for (int i = 0; i < numSteps; i++)
                {
                    currentUV += stepUV;
                    currentInvZ += invZStep;

                    // 超出图像边界则停止
                    if (currentUV.x < 0.0 || currentUV.x > 1.0 ||
                        currentUV.y < 0.0 || currentUV.y > 1.0)
                        break;

                    // 当前射线的透视正确深度（1/(1/z) 还原）
                    float currentRayZ = 1.0 / currentInvZ;

                    // 场景深度（正 Z）—— 直接用 LinearEyeDepth，省去 xy 重建
                    float sceneDepth = SampleSceneDepth(currentUV);
                    float sceneZ = LinearEyeDepth(sceneDepth, _ZBufferParams);

                    // 命中检测：射线已穿到几何体后方，且在厚度容差内
                    float depthDiff = currentRayZ - sceneZ;
                    if (depthDiff > 0.0 && abs(depthDiff) < _SSRThickness)
                    {
                        hitColor = SampleSceneColor(currentUV);
                        hit = true;
                        break;
                    }
                }
                // 5. Composite: blend reflected color with original
                float4 originalColor = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv);
                if (hit)
                {
                    return float4(lerp(originalColor,hitColor,0.5f), 1);
                }
                return originalColor;
            }
            ENDHLSL
        }
    }
}
