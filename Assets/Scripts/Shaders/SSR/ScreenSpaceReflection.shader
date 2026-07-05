Shader "Hidden/SSR/ScreenSpaceReflection"
{
    Properties
    {
        _MainTex ("Source", 2D) = "white" {}
    }

    SubShader
    {
        Tags { "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" }

        // ============================================================
        // Pass 0: SSR Raymarch
        //   输出: RGB = 反射颜色, A = 混合系数 (hit=0.5, miss=0)
        // ============================================================
        Pass
        {
            Name "SSR"
            Cull Off
            ZWrite Off
            ZTest Always

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment FragSSR
            #pragma multi_compile _ _JITTER_ON

            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareOpaqueTexture.hlsl"
            #include "../GI/Commond.hlsl"

            float _SSRMaxDistance;
            int   _SSRMaxSteps;
            float _SSRThickness;

            #define DECLARE_HIZ(i) TEXTURE2D(_HiZTexture_##i)
            DECLARE_HIZ(0); DECLARE_HIZ(1); DECLARE_HIZ(2); DECLARE_HIZ(3);
            DECLARE_HIZ(4); DECLARE_HIZ(5); DECLARE_HIZ(6); DECLARE_HIZ(7);
            int _HiZMipCount;
            float _HiZMaxMip;

            // 每层是独立 RT，按 mipLevel 选纹理采样（复用 Blit.hlsl 的 sampler_PointClamp）
            float SampleHiZ(float2 uv, int mip)
            {
                [flatten]
                if (mip <= 0) return _HiZTexture_0.Sample(sampler_PointClamp, uv).r;
                else if (mip == 1) return _HiZTexture_1.Sample(sampler_PointClamp, uv).r;
                else if (mip == 2) return _HiZTexture_2.Sample(sampler_PointClamp, uv).r;
                else if (mip == 3) return _HiZTexture_3.Sample(sampler_PointClamp, uv).r;
                else if (mip == 4) return _HiZTexture_4.Sample(sampler_PointClamp, uv).r;
                else if (mip == 5) return _HiZTexture_5.Sample(sampler_PointClamp, uv).r;
                else if (mip == 6) return _HiZTexture_6.Sample(sampler_PointClamp, uv).r;
                else return _HiZTexture_7.Sample(sampler_PointClamp, uv).r;
            }

            #define SSR_BINARY_STEPS 10

            struct SSRHit
            {
                bool   hit;
                float2 hitUV;
                float3 color;
            };

            float2 ProjectVStoUV(float3 vsPos)
            {
                float4 clip = mul(UNITY_MATRIX_P, float4(vsPos.xy, -vsPos.z, 1.0));
                return (clip.xy / clip.w) * 0.5 + 0.5;
            }

            #if defined(_JITTER_ON)
            // 4x4 Bayer dither 表：不同像素给射线起点加 [0,1) 步长的随机偏移，
            // 打破规律采样、大幅减少步进次数。格子瑕疵由后续高斯模糊消除。
            static const float _DitherTable[16] = {
                0.0,    0.5,    0.125,  0.625,
                0.75,   0.25,   0.875,  0.375,
                0.1875, 0.6875, 0.0625, 0.5625,
                0.9375, 0.4375, 0.8125, 0.3125
            };
            float GetJitter(float2 uv)
            {
                uint2 pix = uint2(uv * _ScreenParams.xy);
                return _DitherTable[(pix.x & 3) * 4 + (pix.y & 3)];
            }
            #endif

            // 方法一：DDA 直接命中
            SSRHit SSRMarchDDA(float3 rayStartVS, float3 rayEndVS, int maxSteps, float thickness)
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

                float2 currentUV   = startUV;
                float  currentInvZ = invZ0;

                #if defined(_JITTER_ON)
                {
                    float jit = GetJitter(startUV);
                    currentUV   += stepUV * jit;
                    currentInvZ += invZStep * jit;
                }
                #endif

                [loop]
                for (int i = 0; i < numSteps; i++)
                {
                    currentUV   += stepUV;
                    currentInvZ += invZStep;

                    if (currentUV.x < 0.0 || currentUV.x > 1.0 ||
                        currentUV.y < 0.0 || currentUV.y > 1.0)
                        break;

                    float currentRayZ = 1.0 / currentInvZ;
                    float sceneZ = LinearEyeDepth(SampleSceneDepth(currentUV), _ZBufferParams);

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

            // 方法二：DDA 粗步进 + 二分细化
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

                // Phase 1: 粗步进找穿越点
                float2 prevUV      = startUV;
                float  prevInvZ    = invZ0;
                float2 currentUV   = startUV;
                float  currentInvZ = invZ0;
                bool   crossed     = false;

                #if defined(_JITTER_ON)
                {
                    float jit = GetJitter(startUV);
                    currentUV   += stepUV * jit;
                    currentInvZ += invZStep * jit;
                }
                #endif

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

                    if (currentRayZ > sceneZ)
                    {
                        crossed = true;
                        break;
                    }
                }

                if (!crossed)
                    return result;

                // Phase 2: 二分细化
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
                        hiUV   = midUV;
                        hiInvZ = midInvZ;
                    }
                    else
                    {
                        loUV   = midUV;
                        loInvZ = midInvZ;
                    }
                }

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

            // 方法三：基于 HiZ（层次化深度）的自适应步进
            // 空旷区域升 mip 大步跳过，遇到障碍降 mip 精细化，通常比 DDA/二分步数少得多
            SSRHit SSRMarchHiZ(float3 rayStartVS, float3 rayEndVS, int maxSteps, float thickness)
            {
                SSRHit result = (SSRHit)0;

                float2 startUV = ProjectVStoUV(rayStartVS);
                float2 endUV   = ProjectVStoUV(rayEndVS);

                float2 startPos  = startUV * _ScreenParams.xy;
                float2 endPos    = endUV   * _ScreenParams.xy;
                float  totalDist = distance(endPos, startPos);
                if (totalDist < 1.0) return result;

                float2 rayDir2D = (endPos - startPos) / totalDist;

                float invZ0 = 1.0 / rayStartVS.z;
                float invZ1 = 1.0 / rayEndVS.z;

                float2 pos      = startPos;
                float  maxMip   = _HiZMaxMip;
                float  mipLevel = maxMip * 0.5;   // 从 maxMip/2 开始

                #if defined(_JITTER_ON)
                {
                    float jit = GetJitter(startUV);
                    pos += rayDir2D * jit;
                }
                #endif

                // 纯 HiZ（参考 Efficient GPU Screen-Space Ray Tracing）：
                // 穿过时回溯+降 mip 精化（不 break），mip 0 命中。bias 挡自反射后射线继续降 mip 找命中，不会 miss
                [loop]
                for (int i = 0; i < maxSteps; i++)
                {
                    float stride = exp2(mipLevel);
                    pos += rayDir2D * stride;

                    float2 uv = pos / _ScreenParams.xy;
                    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
                        break;

                    float t = saturate(distance(pos, startPos) / totalDist);
                    float rayZ = 1.0 / lerp(invZ0, invZ1, t);

                    float rawDepth = SampleHiZ(uv, (int)mipLevel);
                    float sceneZ = LinearEyeDepth(rawDepth, _ZBufferParams);

                    if (rayZ > sceneZ)   // 穿过
                    {
                        if (mipLevel <= 0.0)
                        {
                            float depthDiff = rayZ - sceneZ;
                            if (depthDiff > 0.0 && depthDiff < thickness)
                            {
                                result.hit   = true;
                                result.hitUV = uv;
                                result.color = SampleSceneColor(uv);
                            }
                            break;
                        }
                        pos -= rayDir2D * stride;
                        mipLevel -= 1.0;
                    }
                    else
                    {
                        mipLevel = min(maxMip, mipLevel + 1.0);
                    }
                }
                return result;
            }

            float4 FragSSR(Varyings input) : SV_Target
            {
                float2 uv = input.texcoord;

                float rawDepth = SampleSceneDepth(uv);
                #if UNITY_REVERSED_Z
                    if (rawDepth == 1.0) return float4(0, 0, 0, 0); // sky
                #else
                    if (rawDepth == 0.0) return float4(0, 0, 0, 0);
                #endif

                float3 viewPos = ComputeViewSpacePosition(uv, rawDepth);
                float3 normal  = GetNormalFromPosition(viewPos);
                float3 viewDir = normalize(viewPos);
                float3 rayDir  = reflect(viewDir, normal);

                if (rayDir.z < 0.0) return float4(0, 0, 0, 0); // 朝相机

                float3 rayStartVS = viewPos;
                float3 rayEndVS   = viewPos + rayDir * _SSRMaxDistance;

                SSRHit hit = SSRMarchBinary(rayStartVS, rayEndVS, _SSRMaxSteps, _SSRThickness, SSR_BINARY_STEPS);
                // SSRHit hit = SSRMarchDDA(rayStartVS, rayEndVS, _SSRMaxSteps, _SSRThickness);
                // SSRHit hit = SSRMarchHiZ(rayStartVS, rayEndVS, _SSRMaxSteps, _SSRThickness);

                // RGB=反射色, A=混合系数（hit=0.5, miss=0）
                if (hit.hit)
                    return float4(hit.color, 0.5);
                return float4(0, 0, 0, 0);
            }
            ENDHLSL
        }

        // ============================================================
        // Pass 1: 水平高斯模糊 (5-tap 分离卷积)
        // ============================================================
        Pass
        {
            Name "BlurH"
            Cull Off
            ZWrite Off
            ZTest Always

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment FragBlurH

            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"

            float _BlurSpread;

            // 5-tap 二项式权重 [1,4,6,4,1]/16，归一化总和=1
            static const float w0 = 0.375;   // 中心
            static const float w1 = 0.25;    // ±1
            static const float w2 = 0.0625;  // ±2

            float4 FragBlurH(Varyings input) : SV_Target
            {
                float2 uv    = input.texcoord;
                float  off   = _BlurSpread / _ScreenParams.x;

                float4 col = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv) * w0;
                col += SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv + float2(off, 0.0)) * w1;
                col += SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv - float2(off, 0.0)) * w1;
                col += SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv + float2(2.0 * off, 0.0)) * w2;
                col += SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv - float2(2.0 * off, 0.0)) * w2;
                return col;
            }
            ENDHLSL
        }

        // ============================================================
        // Pass 2: 垂直高斯模糊 (5-tap 分离卷积)
        // ============================================================
        Pass
        {
            Name "BlurV"
            Cull Off
            ZWrite Off
            ZTest Always

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment FragBlurV

            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"

            float _BlurSpread;

            static const float w0 = 0.375;
            static const float w1 = 0.25;
            static const float w2 = 0.0625;

            float4 FragBlurV(Varyings input) : SV_Target
            {
                float2 uv    = input.texcoord;
                float  off   = _BlurSpread / _ScreenParams.y;

                float4 col = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv) * w0;
                col += SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv + float2(0.0, off)) * w1;
                col += SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv - float2(0.0, off)) * w1;
                col += SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv + float2(0.0, 2.0 * off)) * w2;
                col += SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv - float2(0.0, 2.0 * off)) * w2;
                return col;
            }
            ENDHLSL
        }

        // ============================================================
        // Pass 3: 合成
        //   _BlitTexture     = 模糊后的反射 (rgb=反射色, a=混合系数)
        //   _OriginalTexture = 原图
        //   result = lerp(原图, 反射色, a)
        // ============================================================
        Pass
        {
            Name "Composite"
            Cull Off
            ZWrite Off
            ZTest Always

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment FragComposite

            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"

            TEXTURE2D_X(_OriginalTexture);

            float4 FragComposite(Varyings input) : SV_Target
            {
                float2 uv = input.texcoord;
                float4 reflected = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv);
                float3 original  = SAMPLE_TEXTURE2D_X(_OriginalTexture, sampler_LinearClamp, uv).rgb;
                return float4(lerp(original, reflected.rgb, reflected.a), 1.0);
            }
            ENDHLSL
        }
    }
}
