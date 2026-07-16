Shader "Hidden/SSGI/ScreenSpaceGlobalIllumination"
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
            Name "SSGI"
            Cull Off
            ZWrite Off
            ZTest Always

            HLSLPROGRAM
            #pragma target 3.5
            #pragma vertex Vert
            #pragma fragment FragSSGI
            #pragma multi_compile _ _JITTER_ON
            #pragma multi_compile _ _SSGI_HIZ_ON

            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Sampling/Sampling.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareOpaqueTexture.hlsl"
            #include "../GI/Commond.hlsl"
            #include "../GI/ScreenSpaceRayMarch.hlsl"

            int _SSGIRayCount;
            int _SSGISampleOffset;
            int _SSGIMaxSteps;
            float _SSGIMaxDistance;
            float _SSGIThickness;
            float _SSGIOriginBias;
            float _SSGIIntensity;
            float _SSGIMaxBlend;

            TEXTURE2D_X(_SSGIRadianceTexture);

            static const uint2 SSGI_R2_STEP24 = uint2(12664746u, 9560334u);
            static const float SSGI_R2_SCALE24 = 1.0 / 16777216.0;

            float2 HashRandom(float2 pixelPosition)
            {
                float3 p3 = frac(float3(pixelPosition.xyx) * float3(0.1031, 0.1030, 0.0973));
                p3 += dot(p3, p3.yzx + 33.33);
                return frac((p3.xx + p3.yz) * p3.zy);
            }

            float3 BuildHemisphereDirection(float3 normal, float2 pixelPosition, int rayIndex)
            {
                uint sampleIndex = ((uint)clamp(_SSGISampleOffset, 0, 65535) + (uint)rayIndex) & 0xFFFFu;
                uint2 phase24 = (uint2(sampleIndex, sampleIndex) * SSGI_R2_STEP24) & uint2(0x00FFFFFFu, 0x00FFFFFFu);
                float2 temporalSample = float2(phase24) * SSGI_R2_SCALE24;
                float2 xi = frac(HashRandom(pixelPosition) + temporalSample);
                return SampleHemisphereCosine(xi.x, xi.y, normal);
            }

            float4 FragSSGI(Varyings input) : SV_Target
            {
                float2 uv = input.texcoord;

                float rawDepth = SampleSceneDepth(uv);
                #if UNITY_REVERSED_Z
                    if (rawDepth <= 0.000001) return float4(0, 0, 0, -1.0);
                #else
                    if (rawDepth >= 0.999999) return float4(0, 0, 0, -1.0);
                #endif

                float3 viewPos = ComputeViewSpacePosition(uv, rawDepth);
                float3 normal = GetNormalFromPosition(viewPos);
                if (dot(normal, -viewPos) <= 0.0) normal = -normal;

                float3 irradiance = 0.0;
                int rayCount = clamp(_SSGIRayCount, 1, 128);

                [loop]
                for (int i = 0; i < rayCount; i++)
                {
                    float3 rayDir = BuildHemisphereDirection(normal, input.positionCS.xy, i);
                    float3 rayStartVS = viewPos + normal * _SSGIOriginBias;
                    float3 rayEndVS = rayStartVS + rayDir * _SSGIMaxDistance;

                    #if defined(_SSGI_HIZ_ON)
                    ScreenSpaceRayHit hit = MarchScreenSpaceRayHiZ(rayStartVS, rayEndVS, _SSGIMaxSteps, _SSGIThickness);
                    #else
                    // ScreenSpaceRayHit hit = MarchScreenSpaceRayDDA(rayStartVS, rayEndVS, _SSGIMaxSteps, _SSGIThickness);
                    ScreenSpaceRayHit hit = MarchScreenSpaceRayBinary(rayStartVS, rayEndVS, _SSGIMaxSteps, _SSGIThickness, SCREEN_SPACE_RAY_BINARY_STEPS);
                    #endif
                    if (hit.hit)
                    {
                        float3 hitRadiance = SAMPLE_TEXTURE2D_X(_SSGIRadianceTexture, sampler_LinearClamp, hit.hitUV).rgb;
                        irradiance += hitRadiance;
                    }
                }

                float3 outgoingRadiance = irradiance/rayCount;
                return float4(outgoingRadiance * _SSGIIntensity, min(viewPos.z, 65500.0));
            }
            ENDHLSL
        }

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
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"
            #include "../GI/Commond.hlsl"
            #include "../GI/Blur.hlsl"

            float4 FragBlurH(Varyings input) : SV_Target
            {
                return SampleSSGIEdgeAwareFilter(input.texcoord, _BlurSpread);
            }
            ENDHLSL
        }

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
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"
            #include "../GI/Commond.hlsl"
            #include "../GI/Blur.hlsl"

            float4 FragBlurV(Varyings input) : SV_Target
            {
                return SampleSSGIEdgeAwareFilter(input.texcoord, _BlurSpread * 2.0);
            }
            ENDHLSL
        }

        Pass
        {
            Name "Accumulate"
            Cull Off
            ZWrite Off
            ZTest Always

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment FragAccumulate

            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"

            TEXTURE2D_X(_SSGIHistoryTexture);
            float _SSGIHistoryValid;
            float _SSGIHistoryWeight;
            float _SSGIHistoryDepthThreshold;

            float4 FragAccumulate(Varyings input) : SV_Target
            {
                float2 uv = input.texcoord;
                float4 current = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv);
                if (_SSGIHistoryValid <= 0.0)
                    return current;

                float4 history = SAMPLE_TEXTURE2D_X(_SSGIHistoryTexture, sampler_LinearClamp, uv);
                bool currentHasGeometry = current.a >= 0.0;
                bool historyHasGeometry = history.a >= 0.0;
                if (currentHasGeometry != historyHasGeometry || !currentHasGeometry)
                    return current;

                float currentDepth = current.a;
                float historyDepth = history.a;
                float depthThreshold = max(_SSGIHistoryDepthThreshold, currentDepth * 0.01);
                if (abs(currentDepth - historyDepth) > depthThreshold)
                    return current;

                float3 accumulated = history.rgb + (current.rgb - history.rgb) * _SSGIHistoryWeight;
                return float4(accumulated, current.a);
            }
            ENDHLSL
        }

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
            float _SSGIMaxBlend;

            float4 FragComposite(Varyings input) : SV_Target
            {
                float2 uv = input.texcoord;
                float4 indirect = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv);
                float3 original = SAMPLE_TEXTURE2D_X(_OriginalTexture, sampler_LinearClamp, uv).rgb;
                return float4(original + indirect.rgb * saturate(_SSGIMaxBlend), 1.0);
            }
            ENDHLSL
        }
    }
}
