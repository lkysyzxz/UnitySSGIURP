Shader "Hidden/SSGI/ScreenSpaceGlobalIllumination"
{
    Properties
    {
        _MainTex ("Source", 2D) = "white" {}
    }

    SubShader
    {
        Tags { "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" }

        // =====================================================================
        // Pass 0: SSGI trace
        //
        // Reads _SSGIGBufferAlbedo / _SSGIGBufferNormalWS / _SSGIGBufferPositionWS
        // (rendered by ForwardGBufferPass at full resolution). When this pass
        // is dispatched with a half-resolution destination RT, _ScreenParams.xy
        // reports the half-res dimensions while the G-buffer textures keep
        // their full-res footprint; UVs are normalised so the same input.uv
        // maps into both spaces and we get correct per-pixel G-buffer data.
        //
        // Hemisphere RNG is selected by the _SSGI_RNG_HASH / _SSGI_RNG_R2
        // multi_compile directives (see SSGIRandom.hlsl).
        // =====================================================================
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
            // Hemisphere RNG selection keeps both algorithms swappable for
            // A/B comparison as requested by the user. Default (no keyword) is
            // the legacy R2 low-discrepancy sequence; _SSGI_RNG_HASH switches
            // to the UnitySSGIURP hash+frame-counter generator.
            #pragma multi_compile _ _SSGI_RNG_HASH

            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Sampling/Sampling.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareOpaqueTexture.hlsl"
            #include "../GI/Commond.hlsl"
            #include "../GI/SSGIRandom.hlsl"
            #include "../GI/ScreenSpaceRayMarch.hlsl"

            int _SSGIRayCount;
            int _SSGIMaxSteps;
            float _SSGIMaxDistance;
            float _SSGIThickness;
            float _SSGIOriginBias;
            float _SSGIIntensity;
            float _SSGIPreviousIrradianceValid;
            float4 _SSGITraceSize;

            TEXTURE2D_X(_SSGIRadianceTexture);
            TEXTURE2D_X(_SSGIIrradianceTexture);
            TEXTURE2D_X(_SSGIGBufferAlbedo);
            TEXTURE2D_X(_SSGIGBufferMaterial);
            TEXTURE2D_X(_SSGIGBufferNormalWS);
            TEXTURE2D_X(_SSGIGBufferPositionWS);

            static const float SSGI_INV_PI = 0.31830988618;
            static const float SSGI_MIN_PDF = 0.00001;

            float4 FragSSGI(Varyings input) : SV_Target
            {
                float2 uv = input.texcoord;

                float rawDepth = SampleSceneDepth(uv);
                #if UNITY_REVERSED_Z
                    if (rawDepth <= 0.000001) return float4(0, 0, 0, -1.0);
                #else
                    if (rawDepth >= 0.999999) return float4(0, 0, 0, -1.0);
                #endif

                float4 positionData = SAMPLE_TEXTURE2D_X(
                    _SSGIGBufferPositionWS, sampler_PointClamp, uv);
                float4 normalData = SAMPLE_TEXTURE2D_X(
                    _SSGIGBufferNormalWS, sampler_PointClamp, uv);
                if (positionData.a < 0.5 || normalData.a < 0.5)
                    return float4(0, 0, 0, -1.0);

                // The marcher uses +Z-forward view space; Unity uses -Z.
                float3 viewPos = mul(
                    UNITY_MATRIX_V, float4(positionData.xyz, 1.0)).xyz;
                viewPos.z = -viewPos.z;
                float3 normal = mul((float3x3)UNITY_MATRIX_V, normalData.xyz);
                normal.z = -normal.z;
                normal = normalize(normal);
                if (dot(normal, -viewPos) <= 0.0)
                    normal = -normal;


                float3 irradiance = 0.0;
                int rayCount = clamp(_SSGIRayCount, 1, 128);

                [loop]
                for (int i = 0; i < rayCount; i++)
                {
                    float3 rayDir = SSGIBuildHemisphereDirection(
                        normal,
                        input.positionCS.xy,
                        i,
                        uv,
                        _SSGITraceSize.xy);

                    // Monte Carlo estimator for diffuse transport:
                    // Li * f_d * cos(theta) / pdf. The hemisphere sampler is
                    // cosine weighted, so pdf = cos(theta) / PI. Receiver
                    // albedo is factored out and applied once in Composite;
                    // the BRDF term represented here is therefore 1 / PI.
                    float receiverCosine = saturate(dot(normal, rayDir));
                    float samplePDF = max(
                        receiverCosine * SSGI_INV_PI,
                        SSGI_MIN_PDF);
                    float receiverDiffuseBRDF = SSGI_INV_PI;
                    float monteCarloWeight =
                        receiverDiffuseBRDF * receiverCosine / samplePDF;

                    float3 rayStartVS = viewPos + normal * _SSGIOriginBias;
                    float3 rayEndVS = rayStartVS + rayDir * _SSGIMaxDistance;

                    #if defined(_SSGI_HIZ_ON)
                    ScreenSpaceRayHit hit = MarchScreenSpaceRayHiZ(rayStartVS, rayEndVS, _SSGIMaxSteps, _SSGIThickness);
                    #else
                    //ScreenSpaceRayHit hit = MarchScreenSpaceRayDDA(rayStartVS, rayEndVS,  _SSGIMaxSteps, _SSGIThickness);
                    ScreenSpaceRayHit hit = MarchScreenSpaceRayBinary(rayStartVS, rayEndVS, _SSGIMaxSteps, _SSGIThickness, SCREEN_SPACE_RAY_BINARY_STEPS);
                    #endif
                    if (hit.hit)
                    {
                        // A depth crossing immediately next to the ray origin
                        // is the source surface, not an indirect-light hit.
                        // Keeping it turns the marcher's discrete step pattern
                        // into stable horizontal bands on large planar walls.
                        // float minimumHitDistance = max(
                        //     _SSGIOriginBias * 2.0,
                        //     _SSGIThickness);
                        // float hitDistance = distance(hit.hitPosVS, rayStartVS);
                        // // if (hitDistance <= minimumHitDistance)
                        // //     continue;

                        // Camera color already contains the hit surface's
                        // direct material response, including its albedo.
                        float3 hitOutgoingRadiance = SAMPLE_TEXTURE2D_X(
                            _SSGIRadianceTexture,
                            sampler_LinearClamp,
                            hit.hitUV).rgb;

                        // Previous irradiance is incident light at the hit
                        // surface. Convert it to diffuse outgoing radiance with
                        // that surface's albedo and metallic mask. This is the
                        // only place where hit albedo is applied.
                        if (_SSGIPreviousIrradianceValid > 0.5)
                        {
                            float3 hitIndirectIrradiance = SAMPLE_TEXTURE2D_X(
                                _SSGIIrradianceTexture,
                                sampler_LinearClamp,
                                hit.hitUV).rgb;
                            float3 hitAlbedo = SAMPLE_TEXTURE2D_X(
                                _SSGIGBufferAlbedo,
                                sampler_PointClamp,
                                hit.hitUV).rgb;
                            float hitMetallic = SAMPLE_TEXTURE2D_X(
                                _SSGIGBufferMaterial,
                                sampler_PointClamp,
                                hit.hitUV).r;
                            float3 hitDiffuseReflectance =
                                hitAlbedo * (1.0 - saturate(hitMetallic));
                            hitOutgoingRadiance +=
                                hitIndirectIrradiance * hitDiffuseReflectance;
                        }

                        irradiance +=
                            hitOutgoingRadiance * monteCarloWeight;
                    }
                }

                float surfaceDepth = min(max(viewPos.z, 0.0), 65500.0);
                // A miss is a valid zero-valued Monte Carlo sample. Dividing
                // by the configured ray count, rather than the hit count,
                // keeps the estimator unbiased with respect to screen-space
                // visibility and lets temporal accumulation count every frame.
                float3 outgoingRadiance =
                    irradiance / rayCount * _SSGIIntensity;
                return float4(outgoingRadiance, surfaceDepth);
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
            #include "../GI/SSGIBlur.hlsl"

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
            #include "../GI/SSGIBlur.hlsl"

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
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"
            #include "../GI/Temporal.hlsl"

            float4 FragAccumulate(Varyings input) : SV_Target
            {
                float2 uv = input.texcoord;
                float4 current = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv);
                float encodedDepth = SAMPLE_TEXTURE2D_X(
                    _BlitTexture, sampler_PointClamp, uv).a;

                float rawDepth = SampleSceneDepth(uv);
                if (!IsSSGIRawDepthValid(rawDepth))
                    return current;

                bool hasCurrentObservation = encodedDepth > 0.0;
                float currentDepth = abs(encodedDepth);
                current.a = currentDepth;

                SSGITemporalHistory temporal = GetSSGITemporalHistory(uv, rawDepth);

                // Negative alpha means the current pixel has no valid receiver
                // G-buffer data. Ray misses on a valid receiver use positive
                // depth and zero RGB, so they still participate in the mean.
                // Reuse history here only after reprojection/depth validation.
                if (!hasCurrentObservation)
                {
                    return temporal.valid
                        ? float4(temporal.irradianceDepth.rgb, currentDepth)
                        : float4(0.0, 0.0, 0.0, currentDepth);
                }

                if (!temporal.valid)
                    return current;

                float maxHistoryFrames = max(_SSGIMaxHistoryFrames, 1.0);
                float nextSampleCount = min(
                    temporal.sampleCount + 1.0,
                    maxHistoryFrames);
                // Before the cap this is the incremental Monte Carlo mean:
                // mean_N = mean_(N-1) * (N-1)/N + sample_N / N.
                // At the cap, use the configured history weight so lighting
                // can still respond to changes instead of freezing forever.
                float historyWeight = temporal.sampleCount >= maxHistoryFrames
                    ? saturate(_SSGIHistoryWeight)
                    : temporal.sampleCount / nextSampleCount;
                float currentFrameWeight = temporal.sampleCount >= maxHistoryFrames ? 1.0 - historyWeight:1.0f/nextSampleCount;
                float3 accumulated =
                    temporal.irradianceDepth.rgb * historyWeight +
                    current.rgb * currentFrameWeight;
                return float4(accumulated, currentDepth);
            }
            ENDHLSL
        }

        Pass
        {
            Name "UpdateSampleCount"
            Cull Off
            ZWrite Off
            ZTest Always

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment FragUpdateSampleCount

            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"
            #include "../GI/Temporal.hlsl"

            float FragUpdateSampleCount(Varyings input) : SV_Target
            {
                float2 uv = input.texcoord;
                float encodedDepth = SAMPLE_TEXTURE2D_X(
                    _BlitTexture, sampler_PointClamp, uv).a;
                float rawDepth = SampleSceneDepth(uv);
                if (!IsSSGIRawDepthValid(rawDepth))
                    return 0.0;

                bool hasCurrentObservation = encodedDepth > 0.0;
                SSGITemporalHistory temporal =
                    GetSSGITemporalHistory(uv, rawDepth);

                if (!hasCurrentObservation)
                    return temporal.valid ? temporal.sampleCount : 0.0;
                if (!temporal.valid)
                    return 1.0;

                return min(
                    temporal.sampleCount + 1.0,
                    max(_SSGIMaxHistoryFrames, 1.0));
            }
            ENDHLSL
        }

        // =====================================================================
        // Pass 5: Combine (full-resolution composite)
        //
        // Samples the half-resolution _SSGIIrradianceTexture using a
        // nearest-depth bilinear upsample (modelled on UnitySSGIURP's
        // DepthNormalsUpscale), multiplies by albedo from _SSGIGBufferAlbedo
        // and adds the result to the current camera color (_BlitTexture).
        //
        // Writes the composited color to the destination RT, which is an
        // intermediate finalRT owned by the SSGI feature. A subsequent plain
        // blit copies finalRT back into the camera color target.
        // =====================================================================
        Pass
        {
            Name "Combine"
            Cull Off
            ZWrite Off
            ZTest Always

            HLSLPROGRAM
            #pragma target 3.5
            #pragma vertex Vert
            #pragma fragment FragCombine

            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"

            TEXTURE2D_X(_SSGIIrradianceTexture);
            TEXTURE2D_X(_SSGIGBufferAlbedo);
            TEXTURE2D_X(_SSGIGBufferMaterial);
            TEXTURE2D_X(_SSGIGBufferNormalWS);
            TEXTURE2D_X(_SSGIGBufferPositionWS);

            // Texel size of the half-resolution irradiance RT. The CPU side
            // keeps this in sync regardless of the actual allocation size.
            float4 _SSGIIrradianceTexture_TexelSize;
            float _SSGIHalfResolution;

            bool SSGICombineIsSkyDepth(float rawDepth)
            {
                #if UNITY_REVERSED_Z
                    return rawDepth <= 0.000001;
                #else
                    return rawDepth >= 0.999999;
                #endif
            }

            float3 SSGICombineDepthAwareUpsample(float2 uv)
            {
                if (_SSGIHalfResolution < 0.5)
                    return SAMPLE_TEXTURE2D_X(
                        _SSGIIrradianceTexture, sampler_LinearClamp, uv).rgb;

                float2 texelPosition =
                    uv * _SSGIIrradianceTexture_TexelSize.zw - 0.5;
                float2 baseTexel = floor(texelPosition);
                float2 uv0 = (baseTexel + float2(0.5, 0.5)) *
                    _SSGIIrradianceTexture_TexelSize.xy;
                float2 uv1 = (baseTexel + float2(1.5, 0.5)) *
                    _SSGIIrradianceTexture_TexelSize.xy;
                float2 uv2 = (baseTexel + float2(0.5, 1.5)) *
                    _SSGIIrradianceTexture_TexelSize.xy;
                float2 uv3 = (baseTexel + float2(1.5, 1.5)) *
                    _SSGIIrradianceTexture_TexelSize.xy;
                uv0 = saturate(uv0);
                uv1 = saturate(uv1);
                uv2 = saturate(uv2);
                uv3 = saturate(uv3);

                float4 centerPosition = SAMPLE_TEXTURE2D_X(
                    _SSGIGBufferPositionWS, sampler_PointClamp, uv);
                float3 centerNormal = normalize(SAMPLE_TEXTURE2D_X(
                    _SSGIGBufferNormalWS, sampler_PointClamp, uv).xyz);
                float4 p0 = SAMPLE_TEXTURE2D_X(_SSGIGBufferPositionWS, sampler_PointClamp, uv0);
                float4 p1 = SAMPLE_TEXTURE2D_X(_SSGIGBufferPositionWS, sampler_PointClamp, uv1);
                float4 p2 = SAMPLE_TEXTURE2D_X(_SSGIGBufferPositionWS, sampler_PointClamp, uv2);
                float4 p3 = SAMPLE_TEXTURE2D_X(_SSGIGBufferPositionWS, sampler_PointClamp, uv3);
                float3 n0 = normalize(SAMPLE_TEXTURE2D_X(_SSGIGBufferNormalWS, sampler_PointClamp, uv0).xyz);
                float3 n1 = normalize(SAMPLE_TEXTURE2D_X(_SSGIGBufferNormalWS, sampler_PointClamp, uv1).xyz);
                float3 n2 = normalize(SAMPLE_TEXTURE2D_X(_SSGIGBufferNormalWS, sampler_PointClamp, uv2).xyz);
                float3 n3 = normalize(SAMPLE_TEXTURE2D_X(_SSGIGBufferNormalWS, sampler_PointClamp, uv3).xyz);

                float normalScale = max(
                    length(centerPosition.xyz - _WorldSpaceCameraPos) * 0.01,
                    0.05);
                float4 distances;
                distances.x = length(p0.xyz - centerPosition.xyz) +
                    (1.0 - saturate(dot(n0, centerNormal))) * normalScale;
                distances.y = length(p1.xyz - centerPosition.xyz) +
                    (1.0 - saturate(dot(n1, centerNormal))) * normalScale;
                distances.z = length(p2.xyz - centerPosition.xyz) +
                    (1.0 - saturate(dot(n2, centerNormal))) * normalScale;
                distances.w = length(p3.xyz - centerPosition.xyz) +
                    (1.0 - saturate(dot(n3, centerNormal))) * normalScale;

                float4 i0 = SAMPLE_TEXTURE2D_X(_SSGIIrradianceTexture, sampler_PointClamp, uv0);
                float4 i1 = SAMPLE_TEXTURE2D_X(_SSGIIrradianceTexture, sampler_PointClamp, uv1);
                float4 i2 = SAMPLE_TEXTURE2D_X(_SSGIIrradianceTexture, sampler_PointClamp, uv2);
                float4 i3 = SAMPLE_TEXTURE2D_X(_SSGIIrradianceTexture, sampler_PointClamp, uv3);
                distances.x = p0.a < 0.5 || i0.a < 0.0 ? 1e9 : distances.x;
                distances.y = p1.a < 0.5 || i1.a < 0.0 ? 1e9 : distances.y;
                distances.z = p2.a < 0.5 || i2.a < 0.0 ? 1e9 : distances.z;
                distances.w = p3.a < 0.5 || i3.a < 0.0 ? 1e9 : distances.w;

                float bestDistance = min(
                    min(distances.x, distances.y),
                    min(distances.z, distances.w));
                if (bestDistance >= 1e8) return 0.0;
                if (bestDistance == distances.x) return i0.rgb;
                if (bestDistance == distances.y) return i1.rgb;
                if (bestDistance == distances.z) return i2.rgb;
                return i3.rgb;
            }

            float4 FragCombine(Varyings input) : SV_Target
            {
                float2 uv = input.texcoord;

                float rawDepth = SampleSceneDepth(uv);
                bool isSky = SSGICombineIsSkyDepth(rawDepth);

                // Camera color is _BlitTexture, set by Blitter.BlitCameraTexture.
                float3 cameraColor = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv).rgb;

                if (isSky)
                    return float4(cameraColor, 1.0);

                float4 positionData = SAMPLE_TEXTURE2D_X(
                    _SSGIGBufferPositionWS, sampler_PointClamp, uv);
                if (positionData.a < 0.5)
                    return float4(cameraColor, 1.0);

                float3 indirectDiffuse = SSGICombineDepthAwareUpsample(uv);
                float3 albedo = SAMPLE_TEXTURE2D_X(
                    _SSGIGBufferAlbedo, sampler_PointClamp, uv).rgb;
                float metallic = SAMPLE_TEXTURE2D_X(
                    _SSGIGBufferMaterial, sampler_PointClamp, uv).r;
                float3 diffuseReflectance =
                    albedo * (1.0 - saturate(metallic));

                float3 color =
                    cameraColor + indirectDiffuse * diffuseReflectance;
                return float4(color, 1.0);
            }
            ENDHLSL
        }

    }
}
