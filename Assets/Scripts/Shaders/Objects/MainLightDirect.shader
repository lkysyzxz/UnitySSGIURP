Shader "GIDev/URP/MainLightDirect"
{
    Properties
    {
        _BaseColor ("Base Color", Color) = (1, 1, 1, 1)
        _MainTex ("Base Map", 2D) = "white" {}
    }

    SubShader
    {
        Tags
        {
            "RenderType" = "Opaque"
            "Queue" = "Geometry"
            "RenderPipeline" = "UniversalPipeline"
        }

        Pass
        {
            Name "MainLightDirect"
            Tags { "LightMode" = "UniversalForward" }

            Cull Back
            ZWrite On
            ZTest LEqual

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN
            #pragma multi_compile _ _ADDITIONAL_LIGHTS
            #pragma multi_compile_fragment _ _SHADOWS_SOFT _SHADOWS_SOFT_LOW _SHADOWS_SOFT_MEDIUM _SHADOWS_SOFT_HIGH

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            CBUFFER_START(UnityPerMaterial)
                float4 _BaseColor;
                float4 _MainTex_ST;
            CBUFFER_END

            TEXTURE2D(_MainTex);
            SAMPLER(sampler_MainTex);
            TEXTURE2D_X(_SSGIIrradianceTexture);
            float _SSGIIrradianceValid;
            float _SSGIReprojectIrradiance;
            float _SSGIDisocclusionFallback;
            float _SSGIHistoryDepthThreshold;
            float4x4 _SSGIPreviousViewProjectionMatrix;
            float4x4 _SSGIPreviousWorldToCameraMatrix;

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 normalWS : TEXCOORD0;
                float3 positionWS : TEXCOORD1;
                float2 uv : TEXCOORD2;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings Vert(Attributes input)
            {
                Varyings output;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
                VertexPositionInputs positionInputs = GetVertexPositionInputs(input.positionOS.xyz);

                output.positionCS = positionInputs.positionCS;
                output.normalWS = TransformObjectToWorldNormal(input.normalOS);
                output.positionWS = positionInputs.positionWS;
                output.uv = TRANSFORM_TEX(input.uv, _MainTex);

                return output;
            }

            half4 Frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                half3 normalWS = NormalizeNormalPerPixel(input.normalWS);
                float4 shadowCoord = TransformWorldToShadowCoord(input.positionWS);
                Light mainLight = GetMainLight(shadowCoord);
                half ndotl = saturate(dot(normalWS, mainLight.direction));

                half4 baseColor = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, input.uv) * _BaseColor;
                half3 color = baseColor.rgb * mainLight.color * ndotl * mainLight.shadowAttenuation;

                #if defined(_ADDITIONAL_LIGHTS)
                uint additionalLightCount = GetAdditionalLightsCount();
                LIGHT_LOOP_BEGIN(additionalLightCount)
                    Light additionalLight = GetAdditionalLight(lightIndex, input.positionWS);
                    half additionalNdotL = saturate(dot(normalWS, additionalLight.direction));
                    color += baseColor.rgb * additionalLight.color * additionalNdotL
                        * additionalLight.distanceAttenuation * additionalLight.shadowAttenuation;
                LIGHT_LOOP_END
                #endif

                if (_SSGIIrradianceValid > 0.0)
                {
                    float2 screenUV = GetNormalizedScreenSpaceUV(input.positionCS);
                    bool requiresDepthValidation = _SSGIReprojectIrradiance > 0.5;
                    half3 indirectIrradiance = 0.0;
                    half indirectConfidence = 0.0;

                    if (!requiresDepthValidation)
                    {
                        half4 irradianceDepth = SAMPLE_TEXTURE2D_X(
                            _SSGIIrradianceTexture, sampler_LinearClamp, screenUV);
                        if (irradianceDepth.a >= 0.0)
                        {
                            indirectIrradiance = irradianceDepth.rgb;
                            indirectConfidence = 1.0;
                        }
                    }
                    else
                    {
                        float4 previousCS = mul(
                            _SSGIPreviousViewProjectionMatrix,
                            float4(input.positionWS, 1.0));
                        float2 previousUV = previousCS.xy /
                            max(previousCS.w, 0.000001) * 0.5 + 0.5;
                        bool previousUVValid = previousCS.w > 0.0 &&
                            all(previousUV >= 0.0) && all(previousUV <= 1.0);

                        if (previousUVValid)
                        {
                            half3 reprojectedIrradiance = SAMPLE_TEXTURE2D_X(
                                _SSGIIrradianceTexture,
                                sampler_LinearClamp,
                                previousUV).rgb;
                            float reprojectedDepth = SAMPLE_TEXTURE2D_X(
                                _SSGIIrradianceTexture,
                                sampler_PointClamp,
                                previousUV).a;
                            float expectedDepth = -mul(
                                _SSGIPreviousWorldToCameraMatrix,
                                float4(input.positionWS, 1.0)).z;
                            float depthThreshold = max(
                                _SSGIHistoryDepthThreshold, expectedDepth * 0.01);
                            bool depthValid = reprojectedDepth >= 0.0 &&
                                expectedDepth > 0.0 &&
                                abs(reprojectedDepth - expectedDepth) <= depthThreshold;

                            if (depthValid)
                            {
                                indirectIrradiance = reprojectedIrradiance;
                                indirectConfidence = 1.0;
                            }
                        }

                        if (indirectConfidence <= 0.0 && _SSGIDisocclusionFallback > 0.0)
                        {
                            half4 fallbackIrradiance = SAMPLE_TEXTURE2D_X(
                                _SSGIIrradianceTexture, sampler_LinearClamp, screenUV);
                            float fallbackDepth = SAMPLE_TEXTURE2D_X(
                                _SSGIIrradianceTexture, sampler_PointClamp, screenUV).a;
                            if (fallbackDepth >= 0.0)
                            {
                                indirectIrradiance = fallbackIrradiance.rgb;
                                indirectConfidence = _SSGIDisocclusionFallback;
                            }
                        }
                    }

                    color += indirectIrradiance * baseColor.rgb * indirectConfidence;
                }

                return half4(color, baseColor.a);
            }
            ENDHLSL
        }

        Pass
        {
            Name "ShadowCaster"
            Tags { "LightMode" = "ShadowCaster" }

            ZWrite On
            ZTest LEqual
            ColorMask 0
            Cull Back

            HLSLPROGRAM
            #pragma vertex ShadowPassVertex
            #pragma fragment ShadowPassFragment
            #pragma multi_compile_vertex _ _CASTING_PUNCTUAL_LIGHT_SHADOW

            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/CommonMaterial.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/Shaders/ShadowCasterPass.hlsl"
            ENDHLSL
        }
    }
}
