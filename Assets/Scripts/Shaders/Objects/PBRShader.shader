Shader "GIDev/URP/PBR"
{
    Properties
    {
        [MainTexture] _BaseColorMap ("Albedo Map", 2D) = "white" {}
        [MainColor] _BaseColor ("Albedo", Color) = (1, 1, 1, 1)
        _Roughness ("Roughness", Range(0, 1)) = 0.5
        _Metallic ("Metallic", Range(0, 1)) = 0.0
        [Toggle(_ALPHATEST_ON)] _AlphaCutoffEnable ("Alpha Clipping", Float) = 0.0
        _Cutoff ("Alpha Cutoff", Range(0, 1)) = 0.5
        [Enum(UnityEngine.Rendering.CullMode)] _CullMode ("Cull Mode", Float) = 2.0
    }

    SubShader
    {
        Tags
        {
            "RenderType" = "Opaque"
            "Queue" = "Geometry"
            "RenderPipeline" = "UniversalPipeline"
            "UniversalMaterialType" = "Lit"
        }

        HLSLINCLUDE
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
        #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Color.hlsl"

        TEXTURE2D(_BaseColorMap);
        SAMPLER(sampler_BaseColorMap);

        CBUFFER_START(UnityPerMaterial)
            half4 _BaseColor;
            float4 _BaseColorMap_ST;
            half _Roughness;
            half _Metallic;
            half _AlphaCutoffEnable;
            half _Cutoff;
        CBUFFER_END

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
            float3 positionWS : TEXCOORD0;
            half3 normalWS : TEXCOORD1;
            float2 uv : TEXCOORD2;
            UNITY_VERTEX_INPUT_INSTANCE_ID
            UNITY_VERTEX_OUTPUT_STEREO
        };

        Varyings LitVertex(Attributes input)
        {
            Varyings output = (Varyings)0;
            UNITY_SETUP_INSTANCE_ID(input);
            UNITY_TRANSFER_INSTANCE_ID(input, output);
            UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);

            VertexPositionInputs positionInputs = GetVertexPositionInputs(input.positionOS.xyz);
            VertexNormalInputs normalInputs = GetVertexNormalInputs(input.normalOS);
            output.positionCS = positionInputs.positionCS;
            output.positionWS = positionInputs.positionWS;
            output.normalWS = normalInputs.normalWS;
            output.uv = TRANSFORM_TEX(input.uv, _BaseColorMap);
            return output;
        }

        half4 SampleBaseColor(float2 uv)
        {
            return SAMPLE_TEXTURE2D(_BaseColorMap, sampler_BaseColorMap, uv) * _BaseColor;
        }

        void ApplyAlphaClip(half alpha)
        {
            #if defined(_ALPHATEST_ON)
                clip(alpha - _Cutoff);
            #endif
        }

        void InitializePBRSurfaceData(half4 baseColor, out SurfaceData surfaceData)
        {
            surfaceData = (SurfaceData)0;
            surfaceData.albedo = baseColor.rgb;
            surfaceData.metallic = saturate(_Metallic);
            surfaceData.specular = half3(0.0h, 0.0h, 0.0h);
            surfaceData.smoothness = 1.0h - saturate(_Roughness);
            surfaceData.normalTS = half3(0.0h, 0.0h, 1.0h);
            surfaceData.emission = half3(0.0h, 0.0h, 0.0h);
            surfaceData.occlusion = 1.0h;
            surfaceData.alpha = baseColor.a;
            surfaceData.clearCoatMask = 0.0h;
            surfaceData.clearCoatSmoothness = 0.0h;
        }

        void InitializePBRInputData(Varyings input, out InputData inputData)
        {
            inputData = (InputData)0;
            inputData.positionWS = input.positionWS;
            inputData.positionCS = input.positionCS;
            inputData.normalWS = NormalizeNormalPerPixel(input.normalWS);
            inputData.viewDirectionWS = GetWorldSpaceNormalizeViewDir(input.positionWS);
            inputData.shadowCoord = TransformWorldToShadowCoord(input.positionWS);
            inputData.vertexLighting = VertexLighting(input.positionWS, inputData.normalWS);
            inputData.normalizedScreenSpaceUV = GetNormalizedScreenSpaceUV(input.positionCS);
            inputData.shadowMask = half4(1.0h, 1.0h, 1.0h, 1.0h);
        }

        half3 CalculateDirectPBRLighting(InputData inputData, SurfaceData surfaceData)
        {
            BRDFData brdfData;
            InitializeBRDFData(surfaceData, brdfData);

            BRDFData brdfDataClearCoat = CreateClearCoatBRDFData(surfaceData, brdfData);
            half4 shadowMask = half4(1.0h, 1.0h, 1.0h, 1.0h);
            AmbientOcclusionFactor aoFactor = CreateAmbientOcclusionFactor(inputData, surfaceData);
            uint meshRenderingLayers = GetMeshRenderingLayer();
            Light mainLight = GetMainLight(inputData, shadowMask, aoFactor);
            half3 directLighting = 0.0h;

            #ifdef _LIGHT_LAYERS
            if (IsMatchingLightLayer(mainLight.layerMask, meshRenderingLayers))
            #endif
            {
                directLighting += LightingPhysicallyBased(
                    brdfData,
                    brdfDataClearCoat,
                    mainLight,
                    inputData.normalWS,
                    inputData.viewDirectionWS,
                    surfaceData.clearCoatMask,
                    false);
            }

            #if defined(_ADDITIONAL_LIGHTS)
            uint pixelLightCount = GetAdditionalLightsCount();

            #if USE_FORWARD_PLUS
            for (uint lightIndex = 0;
                 lightIndex < min(URP_FP_DIRECTIONAL_LIGHTS_COUNT, MAX_VISIBLE_LIGHTS);
                 lightIndex++)
            {
                FORWARD_PLUS_SUBTRACTIVE_LIGHT_CHECK
                Light light = GetAdditionalLight(lightIndex, inputData, shadowMask, aoFactor);

                #ifdef _LIGHT_LAYERS
                if (IsMatchingLightLayer(light.layerMask, meshRenderingLayers))
                #endif
                {
                    directLighting += LightingPhysicallyBased(
                        brdfData,
                        brdfDataClearCoat,
                        light,
                        inputData.normalWS,
                        inputData.viewDirectionWS,
                        surfaceData.clearCoatMask,
                        false);
                }
            }
            #endif

            LIGHT_LOOP_BEGIN(pixelLightCount)
                Light light = GetAdditionalLight(lightIndex, inputData, shadowMask, aoFactor);

                #ifdef _LIGHT_LAYERS
                if (IsMatchingLightLayer(light.layerMask, meshRenderingLayers))
                #endif
                {
                    directLighting += LightingPhysicallyBased(
                        brdfData,
                        brdfDataClearCoat,
                        light,
                        inputData.normalWS,
                        inputData.viewDirectionWS,
                        surfaceData.clearCoatMask,
                        false);
                }
            LIGHT_LOOP_END
            #endif

            #if defined(_ADDITIONAL_LIGHTS_VERTEX)
            directLighting += inputData.vertexLighting * brdfData.diffuse;
            #endif

            return directLighting;
        }

        half4 LitFragment(Varyings input) : SV_Target
        {
            UNITY_SETUP_INSTANCE_ID(input);
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

            half4 baseColor = SampleBaseColor(input.uv);
            ApplyAlphaClip(baseColor.a);

            SurfaceData surfaceData;
            InitializePBRSurfaceData(baseColor, surfaceData);

            InputData inputData;
            InitializePBRInputData(input, inputData);

            return half4(CalculateDirectPBRLighting(inputData, surfaceData), surfaceData.alpha);
        }

        struct ForwardGBufferOutput
        {
            half4 albedo : SV_Target0;
            half4 material : SV_Target1;
            half4 normalWS : SV_Target2;
            float4 positionWS : SV_Target3;
        };

        ForwardGBufferOutput ForwardGBufferFragment(Varyings input)
        {
            UNITY_SETUP_INSTANCE_ID(input);
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

            half4 baseColor = SampleBaseColor(input.uv);
            ApplyAlphaClip(baseColor.a);

            ForwardGBufferOutput output;
            output.albedo = baseColor;
            output.material = half4(saturate(_Metallic), saturate(_Roughness), 0.0h, 1.0h);
            output.normalWS = half4(NormalizeNormalPerPixel(input.normalWS), 1.0h);
            output.positionWS = float4(input.positionWS, 1.0);
            return output;
        }
        ENDHLSL

        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode" = "UniversalForward" }

            Cull [_CullMode]
            ZWrite On
            ZTest LEqual

            HLSLPROGRAM
            #pragma target 3.5
            #pragma vertex LitVertex
            #pragma fragment LitFragment

            #pragma shader_feature_local_fragment _ALPHATEST_ON

            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN
            #pragma multi_compile _ _ADDITIONAL_LIGHTS_VERTEX _ADDITIONAL_LIGHTS
            #pragma multi_compile_fragment _ _ADDITIONAL_LIGHT_SHADOWS
            #pragma multi_compile_fragment _ _SHADOWS_SOFT _SHADOWS_SOFT_LOW _SHADOWS_SOFT_MEDIUM _SHADOWS_SOFT_HIGH
            #pragma multi_compile_fragment _ _LIGHT_COOKIES
            #pragma multi_compile _ _FORWARD_PLUS
            #pragma multi_compile_instancing
            ENDHLSL
        }

        Pass
        {
            Name "ForwardGBuffer"
            Tags { "LightMode" = "SSGIForwardGBuffer" }

            Cull [_CullMode]
            // ForwardGBufferPass overrides this to ZWrite Off + Equal when it
            // reuses camera depth, and to ZWrite On + LEqual for its fallback
            // depth target. Keep the shader's standalone state compatible with
            // the fallback path so MRT occlusion never depends on draw order.
            ZWrite On
            ZTest LEqual

            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex LitVertex
            #pragma fragment ForwardGBufferFragment
            #pragma shader_feature_local_fragment _ALPHATEST_ON
            #pragma multi_compile_instancing
            ENDHLSL
        }

        Pass
        {
            Name "ShadowCaster"
            Tags { "LightMode" = "ShadowCaster" }

            ZWrite On
            ZTest LEqual
            ColorMask 0
            Cull [_CullMode]

            HLSLPROGRAM
            #pragma target 3.5
            #pragma vertex ShadowVertex
            #pragma fragment ShadowFragment
            #pragma shader_feature_local_fragment _ALPHATEST_ON
            #pragma multi_compile_vertex _ _CASTING_PUNCTUAL_LIGHT_SHADOW
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Shadows.hlsl"

            float3 _LightDirection;
            float3 _LightPosition;

            struct ShadowVaryings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
                UNITY_VERTEX_OUTPUT_STEREO
            };

            ShadowVaryings ShadowVertex(Attributes input)
            {
                ShadowVaryings output = (ShadowVaryings)0;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_TRANSFER_INSTANCE_ID(input, output);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);

                float3 positionWS = TransformObjectToWorld(input.positionOS.xyz);
                float3 normalWS = TransformObjectToWorldNormal(input.normalOS);

                #if defined(_CASTING_PUNCTUAL_LIGHT_SHADOW)
                    float3 lightDirectionWS = normalize(_LightPosition - positionWS);
                #else
                    float3 lightDirectionWS = _LightDirection;
                #endif

                output.positionCS = TransformWorldToHClip(
                    ApplyShadowBias(positionWS, normalWS, lightDirectionWS));

                #if UNITY_REVERSED_Z
                    output.positionCS.z = min(output.positionCS.z, UNITY_NEAR_CLIP_VALUE);
                #else
                    output.positionCS.z = max(output.positionCS.z, UNITY_NEAR_CLIP_VALUE);
                #endif

                output.uv = TRANSFORM_TEX(input.uv, _BaseColorMap);

                return output;
            }

            half4 ShadowFragment(ShadowVaryings input) : SV_Target
            {
                UNITY_SETUP_INSTANCE_ID(input);
                ApplyAlphaClip(SampleBaseColor(input.uv).a);
                return 0.0h;
            }
            ENDHLSL
        }
    }

    FallBack "Hidden/Universal Render Pipeline/FallbackError"
}
