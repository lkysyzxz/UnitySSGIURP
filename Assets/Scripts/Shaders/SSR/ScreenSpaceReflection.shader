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
            #include "../GI/ScreenSpaceRayMarch.hlsl"

            float _SSRMaxDistance;
            int   _SSRMaxSteps;
            float _SSRThickness;

            float4 FragSSR(Varyings input) : SV_Target
            {
                float2 uv = input.texcoord;

                float rawDepth = SampleSceneDepth(uv);
                #if UNITY_REVERSED_Z
                    if (rawDepth <= 0.000001) return float4(0, 0, 0, 0); // sky
                #else
                    if (rawDepth >= 0.999999) return float4(0, 0, 0, 0);
                #endif

                float3 viewPos = ComputeViewSpacePosition(uv, rawDepth);
                float3 normal  = GetNormalFromPosition(viewPos);
                float3 viewDir = normalize(viewPos);
                float3 rayDir  = reflect(viewDir, normal);

                if (rayDir.z < 0.0) return float4(0, 0, 0, 0); // 朝相机

                float3 rayStartVS = viewPos;
                float3 rayEndVS   = viewPos + rayDir * _SSRMaxDistance;

                ScreenSpaceRayHit hit = MarchScreenSpaceRayBinary(rayStartVS, rayEndVS, _SSRMaxSteps, _SSRThickness, SCREEN_SPACE_RAY_BINARY_STEPS);
                // ScreenSpaceRayHit hit = MarchScreenSpaceRayDDA(rayStartVS, rayEndVS, _SSRMaxSteps, _SSRThickness);
                // ScreenSpaceRayHit hit = MarchScreenSpaceRayHiZ(rayStartVS, rayEndVS, _SSRMaxSteps, _SSRThickness);

                // RGB=反射色, A=混合系数（hit=0.5, miss=0）
                if (hit.hit)
                    return float4(SampleSceneColor(hit.hitUV), 0.5);
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
            #include "../GI/Blur.hlsl"

            float4 FragBlurH(Varyings input) : SV_Target
            {
                return SampleGaussianBlurHorizontal(input.texcoord, _BlurSpread);
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
            #include "../GI/Blur.hlsl"

            float4 FragBlurV(Varyings input) : SV_Target
            {
                return SampleGaussianBlurVertical(input.texcoord, _BlurSpread);
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
