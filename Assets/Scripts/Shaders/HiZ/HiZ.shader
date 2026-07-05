Shader "Hidden/HiZ"
{
    Properties
    {
        _MainTex ("Source", 2D) = "white" {}
    }

    SubShader
    {
        Tags { "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" }

        // ============================================================
        // Pass 0: CopyDepth
        //   从 _CameraDepthTexture 采样 raw depth，写入 _HiZTexture mip 0
        // ============================================================
        Pass
        {
            Name "CopyDepth"
            Cull Off
            ZWrite Off
            ZTest Always

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment FragCopyDepth

            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"

            float4 FragCopyDepth(Varyings input) : SV_Target
            {
                return SampleSceneDepth(input.texcoord).rrrr;   // raw depth（reversed-z: 近=1 远=0）
            }
            ENDHLSL
        }

        // ============================================================
        // Pass 1: HiZDownsample
        //   从源纹理的 _HiZSrcMip 层采样 2x2 像素，取最小值（reversed-z
        //   下 min = 最远表面），写入下一 mip 层。
        // ============================================================
        Pass
        {
            Name "HiZDownsample"
            Cull Off
            ZWrite Off
            ZTest Always

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment FragHiZDownsample

            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"

            float4 _HiZSrcSize;   // Feature 传入的源纹理尺寸（Blitter 不设 _BlitTextureSize）

            float4 FragHiZDownsample(Varyings input) : SV_Target
            {
                float2 uv = input.texcoord;
                float2 o = (1.0 / _HiZSrcSize.xy) * 0.5;

                // 采样源（上一层独立 RT）的 2x2 块，取 max
                // reversed-z 下 max raw depth = 最近表面（用于遮挡判断：射线穿过最近表面 → 命中）
                float d0 = _BlitTexture.Sample(sampler_PointClamp, uv + float2(-o.x, -o.y)).r;
                float d1 = _BlitTexture.Sample(sampler_PointClamp, uv + float2( o.x, -o.y)).r;
                float d2 = _BlitTexture.Sample(sampler_PointClamp, uv + float2(-o.x,  o.y)).r;
                float d3 = _BlitTexture.Sample(sampler_PointClamp, uv + float2( o.x,  o.y)).r;

                return max(max(d0, d1), max(d2, d3)).rrrr;
            }
            ENDHLSL
        }
    }
}
