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

                // 4. Ray march in view space
                float3 rayPos = viewPos;
                float stepSize = _SSRStepSize;
                float3 hitColor = float3(0, 0, 0);
                bool hit = false;
                
                [loop]
                for (int i = 0; i < _SSRMaxSteps; i++)
                {
                    // Advance ray
                    rayPos += rayDir * stepSize;

                    // Clamp ray travel distance
                    // if (length(rayPos - viewPos) > _SSRMaxDistance)
                    //     break;

                    // Project view-space position to UV.
                    // We use positive-Z internally, but UNITY_MATRIX_P expects negative Z
                    // for points in front of the camera. Negate Z here so clip.w comes
                    // out positive (otherwise the NDC x/y get flipped -> mirror reflection).
                    float4 clipPos = mul(UNITY_MATRIX_P, float4(rayPos.xy, -rayPos.z, 1.0));
                    float2 rayUV = clipPos.xy / clipPos.w;
                    rayUV = rayUV * 0.5 + 0.5; // NDC [-1,1] -> UV [0,1]

                    // Check if ray went off screen
                    if (rayUV.x < 0.0 || rayUV.x > 1.0 || rayUV.y < 0.0 || rayUV.y > 1.0)
                        break;

                    // Sample depth at ray UV and reconstruct scene view position
                    float sceneDepth = SampleSceneDepth(rayUV);
                    float3 sceneViewPos = ComputeViewSpacePosition(rayUV, sceneDepth);

                    // Depth comparison (positive-Z convention: larger z = further from camera).
                    // rayPos.z > sceneViewPos.z => ray is further than scene geometry => behind it.
                    float depthDiff = rayPos.z - sceneViewPos.z;
                    if (depthDiff > 0.0 && abs(depthDiff) < _SSRThickness)
                    {
                        // HIT - sample the opaque color at hit UV
                        hitColor = SampleSceneColor(rayUV);
                        hit = true;

                        // // Fade based on iteration count (early hits are more reliable)
                        // float iterationFade = 1.0 - (float)i / (float)_SSRMaxSteps;
                        // // Fade based on edge proximity
                        // float2 edgeFade = smoothstep(0.0, 0.1, rayUV) * smoothstep(0.0, 0.1, 1.0 - rayUV);
                        // float edgeFadeFactor = edgeFade.x * edgeFade.y;
                        //
                        // hitColor *= iterationFade * edgeFadeFactor;
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
