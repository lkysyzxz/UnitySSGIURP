#ifndef GI_POST_PROCESS_INPUT_H
#define GI_POST_PROCESS_INPUT_H

#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"

struct appdata
{
    float4 vertex : POSITION;
    float2 uv : TEXCOORD0;
};

struct v2f
{
    float2 uv : TEXCOORD0;
    float4 vertex : SV_POSITION;
};

v2f vert (appdata v)
{
    v2f o;
    o.vertex = TransformObjectToHClip(v.vertex);
    o.uv = v.uv;
    return o;
}

// --------------------------------------------------------------------------
// Texture Declarations
// --------------------------------------------------------------------------
TEXTURE2D(_CameraDepthAttachment);
SAMPLER(sampler_CameraDepthAttachment);
float4 _CameraDepthAttachment_TexelSize;



#endif
