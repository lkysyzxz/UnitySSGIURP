#ifndef GI_COMMOND_H
#define GI_COMMOND_H

#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/UnityInput.hlsl"

float4 _UVToViewPos;

float3 ComputeViewSpacePosition(float2 uv, float depth)
{
    // return ComputeViewSpacePosition(uv, depth, UNITY_MATRIX_I_P);
    float linearEyeDepth = LinearEyeDepth(depth, _ZBufferParams);
#if defined(UNITY_UV_STARTS_AT_TOP)
    // Blit pass texcoord.y=0 is at the TOP of the screen (D3D convention), but the
    // reconstruction formula below assumes a bottom origin. Flip y so that top
    // pixels map to +Y (up) in view space — otherwise viewPos.y is sign-flipped,
    // which sends floor-reflected rays downward instead of upward (no reflection).
    uv.y = 1.0 - uv.y;
#endif
    return float3((uv*_UVToViewPos.xy + _UVToViewPos.zw)*linearEyeDepth, linearEyeDepth);
}

float3 GetNormalFromPosition(float3 position)
{
    return normalize(cross(ddy(position),ddx(position)));
}

#endif
