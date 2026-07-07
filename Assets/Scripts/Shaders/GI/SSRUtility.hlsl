#ifndef GI_SSR_UTILITY_H
#define GI_SSR_UTILITY_H

#include "Jitter.hlsl"
#include "HiZUtility.hlsl"

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

// DDA direct hit test.
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

// DDA coarse search followed by binary refinement.
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

// HiZ adaptive stepping: skip empty regions at high mip and refine near intersections.
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
    float  mipLevel = maxMip * 0.5;

    #if defined(_JITTER_ON)
    {
        float jit = GetJitter(startUV);
        pos += rayDir2D * jit;
    }
    #endif

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

        if (rayZ > sceneZ)
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

#endif
