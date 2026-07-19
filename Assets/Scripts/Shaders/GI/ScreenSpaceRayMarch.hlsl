#ifndef GI_SCREEN_SPACE_RAY_MARCH_H
#define GI_SCREEN_SPACE_RAY_MARCH_H

#include "Jitter.hlsl"
#include "HiZUtility.hlsl"

#define SCREEN_SPACE_RAY_BINARY_STEPS 10
#define SCREEN_SPACE_RAY_NEAR_EPSILON 0.0001

struct ScreenSpaceRayHit
{
    bool   hit;
    float2 hitUV;
    float  rayT;
    float3 hitPosVS;
};

float2 ProjectVStoUV(float3 vsPos)
{
    float4 clip = mul(UNITY_MATRIX_P, float4(vsPos.xy, -vsPos.z, 1.0));
    float2 uv = (clip.xy / clip.w) * 0.5 + 0.5;
    return uv;
}

bool ClipScreenSpaceRayAxis(float origin, float direction, float boundMin, float boundMax,
                            inout float tEnter, inout float tExit)
{
    if (direction == 0.0)
        return origin >= boundMin && origin <= boundMax;

    float t0 = (boundMin - origin) / direction;
    float t1 = (boundMax - origin) / direction;
    float axisEnter = min(t0, t1);
    float axisExit = max(t0, t1);
    tEnter = max(tEnter, axisEnter);
    tExit = min(tExit, axisExit);
    return tEnter <= tExit;
}

bool ClipScreenSpaceRayToViewport(inout float2 startUV, inout float2 endUV,
                                  inout float invZ0, inout float invZ1)
{
    float2 originalStartUV = startUV;
    float2 deltaUV = endUV - startUV;
    float originalInvZ0 = invZ0;
    float originalInvZ1 = invZ1;
    float tEnter = 0.0;
    float tExit = 1.0;
    float2 halfTexel = 0.5 / _ScreenParams.xy;
    float2 viewportMin = halfTexel;
    float2 viewportMax = 1.0 - halfTexel;

    if (!ClipScreenSpaceRayAxis(originalStartUV.x, deltaUV.x, viewportMin.x, viewportMax.x, tEnter, tExit))
        return false;
    if (!ClipScreenSpaceRayAxis(originalStartUV.y, deltaUV.y, viewportMin.y, viewportMax.y, tEnter, tExit))
        return false;

    startUV = originalStartUV + deltaUV * tEnter;
    endUV = originalStartUV + deltaUV * tExit;
    invZ0 = lerp(originalInvZ0, originalInvZ1, tEnter);
    invZ1 = lerp(originalInvZ0, originalInvZ1, tExit);
    return true;
}

bool ClipScreenSpaceRaySegment(inout float3 rayStartVS, inout float3 rayEndVS)
{
    float nearPlane = max(_ProjectionParams.y, SCREEN_SPACE_RAY_NEAR_EPSILON);
    if (rayStartVS.z <= nearPlane)
        return false;

    float3 ray = rayEndVS - rayStartVS;
    if (dot(ray, ray) <= SCREEN_SPACE_RAY_NEAR_EPSILON * SCREEN_SPACE_RAY_NEAR_EPSILON)
        return false;

    if (rayEndVS.z <= nearPlane)
    {
        float clipT = (rayStartVS.z - nearPlane) /
                      (rayStartVS.z - rayEndVS.z);
        rayEndVS = lerp(rayStartVS, rayEndVS, clipT);
        rayEndVS.z = nearPlane;
    }

    return true;
}

float ScreenSpaceRayT(float3 rayStartVS, float3 rayEndVS, float3 hitPosVS)
{
    float3 ray = rayEndVS - rayStartVS;
    return saturate(dot(hitPosVS - rayStartVS, ray) / max(dot(ray, ray), 0.000001));
}

// DDA traversal with local refinement at the first depth crossing.
ScreenSpaceRayHit MarchScreenSpaceRayDDA(float3 rayStartVS, float3 rayEndVS, int maxSteps, float thickness)
{
    ScreenSpaceRayHit result = (ScreenSpaceRayHit)0;
    if (!ClipScreenSpaceRaySegment(rayStartVS, rayEndVS))
        return result;

    float2 startUV = ProjectVStoUV(rayStartVS);
    float2 endUV   = ProjectVStoUV(rayEndVS);

    float2 deltaUV      = endUV - startUV;
    float2 deltaPixel   = deltaUV * _ScreenParams.xy;
    float  maxPixelSpan = max(abs(deltaPixel.x), abs(deltaPixel.y));
    int    safeMaxSteps = clamp(maxSteps, 1, 256);
    int    numSteps     = (int)clamp(maxPixelSpan, 1.0, (float)safeMaxSteps);

    float  invZ0    = 1.0 / rayStartVS.z;
    float  invZ1    = 1.0 / rayEndVS.z;
    float2 stepUV   = deltaUV / (float)numSteps;
    float  invZStep = (invZ1 - invZ0) / (float)numSteps;

    float2 prevUV      = startUV;
    float  prevInvZ    = invZ0;
    float2 currentUV   = startUV;
    float  currentInvZ = invZ0;
    float  stepPosition = 0.0;
    bool   hasFrontSample = false;
    bool   crossed     = false;

    if (currentUV.x >= 0.0 && currentUV.x <= 1.0 &&
        currentUV.y >= 0.0 && currentUV.y <= 1.0)
    {
        float startSceneZ = LinearEyeDepth(SampleSceneDepth(currentUV), _ZBufferParams);
        hasFrontSample = (1.0 / currentInvZ) <= startSceneZ;
    }

    #if defined(_JITTER_ON)
    {
        float jit = GetJitter(startUV);
        currentUV   += stepUV * jit;
        currentInvZ += invZStep * jit;
        stepPosition = jit;
    }
    #endif

    [loop]
    for (int i = 0; i < numSteps; i++)
    {
        float advance = min(1.0, (float)numSteps - stepPosition);
        if (advance <= 0.0)
            break;
        currentUV   += stepUV * advance;
        currentInvZ += invZStep * advance;
        stepPosition += advance;

        if (currentUV.x < 0.0 || currentUV.x > 1.0 ||
            currentUV.y < 0.0 || currentUV.y > 1.0)
            break;

        float sceneZ = LinearEyeDepth(SampleSceneDepth(currentUV), _ZBufferParams);
        if ((1.0 / currentInvZ) > sceneZ)
        {
            if (hasFrontSample)
            {
                crossed = true;
                break;
            }
        }
        else
        {
            prevUV = currentUV;
            prevInvZ = currentInvZ;
            hasFrontSample = true;
        }
    }

    if (!crossed)
        return result;

    float2 loUV   = prevUV;
    float  loInvZ = prevInvZ;
    float2 hiUV   = currentUV;
    float  hiInvZ = currentInvZ;

    [loop]
    for (int j = 0; j < SCREEN_SPACE_RAY_BINARY_STEPS; j++)
    {
        float2 midUV     = (loUV + hiUV) * 0.5;
        float  midInvZ   = (loInvZ + hiInvZ) * 0.5;
        float  midSceneZ = LinearEyeDepth(SampleSceneDepth(midUV), _ZBufferParams);

        if ((1.0 / midInvZ) > midSceneZ)
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

    float rawDepth = SampleSceneDepth(hiUV);
    float sceneZ = LinearEyeDepth(rawDepth, _ZBufferParams);
    float depthDiff = (1.0 / hiInvZ) - sceneZ;
    if (depthDiff > 0.0 && depthDiff < thickness)
    {
        result.hit = true;
        result.hitUV = hiUV;
        result.hitPosVS = ComputeViewSpacePosition(hiUV, rawDepth);
        result.rayT = saturate(ScreenSpaceRayT(rayStartVS, rayEndVS, result.hitPosVS));
    }
    return result;
}

// DDA coarse search followed by binary refinement.
ScreenSpaceRayHit MarchScreenSpaceRayBinary(float3 rayStartVS, float3 rayEndVS,
                                            int maxSteps, float thickness, int binarySteps)
{
    ScreenSpaceRayHit result = (ScreenSpaceRayHit)0;

    float2 startUV = ProjectVStoUV(rayStartVS);
    float2 endUV   = ProjectVStoUV(rayEndVS);

    float invZ0 = 1.0 / rayStartVS.z;
    float invZ1 = 1.0 / rayEndVS.z;
    // if (!ClipScreenSpaceRayToViewport(startUV, endUV, invZ0, invZ1))
    //     return result;

    float2 deltaUV      = endUV - startUV;
    float2 deltaPixel   = deltaUV * _ScreenParams.xy;
    float  maxPixelSpan = max(abs(deltaPixel.x), abs(deltaPixel.y));
    int    safeMaxSteps = clamp(maxSteps, 1, 256);
    int    numSteps     = maxPixelSpan < 1.0
        ? min(2, safeMaxSteps)
        : (int)clamp(ceil(maxPixelSpan), 1.0, (float)safeMaxSteps);

    float2 stepUV   = deltaUV / (float)numSteps;
    float  invZStep = (invZ1 - invZ0) / (float)numSteps;
    float  firstStepScale = maxPixelSpan < 1.0 && numSteps == 1 ? 0.5 : 1.0;

    float2 prevUV      = startUV;
    float  prevInvZ    = invZ0;
    float2 currentUV   = startUV;
    float  currentInvZ = invZ0;
    bool   crossed     = false;

    #if defined(_JITTER_ON)
    if (maxPixelSpan >= 1.0)
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
        float advance = i == 0 ? firstStepScale : 1.0;
        currentUV   += stepUV * advance;
        currentInvZ += invZStep * advance;

        if (currentUV.x < 0.0 || currentUV.x > 1.0 ||
            currentUV.y < 0.0 || currentUV.y > 1.0)
            break;

        float sceneZ = LinearEyeDepth(SampleSceneDepth(currentUV), _ZBufferParams);
        if ((1.0 / currentInvZ) > sceneZ)
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
        float  midSceneZ = LinearEyeDepth(SampleSceneDepth(midUV), _ZBufferParams);

        if ((1.0 / midInvZ) > midSceneZ)
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

    float rawDepth = SampleSceneDepth(hiUV);
    float sceneZ = LinearEyeDepth(rawDepth, _ZBufferParams);
    float depthDiff = (1.0 / hiInvZ) - sceneZ;
    if (depthDiff > 0.0 && depthDiff < thickness)
    {
        result.hit = true;
        result.hitUV = hiUV;
        result.hitPosVS = ComputeViewSpacePosition(hiUV, rawDepth);
        result.rayT = saturate(ScreenSpaceRayT(rayStartVS, rayEndVS, result.hitPosVS));
    }
    return result;
}

// HiZ adaptive stepping: skip empty regions at high mip and refine near intersections.
ScreenSpaceRayHit MarchScreenSpaceRayHiZ(float3 rayStartVS, float3 rayEndVS, int maxSteps, float thickness)
{
    ScreenSpaceRayHit result = (ScreenSpaceRayHit)0;
    if (!ClipScreenSpaceRaySegment(rayStartVS, rayEndVS))
        return result;

    float2 startUV = ProjectVStoUV(rayStartVS);
    float2 endUV   = ProjectVStoUV(rayEndVS);

    float2 startPos = startUV * _ScreenParams.xy;
    float2 endPos = endUV * _ScreenParams.xy;
    float totalDist = distance(endPos, startPos);
    if (totalDist < 1.0) return result;

    float2 rayDir2D = (endPos - startPos) / totalDist;
    float invZ0 = 1.0 / rayStartVS.z;
    float invZ1 = 1.0 / rayEndVS.z;
    float2 pos = startPos;
    float traveled = 0.0;
    int safeMaxSteps = clamp(maxSteps, 1, 256);
    float maxMip = _HiZMaxMip;
    float mipLevel = maxMip * 0.5;

    #if defined(_JITTER_ON)
    {
        float jit = GetJitter(startUV);
        pos += rayDir2D * jit;
        traveled += jit;
    }
    #endif

    [loop]
    for (int i = 0; i < safeMaxSteps; i++)
    {
        float stride = exp2(mipLevel);
        if (traveled + stride > totalDist) break;
        pos += rayDir2D * stride;
        traveled += stride;

        float2 uv = pos / _ScreenParams.xy;
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
            break;

        float t = traveled / totalDist;
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
                    rawDepth = SampleSceneDepth(uv);
                    result.hit = true;
                    result.hitUV = uv;
                    result.hitPosVS = ComputeViewSpacePosition(uv, rawDepth);
                    result.rayT = saturate(ScreenSpaceRayT(rayStartVS, rayEndVS, result.hitPosVS));
                }
                break;
            }
            pos -= rayDir2D * stride;
            traveled -= stride;
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
