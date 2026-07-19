#ifndef GI_TEMPORAL_H
#define GI_TEMPORAL_H

TEXTURE2D_X(_MotionVectorTexture);
TEXTURE2D_X(_SSGIHistoryTexture);

float _SSGIHistoryValid;
float _SSGIHistoryWeight;
float _SSGIHistoryDepthThreshold;
float _SSGIMaxHistoryFrames;
float _SSGIHistoryFrameCount;
float4x4 _SSGIPreviousWorldToCameraMatrix;

struct SSGITemporalHistory
{
    bool valid;
    float2 previousUV;
    float4 irradianceDepth;
};

bool IsSSGIRawDepthValid(float rawDepth)
{
    #if UNITY_REVERSED_Z
        return rawDepth > 0.000001;
    #else
        return rawDepth < 0.999999;
    #endif
}

bool IsSSGIUVValid(float2 uv)
{
    return all(uv >= 0.0) && all(uv <= 1.0);
}

SSGITemporalHistory GetSSGITemporalHistory(float2 uv, float rawDepth)
{
    SSGITemporalHistory result;
    result.valid = false;
    result.previousUV = uv;
    result.irradianceDepth = 0.0;

    if (_SSGIHistoryValid <= 0.0 || !IsSSGIRawDepthValid(rawDepth))
        return result;

    float2 motion = SAMPLE_TEXTURE2D_X(_MotionVectorTexture, sampler_LinearClamp, uv).xy;
    float2 previousUV = uv - motion;
    result.previousUV = previousUV;
    if (!IsSSGIUVValid(previousUV))
        return result;

    float4 history = SAMPLE_TEXTURE2D_X(_SSGIHistoryTexture, sampler_LinearClamp, previousUV);
    float historyDepth = SAMPLE_TEXTURE2D_X(
        _SSGIHistoryTexture, sampler_PointClamp, previousUV).a;
    if (historyDepth < 0.0)
        return result;

    float3 currentPositionWS = ComputeWorldSpacePosition(uv, rawDepth, UNITY_MATRIX_I_VP);
    float previousExpectedDepth = -mul(
        _SSGIPreviousWorldToCameraMatrix, float4(currentPositionWS, 1.0)).z;
    float depthThreshold = max(
        _SSGIHistoryDepthThreshold, max(previousExpectedDepth, 0.0) * 0.01);
    if (previousExpectedDepth <= 0.0 ||
        abs(historyDepth - previousExpectedDepth) > depthThreshold)
        return result;

    result.valid = true;
    history.a = historyDepth;
    result.irradianceDepth = history;
    return result;
}

float GetSSGICurrentFrameWeight()
{
    float runningAverageWeight = rcp(_SSGIHistoryFrameCount + 1.0);
    return _SSGIHistoryFrameCount < _SSGIMaxHistoryFrames
        ? runningAverageWeight
        : _SSGIHistoryWeight;
}

#endif
