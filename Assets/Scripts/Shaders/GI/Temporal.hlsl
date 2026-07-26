#ifndef GI_TEMPORAL_H
#define GI_TEMPORAL_H

TEXTURE2D_X(_MotionVectorTexture);
TEXTURE2D_X(_SSGIHistoryTexture);
TEXTURE2D_X(_SSGIHistorySampleTexture);
TEXTURE2D_X(_SSGIGBufferPositionWS);

float _SSGIHistoryValid;
float _SSGIHistoryWeight;
float _SSGIHistoryDepthThreshold;
float _SSGIMaxHistoryFrames;
float4x4 _SSGIPreviousWorldToCameraMatrix;

struct SSGITemporalHistory
{
    bool valid;
    float2 previousUV;
    float2 motion;
    float4 irradianceDepth;
    float sampleCount;
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
    result.motion = 0.0;
    result.irradianceDepth = 0.0;
    result.sampleCount = 0.0;

    if (_SSGIHistoryValid <= 0.0 || !IsSSGIRawDepthValid(rawDepth))
        return result;

    float2 motion = SAMPLE_TEXTURE2D_X(
        _MotionVectorTexture, sampler_LinearClamp, uv).xy;
    float2 previousUV = uv - motion;
    result.previousUV = previousUV;
    result.motion = motion;
    if (!IsSSGIUVValid(previousUV))
        return result;

    float4 history = SAMPLE_TEXTURE2D_X(
        _SSGIHistoryTexture, sampler_LinearClamp, previousUV);
    float historyDepth = SAMPLE_TEXTURE2D_X(
        _SSGIHistoryTexture, sampler_PointClamp, previousUV).a;
    float historySampleCount = SAMPLE_TEXTURE2D_X(
        _SSGIHistorySampleTexture, sampler_PointClamp, previousUV).r;
    if (historyDepth <= 0.0 || historySampleCount < 0.5)
        return result;

    float4 currentPosition = SAMPLE_TEXTURE2D_X(
        _SSGIGBufferPositionWS, sampler_PointClamp, uv);
    if (currentPosition.a < 0.5)
        return result;

    // History alpha is previous-frame view depth. Compare it against the
    // current surface transformed into the previous camera, rather than the
    // current-frame view depth, so camera motion can retain valid history.
    float expectedHistoryDepth = -mul(
        _SSGIPreviousWorldToCameraMatrix,
        float4(currentPosition.xyz, 1.0)).z;
    float depthThreshold = max(
        _SSGIHistoryDepthThreshold,
        max(expectedHistoryDepth, historyDepth) * 0.01);
    if (expectedHistoryDepth <= 0.0 ||
        abs(historyDepth - expectedHistoryDepth) > depthThreshold)
        return result;

    result.valid = true;
    history.a = historyDepth;
    result.irradianceDepth = history;
    result.sampleCount = historySampleCount;
    return result;
}

#endif
