#ifndef GI_SSGI_BLUR_H
#define GI_SSGI_BLUR_H

float _BlurSpread;

static const float SSGI_FILTER_DEPTH_FALLOFF = 5.0;
static const float2 SSGI_FILTER_OFFSETS[5] =
{
    float2(0.0, 0.0),
    float2(1.0, 0.0),
    float2(-1.0, 0.0),
    float2(0.0, 1.0),
    float2(0.0, -1.0)
};

bool IsSSGIFilterUVValid(float2 uv)
{
    return uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
}

bool IsSSGIFilterDepthValid(float rawDepth)
{
    #if UNITY_REVERSED_Z
        return rawDepth > 0.000001;
    #else
        return rawDepth < 0.999999;
    #endif
}

float3 ReconstructSSGIFilterNormal(float2 uv, float rawDepth)
{
    float3 viewPos = ComputeViewSpacePosition(uv, rawDepth);
    float3 normal = GetNormalFromPosition(viewPos);
    if (dot(normal, -viewPos) <= 0.0)
        normal = -normal;
    return normal;
}

float4 SampleSSGIEdgeAwareFilter(float2 uv, float filterRadius)
{
    float4 center = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv);
    if (center.a < 0.0)
        return center;

    float centerRawDepth = SampleSceneDepth(uv);
    if (!IsSSGIFilterDepthValid(centerRawDepth))
        return center;

    float centerDepth = center.a;
    float3 centerNormal = ReconstructSSGIFilterNormal(uv, centerRawDepth);
    float2 texelSize = 1.0 / _ScreenParams.xy;
    float3 filtered = 0.0;
    float weightSum = 0.0;

    [unroll]
    for (int i = 0; i < 5; i++)
    {
        float2 sampleUV = uv + SSGI_FILTER_OFFSETS[i] * texelSize * filterRadius;
        if (sampleUV.x < 0.0 || sampleUV.x > 1.0 ||
            sampleUV.y < 0.0 || sampleUV.y > 1.0)
            continue;

        float4 sample = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, sampleUV);
        if (sample.a < 0.0)
            continue;

        float sampleRawDepth = SampleSceneDepth(sampleUV);
        if (!IsSSGIFilterDepthValid(sampleRawDepth))
            continue;

        float sampleDepth = sample.a;
        float3 sampleNormal = ReconstructSSGIFilterNormal(sampleUV, sampleRawDepth);
        float depthWeight = exp(-SSGI_FILTER_DEPTH_FALLOFF * abs(centerDepth - sampleDepth));
        float normalWeight = saturate(dot(centerNormal, sampleNormal));
        float weight = depthWeight * normalWeight;

        filtered += sample.rgb * weight;
        weightSum += weight;
    }

    filtered /= max(0.001, weightSum);
    return float4(filtered, center.a);
}

#endif
