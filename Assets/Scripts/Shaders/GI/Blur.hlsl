#ifndef GI_BLUR_H
#define GI_BLUR_H

float _BlurSpread;
// 5-tap binomial weights [1,4,6,4,1]/16.
static const float GI_BLUR_WEIGHT_0 = 0.375;
static const float GI_BLUR_WEIGHT_1 = 0.25;
static const float GI_BLUR_WEIGHT_2 = 0.0625;

float4 SampleGaussianBlur5Tap(float2 uv, float2 texelOffset)
{
    float4 col = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv) * GI_BLUR_WEIGHT_0;
    col += SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv + texelOffset) * GI_BLUR_WEIGHT_1;
    col += SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv - texelOffset) * GI_BLUR_WEIGHT_1;
    col += SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv + 2.0 * texelOffset) * GI_BLUR_WEIGHT_2;
    col += SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv - 2.0 * texelOffset) * GI_BLUR_WEIGHT_2;
    return col;
}

float4 SampleGaussianBlurHorizontal(float2 uv, float spread)
{
    return SampleGaussianBlur5Tap(uv, float2(spread / _ScreenParams.x, 0.0));
}

float4 SampleGaussianBlurVertical(float2 uv, float spread)
{
    return SampleGaussianBlur5Tap(uv, float2(0.0, spread / _ScreenParams.y));
}

#endif
