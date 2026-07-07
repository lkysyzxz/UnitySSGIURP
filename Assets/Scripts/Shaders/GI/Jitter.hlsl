#ifndef GI_JITTER_H
#define GI_JITTER_H

#if defined(_JITTER_ON)
// 4x4 Bayer dither offsets ray starts by [0,1) step to break regular sampling.
static const float _DitherTable[16] = {
    0.0,    0.5,    0.125,  0.625,
    0.75,   0.25,   0.875,  0.375,
    0.1875, 0.6875, 0.0625, 0.5625,
    0.9375, 0.4375, 0.8125, 0.3125
};

float GetJitter(float2 uv)
{
    uint2 pix = uint2(uv * _ScreenParams.xy);
    return _DitherTable[(pix.x & 3) * 4 + (pix.y & 3)];
}
#endif

#endif
