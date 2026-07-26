#ifndef GI_SSGI_RANDOM_H
#define GI_SSGI_RANDOM_H

// =============================================================================
// SSGIRandom.hlsl
//
// Hemisphere sample generation for the SSGI trace pass. Provides TWO swappable
// random number generators so the two distributions can be compared at runtime:
//
//   * _SSGI_RNG_R2   (default) - Low-discrepancy R2 sequence. Deterministic,
//                                prefix-independent and appendable. Identical
//                                pixel + sampleIndex always yields the same
//                                direction, so temporal accumulation relies on
//                                the per-frame sampleOffset advance performed
//                                on the CPU.
//
//   * _SSGI_RNG_HASH          - Hash + frame-counter RNG matching UnitySSGIURP.
//                                GenerateHashedRandomFloat(uint3(screenUV *
//                                texelSize, _FrameIndex + _Seed)) yields a
//                                different value every frame and every pixel,
//                                which is what UnitySSGIURP relies on for
//                                temporal variation.
//
// Both RNGs feed SampleHemisphereCosine(...) so the resulting hemisphere
// distribution is cosine-weighted in either mode, matching UnitySSGIURP's
// `ray.direction = SampleHemisphereCosine(...)`.
// =============================================================================

#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Random.hlsl"
#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Sampling/Sampling.hlsl"

// CPU-driven frame index (advanced every logical frame by the pass) and a
// rolling seed counter that mirrors UnitySSGIURP's _Seed behaviour: each call
// to GenerateRandomValue() bumps the seed so successive draws in the same
// pixel are decorrelated.
#ifndef SSGI_RNG_HASH_FRAME_INDEX
    #define SSGI_RNG_HASH_FRAME_INDEX 0
#endif

uint _SSGIRngFrameIndex;
uint _SSGIRngSeed;

// Consumed by the R2 branch of SSGIBuildHemisphereDirection. Declared here
// (instead of in each consuming pass body) because HLSL requires variables to
// be declared before use, and SSGIRandom.hlsl is included before the pass-body
// uniform block. Each shader pass compiles independently so there is no
// cross-pass linkage; the C# side still owns the material property via
// Shader.PropertyToID("_SSGISampleOffset").
int _SSGISampleOffset;

// ---------------------------------------------------------------------------
// Shared R2 constants. The 24-bit R2 step keeps the sequence stable across
// shader recompiles and is identical to the legacy SSGI implementation so
// existing comparison tooling keeps working when _SSGI_RNG_R2 is active.
// ---------------------------------------------------------------------------
static const uint2 SSGI_R2_STEP24 = uint2(12664746u, 9560334u);
static const float SSGI_R2_SCALE24 = 1.0 / 16777216.0;

// ---------------------------------------------------------------------------
// Hash-based RNG (UnitySSGIURP parity).
//
// Mirrors SSGIUtilities.hlsl::GenerateRandomValue from jiaozi158/UnitySSGIURP:
//   float GenerateRandomValue(float2 screenUV)
//   {
//       _Seed += 1.0;
//       return GenerateHashedRandomFloat(uint3(screenUV * _BlitTexture_TexelSize.zw,
//                                               _FrameIndex + _Seed));
//   }
//
// We expose SSGIGenerateRandomValueHash(...) which is callable from the trace
// loop without depending on Blit.hlsl's texel size uniform; callers pass the
// half-res render target size explicitly so the hash distributes the same way
// it does in UnitySSGIURP's half-resolution trace pass.
// ---------------------------------------------------------------------------
float SSGIGenerateRandomValueHash(float2 screenUV, float2 renderTargetSize)
{
    _SSGIRngSeed += 1u;
    uint3 hashInput = uint3(screenUV * renderTargetSize,
                            _SSGIRngFrameIndex + _SSGIRngSeed);
    return GenerateHashedRandomFloat(hashInput);
}

// ---------------------------------------------------------------------------
// Hash-based RNG helper (overload) for callers that only have a pixel index
// (e.g. the comparison harness). Kept inline so SSGIRandom.hlsl stays
// dependency-free beyond Unity Core's Random.hlsl.
// ---------------------------------------------------------------------------
float SSGIGenerateRandomValueHash(uint2 pixelPosition, uint frameIndex, uint seed)
{
    return GenerateHashedRandomFloat(uint3(pixelPosition, frameIndex + seed));
}

// ---------------------------------------------------------------------------
// R2 helpers shared with the legacy implementation.
// ---------------------------------------------------------------------------
float2 SSGIHashRandom(float2 pixelPosition)
{
    float3 p3 = frac(float3(pixelPosition.xyx) * float3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.xx + p3.yz) * p3.zy);
}

// Returns the cosine-weighted hemisphere direction in the same space as
// `normal`. Both RNGs feed the same SampleHemisphereCosine helper so the
// resulting distribution is identical to UnitySSGIURP's ray generation; only
// the source of the uniform variates differs.
//
//   normal        - surface normal in the space the marcher expects (view or
//                   world depending on the calling pass).
//   pixelPosition - pixel coordinate (input.positionCS.xy) used by both RNGs
//                   to decorrelate neighbouring pixels.
//   rayIndex      - index of the current ray in the per-pixel trace loop.
//   screenUV      - uv of the pixel (for the hash RNG).
//   renderTargetSize - pixel size of the half-res target the trace writes to
//                      (for the hash RNG).
float3 SSGIBuildHemisphereDirection(float3 normal,
                                    float2 pixelPosition,
                                    int   rayIndex,
                                    float2 screenUV,
                                    float2 renderTargetSize)
{
    #if defined(_SSGI_RNG_HASH)
        // UnitySSGIURP parity: two independent hashed variates per ray.
        float xiX = SSGIGenerateRandomValueHash(screenUV, renderTargetSize);
        float xiY = SSGIGenerateRandomValueHash(screenUV, renderTargetSize);
        return SampleHemisphereCosine(xiX, xiY, normal);
    #else
        // Legacy R2 low-discrepancy sequence (default).
        uint sampleIndex = ((uint)clamp(_SSGISampleOffset, 0, 65535) + (uint)rayIndex) & 0xFFFFu;
        uint2 phase24 = (uint2(sampleIndex, sampleIndex) * SSGI_R2_STEP24) & uint2(0x00FFFFFFu, 0x00FFFFFFu);
        float2 temporalSample = float2(phase24) * SSGI_R2_SCALE24;
        float2 xi = frac(SSGIHashRandom(pixelPosition) + temporalSample);
        return SampleHemisphereCosine(xi.x, xi.y, normal);
    #endif
}

#endif // GI_SSGI_RANDOM_H
