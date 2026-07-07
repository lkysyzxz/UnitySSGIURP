#ifndef GI_HIZ_UTILITY_H
#define GI_HIZ_UTILITY_H

#define DECLARE_HIZ(i) TEXTURE2D(_HiZTexture_##i)
DECLARE_HIZ(0); DECLARE_HIZ(1); DECLARE_HIZ(2); DECLARE_HIZ(3);
DECLARE_HIZ(4); DECLARE_HIZ(5); DECLARE_HIZ(6); DECLARE_HIZ(7);
#undef DECLARE_HIZ

int _HiZMipCount;
float _HiZMaxMip;

// Each mip is stored in a separate RT. Select the texture by mip level.
float SampleHiZ(float2 uv, int mip)
{
    [flatten]
    if (mip <= 0) return _HiZTexture_0.Sample(sampler_PointClamp, uv).r;
    else if (mip == 1) return _HiZTexture_1.Sample(sampler_PointClamp, uv).r;
    else if (mip == 2) return _HiZTexture_2.Sample(sampler_PointClamp, uv).r;
    else if (mip == 3) return _HiZTexture_3.Sample(sampler_PointClamp, uv).r;
    else if (mip == 4) return _HiZTexture_4.Sample(sampler_PointClamp, uv).r;
    else if (mip == 5) return _HiZTexture_5.Sample(sampler_PointClamp, uv).r;
    else if (mip == 6) return _HiZTexture_6.Sample(sampler_PointClamp, uv).r;
    else return _HiZTexture_7.Sample(sampler_PointClamp, uv).r;
}

#endif
