using System;
using System.IO;
using System.Reflection;
using System.Text.RegularExpressions;
using NUnit.Framework;
using UnityEngine;

public class SSGIRefactorTests
{
    private const int MomentSampleCount = 4096;

    private static string ProjectPath(params string[] parts)
    {
        string path = Application.dataPath;
        foreach (string part in parts)
            path = Path.Combine(path, part);
        return path;
    }

    private static string ReadAsset(params string[] parts)
    {
        string path = ProjectPath(parts);
        Assert.That(File.Exists(path), Is.True, $"Expected asset does not exist: {path}");
        return File.ReadAllText(path);
    }

    private static float FractionalPart(float value)
    {
        return value - Mathf.Floor(value);
    }

    private static bool ClipViewportAxis(float origin, float direction, float boundMin, float boundMax,
                                         ref float tEnter, ref float tExit)
    {
        if (direction == 0.0f)
            return origin >= boundMin && origin <= boundMax;

        float t0 = (boundMin - origin) / direction;
        float t1 = (boundMax - origin) / direction;
        tEnter = Mathf.Max(tEnter, Mathf.Min(t0, t1));
        tExit = Mathf.Min(tExit, Mathf.Max(t0, t1));
        return tEnter <= tExit;
    }

    private static bool ClipViewportSegment(Vector2 start, Vector2 end, Vector2 screenSize,
                                            out float tEnter, out float tExit)
    {
        Vector2 direction = end - start;
        Vector2 halfTexel = new Vector2(0.5f / screenSize.x, 0.5f / screenSize.y);
        Vector2 boundMin = halfTexel;
        Vector2 boundMax = Vector2.one - halfTexel;
        tEnter = 0.0f;
        tExit = 1.0f;
        return ClipViewportAxis(start.x, direction.x, boundMin.x, boundMax.x, ref tEnter, ref tExit) &&
               ClipViewportAxis(start.y, direction.y, boundMin.y, boundMax.y, ref tEnter, ref tExit);
    }

    private static Vector2 HashRandom(Vector2 pixelPosition)
    {
        Vector3 p3 = new Vector3(
            FractionalPart(pixelPosition.x * 0.1031f),
            FractionalPart(pixelPosition.y * 0.1030f),
            FractionalPart(pixelPosition.x * 0.0973f));
        Vector3 yzx = new Vector3(p3.y, p3.z, p3.x);
        p3 += Vector3.one * Vector3.Dot(p3, yzx + Vector3.one * 33.33f);
        return new Vector2(
            FractionalPart((p3.x + p3.y) * p3.z),
            FractionalPart((p3.x + p3.z) * p3.y));
    }

    private static uint WrappedSampleIndex(int sampleOffset, int rayIndex)
    {
        return (uint)(sampleOffset + rayIndex) & 0xFFFFu;
    }

    private static Vector3 HemisphereSample(Vector2 pixelPosition, int sampleOffset, int rayIndex)
    {
        uint sampleIndex = WrappedSampleIndex(sampleOffset, rayIndex);
        uint phaseX = unchecked(sampleIndex * 12664746u) & 0x00FFFFFFu;
        uint phaseY = unchecked(sampleIndex * 9560334u) & 0x00FFFFFFu;
        Vector2 temporalSample = new Vector2(phaseX, phaseY) / 16777216.0f;
        Vector2 xi = HashRandom(pixelPosition) + temporalSample;
        xi = new Vector2(FractionalPart(xi.x), FractionalPart(xi.y));
        double phi = 2.0 * Math.PI * xi.y;
        double cosTheta = 1.0 - 2.0 * xi.x;
        double sinTheta = Math.Sqrt(Math.Max(0.0, 1.0 - cosTheta * cosTheta));
        Vector3 pointOnSphere = new Vector3(
            (float)(sinTheta * Math.Cos(phi)),
            (float)(sinTheta * Math.Sin(phi)),
            (float)cosTheta);
        Vector3 sample = Vector3.forward + pointOnSphere;
        return sample.sqrMagnitude > 1e-12f ? sample.normalized : Vector3.forward;
    }

    private static float EdgeAwareWeight(float centerDepth, Vector3 centerNormal,
                                         float sampleDepth, Vector3 sampleNormal, bool valid)
    {
        if (!valid || sampleDepth < 0.0f)
            return 0.0f;

        float depthWeight = Mathf.Exp(-5.0f * Mathf.Abs(centerDepth - sampleDepth));
        float normalWeight = Mathf.Max(0.0f, Vector3.Dot(centerNormal, sampleNormal));
        return depthWeight * normalWeight;
    }

    private static Vector4 FilterEdgeAwareSamples(Vector3[] colors, float[] depths,
                                                  Vector3[] normals, bool[] valid,
                                                  float centerDepth, Vector3 centerNormal)
    {
        Assert.That(colors.Length, Is.EqualTo(depths.Length));
        Assert.That(colors.Length, Is.EqualTo(normals.Length));
        Assert.That(colors.Length, Is.EqualTo(valid.Length));

        Vector3 filtered = Vector3.zero;
        float weightSum = 0.0f;
        for (int i = 0; i < colors.Length; i++)
        {
            float weight = EdgeAwareWeight(centerDepth, centerNormal, depths[i], normals[i], valid[i]);
            filtered += colors[i] * weight;
            weightSum += weight;
        }

        filtered /= Mathf.Max(0.001f, weightSum);
        return new Vector4(filtered.x, filtered.y, filtered.z, centerDepth);
    }

    [Test]
    public void DirectionSequence_IsPrefixIndependentAndAppendable()
    {
        const int offset = 37;
        Vector2 pixelPosition = new Vector2(317.5f, 211.5f);
        Vector3[] shortPrefix = new Vector3[4];
        Vector3[] longPrefix = new Vector3[16];

        for (int i = 0; i < shortPrefix.Length; i++)
            shortPrefix[i] = HemisphereSample(pixelPosition, offset, i);
        for (int i = 0; i < longPrefix.Length; i++)
            longPrefix[i] = HemisphereSample(pixelPosition, offset, i);

        for (int i = 0; i < shortPrefix.Length; i++)
            Assert.That(longPrefix[i], Is.EqualTo(shortPrefix[i]));

        // RNG code lives in SSGIRandom.hlsl (both R2 and hash variants) so the
        // trace shader stays focused on the trace loop itself.
        string randomLibrary = ReadAsset("Scripts", "Shaders", "GI", "SSGIRandom.hlsl");
        Assert.That(randomLibrary, Does.Match(@"SSGI_R2_STEP24\s*=\s*uint2\s*\(\s*12664746u\s*,\s*9560334u\s*\)"));
        Assert.That(randomLibrary, Does.Contain("SSGIHashRandom"));
        Assert.That(randomLibrary, Does.Match(@"sampleIndex\s*=\s*\(\s*\(uint\)clamp[\s\S]*&\s*0xFFFFu"));
        Assert.That(randomLibrary, Does.Match(@"phase24\s*=\s*\(\s*uint2\s*\(\s*sampleIndex\s*,\s*sampleIndex\s*\)\s*\*\s*SSGI_R2_STEP24\s*\)\s*&"));
        Assert.That(randomLibrary, Does.Match(@"frac\s*\(\s*SSGIHashRandom\s*\(\s*pixelPosition\s*\)\s*\+\s*temporalSample\s*\)"));
        Assert.That(randomLibrary, Does.Match(@"SampleHemisphereCosine\s*\(\s*xi\.x\s*,\s*xi\.y\s*,\s*normal\s*\)"));
        Assert.That(randomLibrary, Does.Match(@"SSGIBuildHemisphereDirection\s*\(\s*normal"));
        Assert.That(randomLibrary, Does.Not.Match(@"sampleIndex\s*\*\s*float3"));

        // Hash RNG branch must be present and selectable via _SSGI_RNG_HASH.
        Assert.That(randomLibrary, Does.Contain("SSGIGenerateRandomValueHash"));
        Assert.That(randomLibrary, Does.Match(@"#if\s+defined\s*\(\s*_SSGI_RNG_HASH\s*\)"));
        Assert.That(randomLibrary, Does.Contain("GenerateHashedRandomFloat"));

        // Shader selects between the two via multi_compile.
        string shader = ReadAsset("Scripts", "Shaders", "SSGI", "ScreenSpaceGlobalIllumination.shader");
        Assert.That(shader, Does.Match(@"#pragma\s+multi_compile\s+_\s+_SSGI_RNG_HASH"));
        Assert.That(shader, Does.Match(@"SSGIBuildHemisphereDirection\s*\(\s*normal[\s\S]*input\.positionCS\.xy[\s\S]*i[\s\S]*uv[\s\S]*_ScreenParams\.xy\s*\)"));

        Assert.That(HemisphereSample(pixelPosition + Vector2.one, offset, 0), Is.Not.EqualTo(shortPrefix[0]));
        Assert.That(WrappedSampleIndex(65534, 0), Is.EqualTo(65534u));
        Assert.That(WrappedSampleIndex(65534, 1), Is.EqualTo(65535u));
        Assert.That(WrappedSampleIndex(65534, 2), Is.EqualTo(0u));
        Assert.That(WrappedSampleIndex(65534, 3), Is.EqualTo(1u));
    }

    [Test]
    public void CosineWeightedHemisphereMapping_IsFiniteUnitLengthAndHasExpectedMoments()
    {
        Vector2[] pixelPositions =
        {
            new Vector2(317.5f, 211.5f),
            new Vector2(911.5f, 503.5f),
            new Vector2(17.5f, 997.5f)
        };
        int[] offsets = { 0, 1234, 65530 };

        for (int sequence = 0; sequence < pixelPositions.Length; sequence++)
        {
            Vector3 sum = Vector3.zero;
            Vector3 sumSquares = Vector3.zero;

            for (int i = 0; i < MomentSampleCount; i++)
            {
                Vector3 sample = HemisphereSample(pixelPositions[sequence], offsets[sequence], i);
                Assert.That(float.IsNaN(sample.x) || float.IsInfinity(sample.x), Is.False);
                Assert.That(float.IsNaN(sample.y) || float.IsInfinity(sample.y), Is.False);
                Assert.That(float.IsNaN(sample.z) || float.IsInfinity(sample.z), Is.False);
                Assert.That(sample.magnitude, Is.EqualTo(1.0f).Within(1e-5f));
                Assert.That(sample.z, Is.InRange(0.0f, 1.0f));
                sum += sample;
                sumSquares += Vector3.Scale(sample, sample);
            }

            Vector3 mean = sum / MomentSampleCount;
            Vector3 secondMoment = sumSquares / MomentSampleCount;
            Assert.That(mean.x, Is.EqualTo(0.0f).Within(0.03f));
            Assert.That(mean.y, Is.EqualTo(0.0f).Within(0.03f));
            Assert.That(mean.z, Is.EqualTo(2.0f / 3.0f).Within(0.03f));
            Assert.That(secondMoment.x, Is.EqualTo(0.25f).Within(0.03f));
            Assert.That(secondMoment.y, Is.EqualTo(0.25f).Within(0.03f));
            Assert.That(secondMoment.z, Is.EqualTo(0.5f).Within(0.03f));
        }
    }

    [Test]
    public void CosineWeightedEstimator_AveragesFixedRayCountIncludingMisses()
    {
        const int rayCount = 4;
        double[] hitRadiance = { 1.0, 0.0, 1.0, 0.0 };
        double estimate = 0.0;

        for (int i = 0; i < rayCount; i++)
            estimate += hitRadiance[i];
        estimate /= rayCount;

        Assert.That(estimate, Is.EqualTo(0.5).Within(1e-12));

        double fullVisibility = 0.0;
        for (int i = 0; i < MomentSampleCount; i++)
        {
            double nDotL = HemisphereSample(new Vector2(317.5f, 211.5f), 0, i).z;
            double pdf = nDotL / Math.PI;
            fullVisibility += nDotL / (pdf * Math.PI);
        }
        fullVisibility /= MomentSampleCount;
        Assert.That(fullVisibility, Is.EqualTo(1.0).Within(0.01));

        string shader = ReadAsset("Scripts", "Shaders", "SSGI", "ScreenSpaceGlobalIllumination.shader");
        Assert.That(shader, Does.Match(@"receiverCosine\s*=\s*saturate\s*\(\s*dot\s*\(\s*normal\s*,\s*rayDir\s*\)"));
        Assert.That(shader, Does.Match(@"samplePDF\s*=\s*max\s*\([\s\S]*receiverCosine\s*\*\s*SSGI_INV_PI"));
        Assert.That(shader, Does.Match(@"monteCarloWeight\s*=[\s\S]*receiverDiffuseBRDF\s*\*\s*receiverCosine\s*/\s*samplePDF"));
        Assert.That(shader, Does.Match(@"irradiance\s*\+=\s*hitOutgoingRadiance\s*\*\s*monteCarloWeight"));
        Assert.That(shader, Does.Match(@"hitDiffuseReflectance\s*=\s*hitAlbedo\s*\*\s*\(\s*1\.0\s*-\s*saturate\s*\(\s*hitMetallic\s*\)\s*\)"));
        Assert.That(shader, Does.Match(@"hitOutgoingRadiance\s*\+=\s*hitIndirectIrradiance\s*\*\s*hitDiffuseReflectance"));
        Assert.That(shader, Does.Match(@"irradiance\s*/\s*rayCount\s*\*\s*_SSGIIntensity"));
        Assert.That(shader, Does.Contain("SSGI_INV_PI"));
        Assert.That(shader, Does.Not.Contain("Name \"Composite\""));
        Assert.That(shader, Does.Not.Contain("hitCount"));
        Assert.That(shader, Does.Not.Contain("weightSum"));
        Assert.That(shader, Does.Not.Contain("distanceWeight"));
    }

    [Test]
    public void SSGI_OrientsReconstructedNormalTowardCamera()
    {
        string shader = ReadAsset("Scripts", "Shaders", "SSGI", "ScreenSpaceGlobalIllumination.shader");
        Assert.That(shader, Does.Match(@"if\s*\(\s*dot\s*\(\s*normal\s*,\s*-viewPos\s*\)\s*<=\s*0\.0\s*\)\s*normal\s*=\s*-normal\s*;"));
    }

    [Test]
    public void TemporalSampleOffset_AdvancesByClampedRayCountAndWraps16Bits()
    {
        const int configuredOffset = 65530;
        int rayCount = Mathf.Clamp(48, 1, 128);
        int firstFrameOffset = Mathf.Clamp(configuredOffset, 0, 65535);
        int secondFrameOffset = (firstFrameOffset + rayCount) & 0xFFFF;

        Assert.That(firstFrameOffset, Is.EqualTo(65530));
        Assert.That(rayCount, Is.EqualTo(48));
        Assert.That(secondFrameOffset, Is.EqualTo(42));

        string feature = ReadAsset("Scripts", "Runtime", "Features", "ScreenSpaceGlobalIlluminationFeature.cs");
        Assert.That(Regex.Matches(feature, @"Mathf\.Clamp\s*\(\s*settings\.rayCount\s*,\s*1\s*,\s*128\s*\)").Count, Is.EqualTo(1));
        Assert.That(feature, Does.Match(@"sampleOffset\s*=\s*\(\s*state\.sampleOffset\s*\+\s*rayCount\s*\)\s*&\s*0xFFFF"));
        Assert.That(feature, Does.Match(@"baseSampleOffset\s*=\s*Mathf\.Clamp\s*\(\s*settings\.sampleOffset\s*,\s*0\s*,\s*65535\s*\)"));
        Assert.That(feature, Does.Not.Match(@"settings\.sampleOffset\s*="));

        FieldInfo rayCountField = typeof(ScreenSpaceGlobalIlluminationFeature.SSGISettings).GetField("rayCount");
        UnityEngine.RangeAttribute range = rayCountField.GetCustomAttribute<UnityEngine.RangeAttribute>();
        Assert.That(range.min, Is.EqualTo(1.0f));
        Assert.That(range.max, Is.EqualTo(128.0f));

        string shader = ReadAsset("Scripts", "Shaders", "SSGI", "ScreenSpaceGlobalIllumination.shader");
        Assert.That(shader, Does.Match(@"clamp\s*\(\s*_SSGIRayCount\s*,\s*1\s*,\s*128\s*\)"));

        // _SSGISampleOffset clamp moved into SSGIRandom.hlsl alongside the
        // R2 sequence that consumes it.
        string randomLibrary = ReadAsset("Scripts", "Shaders", "GI", "SSGIRandom.hlsl");
        Assert.That(randomLibrary, Does.Match(@"clamp\s*\(\s*_SSGISampleOffset\s*,\s*0\s*,\s*65535\s*\)"));

    }

    [Test]
    public void TemporalState_IsPerCameraAndReprojectsViewChangesWithoutResettingHistory()
    {
        string feature = ReadAsset("Scripts", "Runtime", "Features", "ScreenSpaceGlobalIlluminationFeature.cs");

        Assert.That(feature, Does.Contain("using System.Collections.Generic;"));
        Assert.That(feature, Does.Match(@"Dictionary\s*<\s*int\s*,\s*CameraState\s*>\s+cameraStates"));
        Assert.That(feature, Does.Contain("camera.GetInstanceID()"));
        Assert.That(feature, Does.Contain("previousPosition"));
        Assert.That(feature, Does.Contain("previousRotation"));
        Assert.That(feature, Does.Contain("previousProjectionMatrix"));
        Assert.That(feature, Does.Contain("previousViewProjectionMatrix"));
        Assert.That(feature, Does.Contain("previousWorldToCameraMatrix"));
        Assert.That(feature, Does.Contain("previousSettingsHash"));
        Assert.That(feature, Does.Contain("currentIrradianceRT"));
        Assert.That(feature, Does.Contain("sampleOffset"));
        Assert.That(feature, Does.Contain("lastUpdateFrame"));
        Assert.That(feature, Does.Match(@"viewChanged\s*=\s*!state\.hasPreviousPose"));
        Assert.That(feature, Does.Match(@"previousPosition\s*!=\s*cam\.transform\.position"));
        Assert.That(feature, Does.Match(@"previousRotation\s*!=\s*cam\.transform\.rotation"));
        Assert.That(feature, Does.Match(@"cam\.nonJitteredProjectionMatrix"));
        Assert.That(feature, Does.Match(@"previousProjectionMatrix\s*!=\s*projectionMatrix"));
        Assert.That(feature, Does.Match(@"previousSettingsHash\s*!=\s*settingsHash"));
        Assert.That(feature, Does.Match(@"resetHistory\s*=\s*configurationChanged"));
        Assert.That(feature, Does.Not.Match(@"resetHistory\s*=.*viewChanged"));
        Assert.That(feature, Does.Match(@"descriptorChanged[\s\S]*ResetTemporalState\s*\(\s*state\s*\)"));
        Assert.That(feature, Does.Match(@"lastUpdateFrame\s*!=\s*Time\.frameCount"));
    }

    [Test]
    public void ViewChange_ReprojectsTemporalHistoryButDisablesHitIrradianceHistory()
    {
        string feature = ReadAsset("Scripts", "Runtime", "Features", "ScreenSpaceGlobalIlluminationFeature.cs");

        Assert.That(feature, Does.Match(@"if\s*\(\s*resetHistory\s*\)\s*ResetTemporalState\s*\(\s*state\s*\)"));
        Assert.That(feature, Does.Not.Contain("MotionHistoryFrameCap"));
        Assert.That(feature, Does.Match(@"bool\s+accumulate\s*=\s*resetHistory\s*\|\|\s*isNewLogicalFrame\s*\|\|\s*viewChanged"));
        Assert.That(feature, Does.Match(@"canUseHistory\s*=\s*accumulate\s*&&\s*state\.historyValid\s*&&[\s\S]*!resetHistory"));
        Assert.That(feature, Does.Not.Match(@"canUseHistory\s*=[^;]*!viewChanged"));
        Assert.That(feature, Does.Match(@"canUsePreviousIrradiance\s*=\s*canUseHistory\s*&&\s*!viewChanged"));
        Assert.That(feature, Does.Match(@"HistoryValidID\s*,\s*canUseHistory\s*\?\s*1\.0f\s*:\s*0\.0f"));
    }

    [Test]
    public void TemporalAccumulation_UsesPerPixelFiniteFrameMonteCarloMean()
    {
        string shader = ReadAsset("Scripts", "Shaders", "SSGI", "ScreenSpaceGlobalIllumination.shader");
        string temporal = ReadAsset("Scripts", "Shaders", "GI", "Temporal.hlsl");
        Assert.That(shader, Does.Contain("Name \"Accumulate\""));
        Assert.That(temporal, Does.Contain("TEXTURE2D_X(_MotionVectorTexture)"));
        Assert.That(temporal, Does.Match(@"previousUV\s*=\s*uv\s*-\s*motion"));
        Assert.That(temporal, Does.Match(@"historyDepth\s*=\s*SAMPLE_TEXTURE2D_X\s*\([\s\S]*sampler_PointClamp\s*,\s*previousUV\s*\)\.a"));
        Assert.That(temporal, Does.Match(@"currentPosition\s*=\s*SAMPLE_TEXTURE2D_X\s*\([\s\S]*_SSGIGBufferPositionWS"));
        Assert.That(temporal, Does.Not.Contain("ClipSSGIHistoryToCurrentNeighborhood"));
        Assert.That(temporal, Does.Contain("TEXTURE2D_X(_SSGIHistorySampleTexture)"));
        Assert.That(temporal, Does.Match(@"historySampleCount\s*=\s*SAMPLE_TEXTURE2D_X\s*\([\s\S]*_SSGIHistorySampleTexture"));
        Assert.That(temporal, Does.Match(@"result\.sampleCount\s*=\s*historySampleCount"));
        Assert.That(temporal, Does.Match(@"expectedHistoryDepth\s*=\s*-mul\s*\([\s\S]*_SSGIPreviousWorldToCameraMatrix"));
        Assert.That(temporal, Does.Match(@"abs\s*\(\s*historyDepth\s*-\s*expectedHistoryDepth\s*\)\s*>\s*depthThreshold"));
        Assert.That(shader, Does.Not.Contain("RGBToYCoCg"));
        Assert.That(shader, Does.Match(@"nextSampleCount\s*=\s*min\s*\(\s*temporal\.sampleCount\s*\+\s*1\.0\s*,\s*maxHistoryFrames"));
        Assert.That(shader, Does.Match(@"historyWeight\s*=\s*temporal\.sampleCount\s*>=\s*maxHistoryFrames[\s\S]*_SSGIHistoryWeight[\s\S]*temporal\.sampleCount\s*/\s*nextSampleCount"));
        Assert.That(shader, Does.Match(@"currentFrameWeight\s*=\s*1\.0\s*-\s*historyWeight"));
        Assert.That(shader, Does.Match(@"accumulated\s*=[\s\S]*temporal\.irradianceDepth\.rgb\s*\*\s*historyWeight\s*\+[\s\S]*current\.rgb\s*\*\s*currentFrameWeight"));
        Assert.That(shader, Does.Match(@"return\s+float4\s*\(\s*accumulated\s*,\s*currentDepth\s*\)"));
        Assert.That(shader, Does.Contain("Name \"UpdateSampleCount\""));
        Assert.That(shader, Does.Match(@"FragUpdateSampleCount[\s\S]*temporal\.sampleCount\s*\+\s*1\.0[\s\S]*_SSGIMaxHistoryFrames"));
        Assert.That(shader, Does.Not.Contain("Name \"UpdateMoments\""));
        Assert.That(shader, Does.Not.Contain("Name \"UpdateNormalCount\""));
        Assert.That(shader, Does.Match(@"irradiance\s*/\s*rayCount\s*\*\s*_SSGIIntensity"));
        Assert.That(Regex.Matches(shader, @"return\s+float4\s*\(\s*0\s*,\s*0\s*,\s*0\s*,\s*-1\.0\s*\)").Count, Is.EqualTo(2));
        Assert.That(shader, Does.Not.Contain("_SSGIMaxBlend"));

        string feature = ReadAsset("Scripts", "Runtime", "Features", "ScreenSpaceGlobalIlluminationFeature.cs");
        Assert.That(feature, Does.Match(@"temporalHistoryWeight\s*=\s*1\.0f\s*-\s*temporalResponse"));
        Assert.That(feature, Does.Match(@"HistoryWeightID\s*,\s*temporalHistoryWeight\s*\)"));
        Assert.That(feature, Does.Match(@"MaxHistoryFramesID\s*,\s*maxHistoryFrames\s*\)"));
        Assert.That(feature, Does.Match(@"Mathf\.Clamp\s*\(\s*settings\.maxHistoryFrames\s*,\s*2\s*,\s*2048\s*\)"));
        Assert.That(feature, Does.Contain("HistorySampleRead"));
        Assert.That(feature, Does.Contain("HistorySampleWrite"));
        Assert.That(feature, Does.Match(@"HistoryDepthThresholdID\s*,\s*temporalDepthThreshold\s*\)"));
        Assert.That(feature, Does.Match(@"ScriptableRenderPassInput\.Motion"));
        Assert.That(feature, Does.Match(@"hash\s*=\s*hash\s*\*\s*31\s*\+\s*TemporalAlgorithmVersion"));
    }

    [Test]
    public void TemporalRenderSequence_RunsHalfResTraceAndCompositesIntoCameraColorViaApplyRT()
    {
        string feature = ReadAsset("Scripts", "Runtime", "Features", "ScreenSpaceGlobalIlluminationFeature.cs");

        // Half-resolution pipeline (UnitySSGIURP style): SSGI trace, Accumulate
        // and the two blur passes all run on half-res working textures.
        Assert.That(feature, Does.Match(@"sourceHandle\s*,\s*state\.radianceRT\s*\)"));
        Assert.That(feature, Does.Not.Match(@"state\.applyRT\s*,\s*state\.radianceRT"));
        Assert.That(feature, Does.Match(@"previousIrradiance\s*,\s*state\.currentIrradianceRT\s*,\s*material\s*,\s*0\s*\)"));
        Assert.That(feature, Does.Match(@"state\.currentIrradianceRT\s*,\s*state\.blurA\s*,\s*material\s*,\s*3\s*\)"));
        Assert.That(feature, Does.Match(@"state\.currentIrradianceRT\s*,\s*state\.HistorySampleWrite\s*,\s*material\s*,\s*4\s*\)"));
        Assert.That(feature, Does.Match(@"state\.blurA\s*,\s*state\.blurB\s*,\s*material\s*,\s*1\s*\)"));
        Assert.That(feature, Does.Match(@"state\.blurB\s*,\s*state\.HistoryWrite\s*,\s*material\s*,\s*2\s*\)"));
        Assert.That(feature, Does.Match(@"material\s*,\s*0\s*\)[\s\S]*material\s*,\s*3\s*\)[\s\S]*material\s*,\s*4\s*\)[\s\S]*material\s*,\s*1\s*\)[\s\S]*material\s*,\s*2\s*\)"));
        Assert.That(feature, Does.Match(@"SetGlobalTexture\s*\(\s*RadianceTextureID\s*,\s*state\.radianceRT\s*\)"));
        Assert.That(feature, Does.Match(@"SetGlobalTexture\s*\(\s*HistorySampleTextureID\s*,\s*state\.HistorySampleRead\s*\)"));
        Assert.That(feature, Does.Match(@"SetGlobalTexture\s*\(\s*HistorySampleTextureID\s*,\s*state\.HistorySampleWrite\s*\)"));
        Assert.That(feature, Does.Match(@"SetGlobalTexture\s*\(\s*IrradianceTextureID\s*,\s*currentIrradiance\s*\)"));
        Assert.That(feature, Does.Match(@"currentIrradiance\s*=\s*accumulate[\s\S]*state\.HistoryWrite[\s\S]*state\.HistoryRead"));

        // Half-res descriptor allocation: working textures must halve the
        // camera descriptor when halfResolution is enabled.
        Assert.That(feature, Does.Match(@"halfResolution"));
        Assert.That(feature, Does.Match(@"traceDesc\.width\s*=\s*Mathf\.Max\s*\(\s*1\s*,\s*desc\.width\s*/\s*2\s*\)"));
        Assert.That(feature, Does.Match(@"traceDesc\.height\s*=\s*Mathf\.Max\s*\(\s*1\s*,\s*desc\.height\s*/\s*2\s*\)"));
        Assert.That(feature, Does.Match(@"ReAllocateIfNeeded\s*\(\s*ref\s+state\.radianceRT\s*,\s*traceDesc"));
        Assert.That(feature, Does.Match(@"ReAllocateIfNeeded\s*\(\s*ref\s+state\.currentIrradianceRT\s*,\s*traceDesc"));
        Assert.That(feature, Does.Match(@"sampleCountDesc\.graphicsFormat\s*=\s*GraphicsFormat\.R16_SFloat"));
        Assert.That(feature, Does.Match(@"ReAllocateIfNeeded\s*\(\s*ref\s+state\.historySampleA\s*,\s*sampleCountDesc"));
        Assert.That(feature, Does.Match(@"ReAllocateIfNeeded\s*\(\s*ref\s+state\.historySampleB\s*,\s*sampleCountDesc"));
        // applyRT must remain full-res because it is the destination of the
        // Combine pass and is copied back into the camera colour target.
        Assert.That(feature, Does.Match(@"ReAllocateIfNeeded\s*\(\s*ref\s+state\.applyRT\s*,\s*desc"));

        // Compositing: a Combine pass (material pass 5) reads camera color +
        // half-res irradiance + albedo and writes to applyRT. applyRT is then
        // blitted back into sourceHandle. The composite must NOT skip via the
        // legacy direct-irradiance-to-sourceHandle path.
        Assert.That(feature, Does.Not.Contain("compositeRT"));
        Assert.That(feature, Does.Not.Contain("irradianceRT"));
        Assert.That(feature, Does.Match(@"BlitCameraTexture\s*\(\s*cmd\s*,\s*sourceHandle\s*,\s*state\.applyRT\s*,\s*material\s*,\s*5\s*\)"));
        Assert.That(feature, Does.Match(@"BlitCameraTexture\s*\(\s*cmd\s*,\s*state\.applyRT\s*,\s*sourceHandle\s*\)"));
        Assert.That(feature, Does.Not.Contain("MomentsWrite"));
        Assert.That(feature, Does.Not.Contain("NormalCountWrite"));

        string shader = ReadAsset("Scripts", "Shaders", "SSGI", "ScreenSpaceGlobalIllumination.shader");
        Assert.That(shader, Does.Contain("Name \"Combine\""));
        Assert.That(shader, Does.Contain("TEXTURE2D_X(_SSGIIrradianceTexture)"));
        Assert.That(shader, Does.Contain("TEXTURE2D_X(_SSGIGBufferAlbedo)"));
        Assert.That(shader, Does.Contain("TEXTURE2D_X(_SSGIGBufferMaterial)"));
        Assert.That(shader, Does.Contain("_SSGIIrradianceTexture_TexelSize"));
        Assert.That(shader, Does.Match(@"diffuseReflectance\s*=\s*albedo\s*\*\s*\(\s*1\.0\s*-\s*saturate\s*\(\s*metallic\s*\)\s*\)"));
        Assert.That(shader, Does.Match(@"cameraColor\s*\+\s*indirectDiffuse\s*\*\s*diffuseReflectance"));
    }

    [Test]
    public void PreviousIrradiance_IsDisabledBeforeOpaqueAndBoundInsideSSGIPass()
    {
        string feature = ReadAsset("Scripts", "Runtime", "Features", "ScreenSpaceGlobalIlluminationFeature.cs");

        Assert.That(feature, Does.Contain("IrradianceHistorySetupPass"));
        Assert.That(feature, Does.Match(@"irradianceHistorySetupPass\.renderPassEvent\s*=\s*RenderPassEvent\.BeforeRenderingOpaques"));
        Assert.That(feature, Does.Match(@"EnqueuePass\s*\(\s*irradianceHistorySetupPass\s*\)"));
        Assert.That(feature, Does.Match(@"BindPreviousIrradiance[\s\S]*SetGlobalFloat\s*\(\s*IrradianceValidID\s*,\s*0\.0f\s*\)"));
        Assert.That(feature, Does.Match(@"previousIrradiance\s*=\s*canUsePreviousIrradiance[\s\S]*state\.HistoryRead[\s\S]*state\.radianceRT"));
        Assert.That(feature, Does.Match(@"SetGlobalTexture\s*\(\s*IrradianceTextureID\s*,\s*previousIrradiance\s*\)"));
        Assert.That(feature, Does.Match(@"PreviousIrradianceValidID\s*,[\s\S]*canUsePreviousIrradiance\s*\?\s*1\.0f\s*:\s*0\.0f"));
        Assert.That(feature, Does.Match(@"SetGlobalMatrix\s*\(\s*PreviousViewProjectionMatrixID\s*,\s*state\.previousViewProjectionMatrix\s*\)"));
        Assert.That(feature, Does.Match(@"SetGlobalMatrix\s*\(\s*PreviousWorldToCameraMatrixID\s*,\s*state\.previousWorldToCameraMatrix\s*\)"));
    }

    [Test]
    public void PerCameraTargets_AreAllReleasedAndStateDictionaryIsCleared()
    {
        string feature = ReadAsset("Scripts", "Runtime", "Features", "ScreenSpaceGlobalIlluminationFeature.cs");

        foreach (string handle in new[]
        {
            "radianceRT", "currentIrradianceRT", "blurA", "blurB",
            "historyA", "historyB", "historySampleA", "historySampleB",
            "applyRT"
        })
            Assert.That(feature, Does.Match(handle + @"\?\.Release\s*\(\s*\)"), $"Expected {handle} to be released");
        Assert.That(feature, Does.Match(@"Shader\.SetGlobalFloat\s*\(\s*IrradianceValidID\s*,\s*0\.0f\s*\)"));
        Assert.That(feature, Does.Match(@"foreach\s*\(\s*CameraState\s+state\s+in\s+cameraStates\.Values\s*\)"));
        Assert.That(feature, Does.Match(@"cameraStates\.Clear\s*\(\s*\)"));
    }

    [Test]
    public void PerCameraStateCache_ReleasesDeadOrLeastRecentlyUsedEntriesAndIsBounded()
    {
        string feature = ReadAsset("Scripts", "Runtime", "Features", "ScreenSpaceGlobalIlluminationFeature.cs");

        Assert.That(feature, Does.Match(@"const\s+int\s+MaxCameraStates\s*=\s*4"));
        Assert.That(feature, Does.Match(@"public\s+Camera\s+camera\s*;"));
        Assert.That(feature, Does.Match(@"public\s+int\s+lastUsedFrame\s*;"));
        Assert.That(Regex.Matches(feature, @"state\.lastUsedFrame\s*=\s*Time\.frameCount").Count, Is.GreaterThanOrEqualTo(2));
        Assert.That(feature, Does.Match(@"state\.camera\s*==\s*null\s*\|\|\s*state\.camera\s*!=\s*camera"));
        Assert.That(feature, Does.Match(@"RetireCameraStates\s*\(\s*cameraId\s*\)[\s\S]*cameraStates\.Add\s*\(\s*cameraId\s*,\s*state\s*\)"));
        Assert.That(feature, Does.Match(@"while\s*\(\s*cameraStates\.Count\s*>=\s*MaxCameraStates\s*\)"));
        Assert.That(feature, Does.Match(@"pair\.Key\s*==\s*currentCameraId[\s\S]*continue\s*;"));
        Assert.That(feature, Does.Match(@"leastRecentlyUsed\.Release\s*\(\s*\)\s*;\s*cameraStates\.Remove\s*\(\s*leastRecentlyUsedId\s*\)"));
        Assert.That(feature, Does.Match(@"pair\.Value\.camera\s*==\s*null[\s\S]*pair\.Value\.Release\s*\(\s*\)"));
    }

    [Test]
    public void SSGIAndSSR_RejectSkyDepthForCorrectZConventionWithTolerance()
    {
        string ssgi = ReadAsset("Scripts", "Shaders", "SSGI", "ScreenSpaceGlobalIllumination.shader");
        string ssr = ReadAsset("Scripts", "Shaders", "SSR", "ScreenSpaceReflection.shader");

        foreach (string shader in new[] { ssgi, ssr })
        {
            Assert.That(shader, Does.Match(@"UNITY_REVERSED_Z[\s\S]*rawDepth\s*<=\s*0\.000001"));
            Assert.That(shader, Does.Match(@"#else[\s\S]*rawDepth\s*>=\s*0\.999999"));
            Assert.That(shader, Does.Not.Match(@"rawDepth\s*==\s*[01]\.0"));
        }
    }

    [Test]
    public void Settings_ExposeSampleOffsetAndIndependentOriginBiasWithBindings()
    {
        Type settingsType = typeof(ScreenSpaceGlobalIlluminationFeature.SSGISettings);
        FieldInfo sampleOffset = settingsType.GetField("sampleOffset");
        FieldInfo originBias = settingsType.GetField("originBias");
        FieldInfo temporalResponse = settingsType.GetField("temporalResponse");
        FieldInfo maxHistoryFrames = settingsType.GetField("maxHistoryFrames");
        FieldInfo temporalDepthThreshold = settingsType.GetField("temporalDepthThreshold");
        FieldInfo disocclusionFallback = settingsType.GetField("disocclusionFallback");

        Assert.That(sampleOffset, Is.Not.Null);
        Assert.That(originBias, Is.Not.Null);
        Assert.That(temporalResponse, Is.Not.Null);
        Assert.That(maxHistoryFrames, Is.Not.Null);
        Assert.That(temporalDepthThreshold, Is.Not.Null);
        Assert.That(disocclusionFallback, Is.Not.Null);
        object settings = Activator.CreateInstance(settingsType);
        Assert.That(sampleOffset.GetValue(settings), Is.EqualTo(0));
        Assert.That(originBias.GetValue(settings), Is.EqualTo(0.01f));
        Assert.That(temporalResponse.GetValue(settings), Is.EqualTo(0.08f));
        Assert.That(maxHistoryFrames.GetValue(settings), Is.EqualTo(8));
        Assert.That(temporalDepthThreshold.GetValue(settings), Is.EqualTo(0.1f));
        Assert.That(disocclusionFallback.GetValue(settings), Is.EqualTo(0.35f));
        Assert.That(sampleOffset.GetCustomAttribute<UnityEngine.RangeAttribute>(), Is.Not.Null);
        Assert.That(originBias.GetCustomAttribute<UnityEngine.RangeAttribute>(), Is.Not.Null);
        Assert.That(temporalResponse.GetCustomAttribute<UnityEngine.RangeAttribute>(), Is.Not.Null);
        UnityEngine.RangeAttribute historyRange =
            maxHistoryFrames.GetCustomAttribute<UnityEngine.RangeAttribute>();
        Assert.That(historyRange, Is.Not.Null);
        Assert.That(historyRange.min, Is.EqualTo(2.0f));
        Assert.That(historyRange.max, Is.EqualTo(2048.0f));
        Assert.That(temporalDepthThreshold.GetCustomAttribute<UnityEngine.MinAttribute>(), Is.Not.Null);
        Assert.That(disocclusionFallback.GetCustomAttribute<UnityEngine.RangeAttribute>(), Is.Not.Null);

        string feature = ReadAsset("Scripts", "Runtime", "Features", "ScreenSpaceGlobalIlluminationFeature.cs");
        Assert.That(feature, Does.Contain("_SSGISampleOffset"));
        Assert.That(feature, Does.Contain("_SSGIOriginBias"));
        Assert.That(feature, Does.Contain("_SSGIHistoryDepthThreshold"));
        Assert.That(feature, Does.Contain("_SSGIMaxHistoryFrames"));
        Assert.That(feature, Does.Contain("_SSGIDisocclusionFallback"));

        string shader = ReadAsset("Scripts", "Shaders", "SSGI", "ScreenSpaceGlobalIllumination.shader");
        Assert.That(shader, Does.Contain("normal * _SSGIOriginBias"));
        Assert.That(shader, Does.Not.Match(@"normal\s*\*\s*max\s*\(\s*_SSGIThickness"));
        Assert.That(shader, Does.Not.Match(@"_SSGIMaxDistance\s*\*\s*lerp"));
    }

    [Test]
    public void Settings_MaxDistanceHasNoHardUpperLimit()
    {
        Type settingsType = typeof(ScreenSpaceGlobalIlluminationFeature.SSGISettings);
        FieldInfo maxDistance = settingsType.GetField("maxDistance");

        Assert.That(maxDistance, Is.Not.Null);
        Assert.That(maxDistance.GetCustomAttribute<UnityEngine.MinAttribute>(), Is.Not.Null);
        Assert.That(maxDistance.GetCustomAttribute<UnityEngine.RangeAttribute>(), Is.Null);

        string feature = ReadAsset("Scripts", "Runtime", "Features", "ScreenSpaceGlobalIlluminationFeature.cs");
        Assert.That(feature, Does.Not.Contain("Mathf.Min(settings.maxDistance"));
        Assert.That(feature, Does.Match(@"Mathf\.Clamp\s*\(\s*settings\.maxDistance\s*,\s*0\.1f\s*,\s*farClip\s*\)"));
    }

    [Test]
    public void RuntimeSettings_AreValidatedBeforeShaderUpload()
    {
        string feature = ReadAsset("Scripts", "Runtime", "Features", "ScreenSpaceGlobalIlluminationFeature.cs");
        Assert.That(feature, Does.Match(@"Mathf\.Clamp\s*\(\s*settings\.rayCount\s*,\s*1\s*,\s*128\s*\)"));
        Assert.That(feature, Does.Match(@"Mathf\.Clamp\s*\(\s*settings\.sampleOffset\s*,\s*0\s*,\s*65535\s*\)"));
        Assert.That(feature, Does.Match(@"Mathf\.Clamp\s*\(\s*settings\.maxSteps\s*,\s*1\s*,\s*256\s*\)"));
        Assert.That(feature, Does.Contain("IsFinite(settings.maxDistance)"));
        Assert.That(feature, Does.Contain("IsFinite(settings.thickness)"));
        Assert.That(feature, Does.Contain("IsFinite(settings.originBias)"));
        Assert.That(feature, Does.Contain("IsFinite(settings.intensity)"));
        Assert.That(feature, Does.Contain("IsFinite(settings.temporalResponse)"));
        Assert.That(feature, Does.Contain("IsFinite(settings.temporalDepthThreshold)"));
        Assert.That(feature, Does.Contain("IsFinite(settings.disocclusionFallback)"));
        Assert.That(feature, Does.Contain("Mathf.Clamp01(settings.disocclusionFallback)"));
        Assert.That(feature, Does.Match(@"Mathf\.Clamp\s*\(\s*settings\.maxHistoryFrames\s*,\s*2\s*,\s*2048\s*\)"));
        Assert.That(feature, Does.Not.Contain("settings.temporalNormalThreshold"));
        Assert.That(feature, Does.Not.Contain("settings.varianceThreshold"));
        Assert.That(feature, Does.Not.Contain("settings.adaptiveRayMultiplier"));
        Assert.That(feature, Does.Not.Contain("settings.varianceFilterStrength"));
    }

    [Test]
    public void SharedMarcher_IsGeometryOnlyAndUsesNeutralHitContract()
    {
        string utility = ReadAsset("Scripts", "Shaders", "GI", "ScreenSpaceRayMarch.hlsl");
        Assert.That(utility, Does.Contain("struct ScreenSpaceRayHit"));
        Assert.That(utility, Does.Match(@"bool\s+hit\s*;"));
        Assert.That(utility, Does.Match(@"float2\s+hitUV\s*;"));
        Assert.That(utility, Does.Match(@"float\s+rayT\s*;"));
        Assert.That(utility, Does.Match(@"float3\s+hitPosVS\s*;"));
        Assert.That(utility, Does.Not.Contain("SampleSceneColor"));
        Assert.That(utility, Does.Not.Match(@"float3\s+color\s*;"));
        Assert.That(Regex.Matches(utility, @"result\.hitPosVS\s*=\s*ComputeViewSpacePosition\s*\(").Count, Is.EqualTo(3));
        Assert.That(Regex.Matches(utility, @"result\.rayT\s*=\s*saturate\s*\(").Count, Is.EqualTo(3));

        string ssr = ReadAsset("Scripts", "Shaders", "SSR", "ScreenSpaceReflection.shader");
        Assert.That(ssr, Does.Contain("../GI/ScreenSpaceRayMarch.hlsl"));
        Assert.That(ssr, Does.Match(@"SampleSceneColor\s*\(\s*hit\.hitUV\s*\)"));
    }

    [Test]
    public void SharedMarcher_RoundTripsDepthTextureUVWithoutSecondVerticalFlip()
    {
        string utility = ReadAsset("Scripts", "Shaders", "GI", "ScreenSpaceRayMarch.hlsl");
        Assert.That(utility, Does.Match(@"float2\s+uv\s*=\s*\(clip\.xy\s*/\s*clip\.w\)\s*\*\s*0\.5\s*\+\s*0\.5"));
        Assert.That(utility, Does.Not.Match(@"uv\.y\s*=\s*1\.0\s*-\s*uv\.y"));

        const float sourceV = 0.2f;
        const float viewZ = 10.0f;
        float tanHalfFov = Mathf.Tan(60.0f * Mathf.Deg2Rad * 0.5f);
        float viewY = (1.0f - 2.0f * sourceV) * tanHalfFov * viewZ;
        float projectionFlippedNdcY = -viewY / (tanHalfFov * viewZ);
        float projectedV = projectionFlippedNdcY * 0.5f + 0.5f;

        Assert.That(projectedV, Is.EqualTo(sourceV).Within(1e-6f));
        Assert.That(Mathf.Abs((1.0f - projectedV) - sourceV), Is.GreaterThan(0.5f));
    }

    [Test]
    public void DdaMarcher_RefinesFirstDepthCrossingBeforeThicknessTest()
    {
        const float thickness = 0.02f;
        const float previousDepthDiff = -0.01f;
        const float currentDepthDiff = 0.03f;

        bool endpointOnlyHit = currentDepthDiff > 0.0f && currentDepthDiff < thickness;
        Assert.That(endpointOnlyHit, Is.False, "The coarse endpoint must reproduce the rear-wall false miss.");

        float loDepthDiff = previousDepthDiff;
        float hiDepthDiff = currentDepthDiff;
        for (int i = 0; i < 10; i++)
        {
            float midDepthDiff = (loDepthDiff + hiDepthDiff) * 0.5f;
            if (midDepthDiff > 0.0f)
                hiDepthDiff = midDepthDiff;
            else
                loDepthDiff = midDepthDiff;
        }

        Assert.That(hiDepthDiff, Is.GreaterThan(0.0f));
        Assert.That(hiDepthDiff, Is.LessThan(thickness));

        const float selfIntersectionDepthDiff = 0.005f;
        bool hasFrontSample = selfIntersectionDepthDiff <= 0.0f;
        bool positivePositiveBracket = hasFrontSample && currentDepthDiff > 0.0f;
        Assert.That(positivePositiveBracket, Is.False,
            "Two samples behind the surface must not form a refinement bracket.");

        const int jitteredStepCount = 7;
        const float maxJitter = 0.9375f;
        float stepPosition = maxJitter;
        for (int i = 0; i < jitteredStepCount; i++)
        {
            float advance = Mathf.Min(1.0f, jitteredStepCount - stepPosition);
            if (advance <= 0.0f)
                break;
            stepPosition += advance;
        }
        Assert.That(stepPosition, Is.EqualTo(jitteredStepCount).Within(1e-6f));

        string utility = ReadAsset("Scripts", "Shaders", "GI", "ScreenSpaceRayMarch.hlsl");
        int ddaStart = utility.IndexOf("ScreenSpaceRayHit MarchScreenSpaceRayDDA", StringComparison.Ordinal);
        int binaryStart = utility.IndexOf("ScreenSpaceRayHit MarchScreenSpaceRayBinary", StringComparison.Ordinal);
        Assert.That(ddaStart, Is.GreaterThanOrEqualTo(0));
        Assert.That(binaryStart, Is.GreaterThan(ddaStart));
        string ddaMarcher = utility.Substring(ddaStart, binaryStart - ddaStart);

        Assert.That(ddaMarcher, Does.Match(@"float2\s+prevUV"));
        Assert.That(ddaMarcher, Does.Match(@"float\s+prevInvZ"));
        Assert.That(ddaMarcher, Does.Match(@"bool\s+hasFrontSample\s*=\s*false"));
        Assert.That(ddaMarcher, Does.Match(@"if\s*\(\s*\(\s*1\.0\s*/\s*currentInvZ\s*\)\s*>\s*sceneZ\s*\)"));
        Assert.That(ddaMarcher, Does.Match(@"if\s*\(\s*hasFrontSample\s*\)[\s\S]*crossed\s*=\s*true"));
        Assert.That(ddaMarcher, Does.Match(@"prevUV\s*=\s*currentUV[\s\S]*prevInvZ\s*=\s*currentInvZ[\s\S]*hasFrontSample\s*=\s*true"));
        Assert.That(ddaMarcher, Does.Match(@"stepPosition\s*=\s*jit"));
        Assert.That(ddaMarcher, Does.Match(@"advance\s*=\s*min\s*\(\s*1\.0\s*,\s*\(float\)numSteps\s*-\s*stepPosition\s*\)"));
        Assert.That(ddaMarcher, Does.Match(@"for\s*\(\s*int\s+j\s*=\s*0\s*;\s*j\s*<\s*SCREEN_SPACE_RAY_BINARY_STEPS"));
        Assert.That(ddaMarcher, Does.Match(@"depthDiff\s*=\s*\(\s*1\.0\s*/\s*hiInvZ\s*\)\s*-\s*sceneZ"));
    }

    [Test]
    public void BinaryMarcher_ClipsVisibleSegmentBeforeAllocatingSteps()
    {
        Vector2 startUV = new Vector2(0.5f, 0.5f);
        Vector2 endUV = new Vector2(300.0f, 0.5f);
        Vector2 deltaUV = endUV - startUV;
        Vector2 screenSize = new Vector2(1920.0f, 1080.0f);
        Vector2 halfTexel = new Vector2(0.5f / screenSize.x, 0.5f / screenSize.y);
        Vector2 viewportMax = Vector2.one - halfTexel;
        const int maxSteps = 256;

        float firstUnclippedX = startUV.x + deltaUV.x / maxSteps;
        float exitT = (viewportMax.x - startUV.x) / deltaUV.x;
        Vector2 clippedEndUV = startUV + deltaUV * exitT;
        float originalInvZ0 = 1.0f / 10.0f;
        float originalInvZ1 = 1.0f / 0.3f;
        float clippedInvZ = Mathf.Lerp(originalInvZ0, originalInvZ1, exitT);

        Assert.That(firstUnclippedX, Is.GreaterThan(1.0f));
        Assert.That(exitT, Is.InRange(0.0f, 1.0f));
        Assert.That(clippedEndUV.x, Is.EqualTo(viewportMax.x).Within(1e-6f));
        Assert.That(clippedEndUV.y, Is.EqualTo(0.5f).Within(1e-6f));
        Assert.That(clippedInvZ, Is.EqualTo(originalInvZ0 + (originalInvZ1 - originalInvZ0) * exitT).Within(1e-7f));

        Vector2 subpixelStart = new Vector2(halfTexel.x, 0.5f);
        Vector2 subpixelEnd = subpixelStart + new Vector2(0.25f / screenSize.x, 0.0f);
        Vector2 subpixelDelta = subpixelEnd - subpixelStart;
        int twoStepCount = Mathf.Min(2, maxSteps);
        Vector2 twoStep = subpixelDelta / twoStepCount;
        Assert.That(subpixelStart + twoStep, Is.EqualTo((subpixelStart + subpixelEnd) * 0.5f));
        Assert.That(subpixelStart + twoStep * 2.0f, Is.EqualTo(subpixelEnd));

        const int singleStepCount = 1;
        Vector2 singleStep = subpixelDelta / singleStepCount;
        const float singleStepScale = 0.5f;
        Assert.That(subpixelStart + singleStep * singleStepScale, Is.EqualTo((subpixelStart + subpixelEnd) * 0.5f));

        Assert.That(ClipViewportSegment(new Vector2(0.5f, 0.25f), new Vector2(0.5f, 0.75f), screenSize, out _, out _), Is.True);
        Assert.That(ClipViewportSegment(new Vector2(-0.1f, 0.25f), new Vector2(-0.1f, 0.75f), screenSize, out _, out _), Is.False);
        Assert.That(ClipViewportSegment(new Vector2(-0.25f, 0.5f), new Vector2(0.25f, 0.5f), screenSize, out float entryT, out _), Is.True);
        Assert.That(entryT, Is.EqualTo((halfTexel.x + 0.25f) / 0.5f).Within(1e-6f));
        Assert.That(ClipViewportSegment(new Vector2(-0.25f, 0.5f), new Vector2(-0.1f, 0.5f), screenSize, out _, out _), Is.False);
        Assert.That(ClipViewportSegment(new Vector2(halfTexel.x - 1e-6f, 0.5f), new Vector2(halfTexel.x, 0.5f), screenSize, out float pointEnterT, out float pointExitT), Is.True);
        Assert.That(pointEnterT, Is.EqualTo(1.0f).Within(1e-6f));
        Assert.That(pointExitT, Is.EqualTo(1.0f).Within(1e-6f));

        string utility = ReadAsset("Scripts", "Shaders", "GI", "ScreenSpaceRayMarch.hlsl");
        Assert.That(utility, Does.Contain("ClipScreenSpaceRayToViewport"));
        Assert.That(utility, Does.Match(@"if\s*\(\s*direction\s*==\s*0\.0\s*\)"));
        Assert.That(utility, Does.Not.Match(@"abs\s*\(\s*direction\s*\)\s*<="));
        Assert.That(utility, Does.Match(@"halfTexel\s*=\s*0\.5\s*/\s*_ScreenParams\.xy"));
        Assert.That(utility, Does.Match(@"viewportMin\s*=\s*halfTexel"));
        Assert.That(utility, Does.Match(@"viewportMax\s*=\s*1\.0\s*-\s*halfTexel"));
        int binaryStart = utility.IndexOf("ScreenSpaceRayHit MarchScreenSpaceRayBinary", StringComparison.Ordinal);
        int hiZStart = utility.IndexOf("ScreenSpaceRayHit MarchScreenSpaceRayHiZ", StringComparison.Ordinal);
        Assert.That(binaryStart, Is.GreaterThanOrEqualTo(0));
        Assert.That(hiZStart, Is.GreaterThan(binaryStart));
        string binaryMarcher = utility.Substring(binaryStart, hiZStart - binaryStart);
        Assert.That(binaryMarcher, Does.Match(@"ClipScreenSpaceRayToViewport\s*\(\s*startUV\s*,\s*endUV\s*,\s*invZ0\s*,\s*invZ1\s*\)"));
        Assert.That(binaryMarcher, Does.Match(@"ceil\s*\(\s*maxPixelSpan\s*\)"));
        Assert.That(binaryMarcher, Does.Match(@"maxPixelSpan\s*<\s*1\.0[\s\S]*\?\s*min\s*\(\s*2\s*,\s*safeMaxSteps\s*\)"));
        Assert.That(binaryMarcher, Does.Match(@"if\s*\(\s*maxPixelSpan\s*>=\s*1\.0\s*\)[\s\S]*GetJitter"));
        Assert.That(binaryMarcher, Does.Match(@"firstStepScale\s*=\s*maxPixelSpan\s*<\s*1\.0\s*&&\s*numSteps\s*==\s*1\s*\?\s*0\.5\s*:\s*1\.0"));
        Assert.That(binaryMarcher, Does.Match(@"advance\s*=\s*i\s*==\s*0\s*\?\s*firstStepScale\s*:\s*1\.0"));
        int clipCall = binaryMarcher.IndexOf("ClipScreenSpaceRayToViewport", StringComparison.Ordinal);
        int deltaPixelCalculation = binaryMarcher.IndexOf("float2 deltaPixel", StringComparison.Ordinal);
        int stepCountCalculation = binaryMarcher.IndexOf("int    numSteps", StringComparison.Ordinal);
        Assert.That(clipCall, Is.LessThan(deltaPixelCalculation));
        Assert.That(clipCall, Is.LessThan(stepCountCalculation));
    }

    [Test]
    public void SharedMarcher_ClipsEveryRayToCameraNearPlane()
    {
        string utility = ReadAsset("Scripts", "Shaders", "GI", "ScreenSpaceRayMarch.hlsl");
        Assert.That(utility, Does.Contain("SCREEN_SPACE_RAY_NEAR_EPSILON"));
        Assert.That(utility, Does.Contain("ClipScreenSpaceRaySegment"));
        Assert.That(utility, Does.Match(@"nearPlane\s*=\s*max\s*\(\s*_ProjectionParams\.y\s*,\s*SCREEN_SPACE_RAY_NEAR_EPSILON\s*\)"));
        Assert.That(utility, Does.Match(@"rayStartVS\.z\s*<=\s*nearPlane"));
        Assert.That(utility, Does.Match(@"rayEndVS\.z\s*<=\s*nearPlane"));
        Assert.That(utility, Does.Match(@"clipT\s*=\s*\(\s*rayStartVS\.z\s*-\s*nearPlane\s*\)\s*/\s*\(\s*rayStartVS\.z\s*-\s*rayEndVS\.z\s*\)"));
        Assert.That(utility, Does.Match(@"rayEndVS\.z\s*=\s*nearPlane"));
        Assert.That(utility, Does.Not.Match(@"rayEndVS\.z\s*=\s*SCREEN_SPACE_RAY_NEAR_EPSILON"));
        Assert.That(Regex.Matches(utility, @"if\s*\(\s*!ClipScreenSpaceRaySegment\s*\(").Count, Is.EqualTo(3));
        Assert.That(Regex.Matches(utility, @"clamp\s*\(\s*maxSteps\s*,\s*1\s*,\s*256\s*\)").Count, Is.EqualTo(3));
    }

    [Test]
    public void HiZTraversal_DoesNotStepPastFiniteRaySegment()
    {
        string utility = ReadAsset("Scripts", "Shaders", "GI", "ScreenSpaceRayMarch.hlsl");
        Assert.That(utility, Does.Match(@"float\s+traveled\s*="));
        Assert.That(utility, Does.Match(@"if\s*\(\s*traveled\s*\+\s*stride\s*>\s*totalDist\s*\)\s*break\s*;"));
        Assert.That(utility, Does.Match(@"traveled\s*\+=\s*stride\s*;"));
        Assert.That(utility, Does.Match(@"float\s+t\s*=\s*traveled\s*/\s*totalDist\s*;"));
        Assert.That(utility, Does.Not.Match(@"float\s+t\s*=\s*saturate"));
    }

    [Test]
    public void EdgeAwareFilter_WeightsDepthAndNormalsAndPreservesCenterDepth()
    {
        Vector3 forward = Vector3.forward;
        Vector3[] uniformColors =
        {
            Vector3.one,
            Vector3.one * 2.0f,
            Vector3.one * 3.0f,
            Vector3.one * 4.0f,
            Vector3.one * 5.0f
        };
        float[] uniformDepths = { 2.0f, 2.0f, 2.0f, 2.0f, 2.0f };
        Vector3[] uniformNormals = { forward, forward, forward, forward, forward };
        bool[] allValid = { true, true, true, true, true };

        Vector4 uniform = FilterEdgeAwareSamples(
            uniformColors, uniformDepths, uniformNormals, allValid, 2.0f, forward);
        Assert.That(uniform.x, Is.EqualTo(3.0f).Within(1e-6f));
        Assert.That(uniform.y, Is.EqualTo(3.0f).Within(1e-6f));
        Assert.That(uniform.z, Is.EqualTo(3.0f).Within(1e-6f));
        Assert.That(uniform.w, Is.EqualTo(2.0f), "The center linear depth must survive both filter passes unchanged.");

        float depthEdgeWeight = Mathf.Exp(-5.0f);
        Vector4 depthEdge = FilterEdgeAwareSamples(
            new[] { Vector3.one, Vector3.one * 10.0f },
            new[] { 2.0f, 3.0f },
            new[] { forward, forward },
            new[] { true, true },
            2.0f,
            forward);
        float expectedDepthEdge = (1.0f + 10.0f * depthEdgeWeight) / (1.0f + depthEdgeWeight);
        Assert.That(depthEdge.x, Is.EqualTo(expectedDepthEdge).Within(1e-6f));

        Assert.That(EdgeAwareWeight(2.0f, forward, 2.0f, Vector3.right, true), Is.Zero);
        Assert.That(EdgeAwareWeight(2.0f, forward, 2.0f, Vector3.back, true), Is.Zero);
        Assert.That(EdgeAwareWeight(2.0f, forward, -1.0f, forward, true), Is.Zero);
        Assert.That(EdgeAwareWeight(2.0f, forward, 2.0f, forward, false), Is.Zero);
    }

    [Test]
    public void SSGIBlur_UsesIterativeFiveTapDepthNormalCrossFilter()
    {
        string blur = ReadAsset("Scripts", "Shaders", "GI", "SSGIBlur.hlsl");
        string sharedBlur = ReadAsset("Scripts", "Shaders", "GI", "Blur.hlsl");
        string shader = ReadAsset("Scripts", "Shaders", "SSGI", "ScreenSpaceGlobalIllumination.shader");
        string feature = ReadAsset("Scripts", "Runtime", "Features", "ScreenSpaceGlobalIlluminationFeature.cs");

        Assert.That(blur, Does.Contain("SampleSSGIEdgeAwareFilter"));
        Assert.That(blur, Does.Match(@"SSGI_FILTER_OFFSETS\s*\[\s*5\s*\]"));
        Assert.That(blur, Does.Match(@"float2\s*\(\s*0(?:\.0)?\s*,\s*0(?:\.0)?\s*\)"));
        Assert.That(blur, Does.Match(@"float2\s*\(\s*1(?:\.0)?\s*,\s*0(?:\.0)?\s*\)"));
        Assert.That(blur, Does.Match(@"float2\s*\(\s*-1(?:\.0)?\s*,\s*0(?:\.0)?\s*\)"));
        Assert.That(blur, Does.Match(@"float2\s*\(\s*0(?:\.0)?\s*,\s*1(?:\.0)?\s*\)"));
        Assert.That(blur, Does.Match(@"float2\s*\(\s*0(?:\.0)?\s*,\s*-1(?:\.0)?\s*\)"));
        Assert.That(blur, Does.Contain("ReconstructSSGIFilterNormal"));
        Assert.That(blur, Does.Contain("GetNormalFromPosition(viewPos)"));
        Assert.That(blur, Does.Match(@"centerDepth\s*=\s*center\.a"));
        Assert.That(blur, Does.Match(@"sampleDepth\s*=\s*sample\.a"));
        Assert.That(blur, Does.Contain("SampleSceneDepth"));
        Assert.That(blur, Does.Match(@"exp\s*\(\s*-SSGI_FILTER_DEPTH_FALLOFF\s*\*\s*abs\s*\(\s*centerDepth\s*-\s*sampleDepth\s*\)\s*\)"));
        Assert.That(blur, Does.Match(@"saturate\s*\(\s*dot\s*\(\s*centerNormal\s*,\s*sampleNormal\s*\)\s*\)"));
        Assert.That(blur, Does.Not.Contain("_SSGIMomentsTexture"));
        Assert.That(blur, Does.Not.Contain("_SSGINormalCountTexture"));
        Assert.That(blur, Does.Not.Contain("_SSGIHistorySampleTexture"));
        Assert.That(blur, Does.Not.Contain("adaptiveFilterRadius"));
        Assert.That(blur, Does.Match(@"SSGI_FILTER_OFFSETS\s*\[\s*i\s*\]\s*\*\s*texelSize\s*\*\s*filterRadius"));
        Assert.That(blur, Does.Match(@"weight\s*=\s*depthWeight\s*\*\s*normalWeight"));
        Assert.That(blur, Does.Not.Contain("luminanceWeight"));
        Assert.That(blur, Does.Match(@"max\s*\(\s*0\.001\s*,\s*weightSum\s*\)"));
        Assert.That(blur, Does.Match(@"return\s+float4\s*\(\s*filtered\s*,\s*center\.a\s*\)"));
        Assert.That(blur, Does.Match(@"sampleUV\.x\s*<\s*0\.0[\s\S]*sampleUV\.x\s*>\s*1\.0"));
        Assert.That(blur, Does.Match(@"sample\.a\s*<\s*0\.0"));
        Assert.That(blur, Does.Not.Contain("GI_BLUR_WEIGHT"));
        Assert.That(sharedBlur, Does.Contain("SampleGaussianBlurHorizontal"));
        Assert.That(sharedBlur, Does.Contain("SampleGaussianBlurVertical"));
        Assert.That(shader, Does.Contain("../GI/SSGIBlur.hlsl"));

        Assert.That(Regex.Matches(shader, "DeclareDepthTexture.hlsl").Count, Is.EqualTo(5));
        Assert.That(Regex.Matches(shader, "../GI/Commond.hlsl").Count, Is.EqualTo(3));
        Assert.That(shader, Does.Match(@"FragBlurH[\s\S]*SampleSSGIEdgeAwareFilter\s*\([\s\S]*input\.texcoord\s*,\s*_BlurSpread\s*\)"));
        Assert.That(shader, Does.Match(@"FragBlurV[\s\S]*SampleSSGIEdgeAwareFilter\s*\([\s\S]*input\.texcoord\s*,\s*_BlurSpread\s*\*\s*2\.0\s*\)"));
        Assert.That(feature, Does.Match(@"(?i)base edge-aware filter radius"));
        Assert.That(feature, Does.Match(@"material\s*,\s*1\s*\)[\s\S]*material\s*,\s*2\s*\)"));
        Assert.That(feature, Does.Match(@"ConfigureInput\s*\([\s\S]*ScriptableRenderPassInput\.Depth\s*\|[\s\S]*ScriptableRenderPassInput\.Color\s*\|[\s\S]*ScriptableRenderPassInput\.Motion\s*\)"));
    }

    [Test]
    public void SSGI_UsesConfiguredRayCountWithoutAdaptiveRayCost()
    {
        string shader = ReadAsset("Scripts", "Shaders", "SSGI", "ScreenSpaceGlobalIllumination.shader");

        Assert.That(shader, Does.Match(@"rayCount\s*=\s*clamp\s*\(\s*_SSGIRayCount\s*,\s*1\s*,\s*128\s*\)"));
        Assert.That(shader, Does.Not.Contain("adaptiveRayCount"));
        Assert.That(shader, Does.Not.Contain("_SSGIVarianceThreshold"));
        Assert.That(shader, Does.Not.Contain("temporal.moments"));
    }

    [Test]
    public void MainLightDirect_AccumulatesPerPixelAdditionalLights()
    {
        string shader = ReadAsset("Scripts", "Shaders", "Objects", "MainLightDirect.shader");

        Assert.That(shader, Does.Match(@"#pragma\s+multi_compile\s+_\s+_ADDITIONAL_LIGHTS"));
        Assert.That(shader, Does.Match(@"#if\s+defined\s*\(\s*_ADDITIONAL_LIGHTS\s*\)"));
        Assert.That(shader, Does.Match(@"GetAdditionalLightsCount\s*\(\s*\)"));
        Assert.That(shader, Does.Match(@"LIGHT_LOOP_BEGIN\s*\(\s*additionalLightCount\s*\)"));
        Assert.That(shader, Does.Contain("LIGHT_LOOP_END"));
        Assert.That(shader, Does.Match(@"GetAdditionalLight\s*\(\s*lightIndex\s*,\s*input\.positionWS\s*\)"));
        Assert.That(shader, Does.Match(@"additionalNdotL\s*=\s*saturate\s*\(\s*dot\s*\(\s*normalWS\s*,\s*additionalLight\.direction\s*\)\s*\)"));
        Assert.That(shader, Does.Match(@"color\s*\+=\s*baseColor\.rgb\s*\*\s*additionalLight\.color\s*\*\s*additionalNdotL"));
        Assert.That(shader, Does.Match(@"additionalLight\.distanceAttenuation\s*\*\s*additionalLight\.shadowAttenuation"));
        Assert.That(shader, Does.Match(@"baseColor\.rgb\s*\*\s*mainLight\.color\s*\*\s*ndotl\s*\*\s*mainLight\.shadowAttenuation"));
    }

    [Test]
    public void LegacyObjectIrradiancePath_IsDisabledBeforeOpaqueRendering()
    {
        string shader = ReadAsset("Scripts", "Shaders", "Objects", "MainLightDirect.shader");

        Assert.That(shader, Does.Contain("TEXTURE2D_X(_SSGIIrradianceTexture)"));
        Assert.That(shader, Does.Contain("_SSGIIrradianceValid"));
        Assert.That(shader, Does.Contain("_SSGIPreviousViewProjectionMatrix"));
        Assert.That(shader, Does.Contain("_SSGIPreviousWorldToCameraMatrix"));
        Assert.That(shader, Does.Contain("_SSGIReprojectIrradiance"));
        Assert.That(shader, Does.Contain("_SSGIDisocclusionFallback"));
        Assert.That(shader, Does.Match(@"screenUV\s*=\s*GetNormalizedScreenSpaceUV\s*\(\s*input\.positionCS\s*\)"));
        Assert.That(shader, Does.Match(@"requiresDepthValidation\s*=\s*_SSGIReprojectIrradiance\s*>\s*0\.5"));
        Assert.That(shader, Does.Match(@"previousCS\s*=\s*mul\s*\("));
        Assert.That(shader, Does.Match(@"previousUV\s*=\s*previousCS\.xy\s*/[\s\S]*max\s*\(\s*previousCS\.w"));
        Assert.That(shader, Does.Match(@"reprojectedDepth\s*=\s*SAMPLE_TEXTURE2D_X\s*\([\s\S]*sampler_PointClamp\s*,[\s\S]*previousUV\s*\)\.a"));
        Assert.That(shader, Does.Match(@"abs\s*\(\s*reprojectedDepth\s*-\s*expectedDepth\s*\)\s*<=\s*depthThreshold"));
        Assert.That(shader, Does.Match(@"indirectConfidence\s*<=\s*0\.0\s*&&\s*_SSGIDisocclusionFallback\s*>\s*0\.0"));
        Assert.That(shader, Does.Match(@"fallbackIrradiance\s*=\s*SAMPLE_TEXTURE2D_X\s*\([\s\S]*sampler_LinearClamp\s*,\s*screenUV\s*\)"));
        Assert.That(shader, Does.Match(@"color\s*\+=\s*indirectIrradiance\s*\*\s*baseColor\.rgb\s*\*\s*indirectConfidence"));

        string feature = ReadAsset("Scripts", "Runtime", "Features", "ScreenSpaceGlobalIlluminationFeature.cs");
        Assert.That(feature, Does.Contain("_SSGIReprojectIrradiance"));
        Assert.That(feature, Does.Contain("_SSGIDisocclusionFallback"));
        Assert.That(feature, Does.Match(@"SetGlobalFloat\s*\(\s*IrradianceValidID\s*,\s*0\.0f\s*\)"));
        Assert.That(feature, Does.Match(@"SetGlobalFloat\s*\(\s*ReprojectIrradianceID\s*,\s*0\.0f\s*\)"));
    }

    [Test]
    public void PBRShader_UsesMaterialParametersAndSupportsRealtimeShadows()
    {
        string shader = ReadAsset("Scripts", "Shaders", "Objects", "PBRShader.shader");

        Assert.That(shader, Does.Match(@"_BaseColorMap\s*\(\s*""Albedo Map""\s*,\s*2D\s*\)"));
        Assert.That(shader, Does.Match(@"_BaseColor\s*\(\s*""Albedo""\s*,\s*Color\s*\)"));
        Assert.That(shader, Does.Match(@"_Roughness\s*\(\s*""Roughness""\s*,\s*Range\s*\(\s*0\s*,\s*1\s*\)\s*\)"));
        Assert.That(shader, Does.Match(@"_Metallic\s*\(\s*""Metallic""\s*,\s*Range\s*\(\s*0\s*,\s*1\s*\)\s*\)"));
        Assert.That(shader, Does.Match(@"SAMPLE_TEXTURE2D\s*\(\s*_BaseColorMap\s*,\s*sampler_BaseColorMap\s*,\s*uv\s*\)\s*\*\s*_BaseColor"));
        Assert.That(shader, Does.Match(@"surfaceData\.albedo\s*=\s*baseColor\.rgb"));
        Assert.That(shader, Does.Match(@"surfaceData\.metallic\s*=\s*saturate\s*\(\s*_Metallic\s*\)"));
        Assert.That(shader, Does.Match(@"surfaceData\.smoothness\s*=\s*1\.0h\s*-\s*saturate\s*\(\s*_Roughness\s*\)"));
        Assert.That(shader, Does.Contain("InitializeBRDFData(surfaceData, brdfData)"));
        Assert.That(shader, Does.Contain("CalculateDirectPBRLighting(inputData, surfaceData)"));
        Assert.That(shader, Does.Contain("LightingPhysicallyBased("));
        Assert.That(shader, Does.Contain("GetMainLight(inputData, shadowMask, aoFactor)"));
        Assert.That(shader, Does.Contain("GetAdditionalLight(lightIndex, inputData, shadowMask, aoFactor)"));
        Assert.That(shader, Does.Not.Contain("UniversalFragmentPBR("));
        Assert.That(shader, Does.Not.Contain("SampleSH("));
        Assert.That(shader, Does.Not.Contain("GlobalIllumination("));
        Assert.That(shader, Does.Not.Contain("GlossyEnvironmentReflection("));
        Assert.That(shader, Does.Not.Contain("MixFog("));
        Assert.That(shader, Does.Match(@"LightMode""\s*=\s*""UniversalForward"));
        Assert.That(shader, Does.Match(@"LightMode""\s*=\s*""ShadowCaster"));
        Assert.That(shader, Does.Contain("_MAIN_LIGHT_SHADOWS"));
        Assert.That(shader, Does.Contain("_ADDITIONAL_LIGHT_SHADOWS"));
        Assert.That(shader, Does.Contain("ApplyShadowBias"));
        Assert.That(shader, Does.Contain("ApplyAlphaClip(SampleBaseColor(input.uv).a)"));
        Assert.That(shader, Does.Match(@"Cull\s*\[\s*_CullMode\s*\]"));
        Assert.That(shader, Does.Not.Match(@"pow\s*\(\s*color\s*,\s*2\.2"));
    }

    [Test]
    public void SponzaMaterials_UsePBRShaderAndPreserveLayerZeroAlbedoMaps()
    {
        const string pbrShaderGuid = "b8f71ddc42f44b8797fb86c5ce344f86";
        string materialsPath = ProjectPath("SponzaHDRP", "Art", "Sponza", "Materials");
        string[] materialPaths = Directory.GetFiles(materialsPath, "*.mat", SearchOption.TopDirectoryOnly);

        Assert.That(materialPaths, Has.Length.EqualTo(33));
        foreach (string materialPath in materialPaths)
        {
            string material = File.ReadAllText(materialPath);
            Assert.That(material, Does.Match($@"m_Shader:.*guid:\s*{pbrShaderGuid}"), Path.GetFileName(materialPath));

            if (!Path.GetFileNameWithoutExtension(materialPath).Contains("Layered"))
                continue;

            Match baseMap = Regex.Match(material, @"- _BaseColorMap:\s*\r?\n\s*m_Texture:.*guid:\s*([0-9a-f]+)");
            Match layerZeroMap = Regex.Match(material, @"- _BaseColorMap0:\s*\r?\n\s*m_Texture:.*guid:\s*([0-9a-f]+)");
            Assert.That(baseMap.Success, Is.True, Path.GetFileName(materialPath));
            Assert.That(layerZeroMap.Success, Is.True, Path.GetFileName(materialPath));
            Assert.That(baseMap.Groups[1].Value, Is.EqualTo(layerZeroMap.Groups[1].Value), Path.GetFileName(materialPath));
        }
    }

    [Test]
    public void ForwardGBufferPass_RendersProjectSpecificFourTargetLayout()
    {
        string shader = ReadAsset("Scripts", "Shaders", "Objects", "PBRShader.shader");
        string feature = ReadAsset("Scripts", "Runtime", "Features", "ScreenSpaceGlobalIlluminationFeature.cs");

        Assert.That(shader, Does.Match(@"LightMode""\s*=\s*""SSGIForwardGBuffer"));
        Assert.That(shader, Does.Match(@"albedo\s*:\s*SV_Target0"));
        Assert.That(shader, Does.Match(@"material\s*:\s*SV_Target1"));
        Assert.That(shader, Does.Match(@"normalWS\s*:\s*SV_Target2"));
        Assert.That(shader, Does.Match(@"positionWS\s*:\s*SV_Target3"));
        Assert.That(shader, Does.Match(@"material\s*=\s*half4\s*\(\s*saturate\s*\(\s*_Metallic\s*\)\s*,\s*saturate\s*\(\s*_Roughness\s*\)"));

        Assert.That(feature, Does.Contain("class ForwardGBufferPass"));
        Assert.That(feature, Does.Contain("new ShaderTagId(\"SSGIForwardGBuffer\")"));
        Assert.That(feature, Does.Match(@"renderPassEvent\s*=\s*RenderPassEvent\.AfterRenderingOpaques"));
        Assert.That(feature, Does.Match(@"new\s+RTHandle\s*\[\s*4\s*\]"));
        Assert.That(feature, Does.Contain("_SSGIGBufferAlbedo"));
        Assert.That(feature, Does.Contain("_SSGIGBufferMaterial"));
        Assert.That(feature, Does.Contain("_SSGIGBufferNormalWS"));
        Assert.That(feature, Does.Contain("_SSGIGBufferPositionWS"));
        Assert.That(feature, Does.Contain("GraphicsFormat.R16G16B16A16_SFloat"));
        Assert.That(feature, Does.Contain("CompareFunction.Equal"));
        Assert.That(feature, Does.Match(@"ConfigureClear\s*\(\s*ClearFlag\.Color\s*\|\s*ClearFlag\.Depth"));
        Assert.That(feature, Does.Contain("context.DrawRenderers"));
    }

    [Test]
    public void SSGITracePass_ConsumesWorldSpaceGBufferInsteadOfReconstructingFromDepth()
    {
        string shader = ReadAsset("Scripts", "Shaders", "SSGI", "ScreenSpaceGlobalIllumination.shader");

        // The trace pass must read albedo, world-space normal and world-space
        // position from the ForwardGBufferPass MRT output instead of relying
        // on cross(ddx, ddy) reconstruction from depth.
        Assert.That(shader, Does.Contain("TEXTURE2D_X(_SSGIGBufferAlbedo)"));
        Assert.That(shader, Does.Contain("TEXTURE2D_X(_SSGIGBufferNormalWS)"));
        Assert.That(shader, Does.Contain("TEXTURE2D_X(_SSGIGBufferPositionWS)"));
        Assert.That(shader, Does.Match(@"positionData\s*=\s*SAMPLE_TEXTURE2D_X\s*\([\s\S]*_SSGIGBufferPositionWS[\s\S]*uv\s*\)"));
        Assert.That(shader, Does.Match(@"normalData\s*=\s*SAMPLE_TEXTURE2D_X\s*\([\s\S]*_SSGIGBufferNormalWS[\s\S]*uv\s*\)"));
        Assert.That(shader, Does.Match(@"hitAlbedo\s*=\s*SAMPLE_TEXTURE2D_X\s*\([\s\S]*_SSGIGBufferAlbedo[\s\S]*hit\.hitUV\s*\)\.rgb"));

        // The shared marcher stays in view space, so the trace pass converts
        // world-space G-buffer data back into view space at the call site.
        Assert.That(shader, Does.Match(@"mul\s*\(\s*UNITY_MATRIX_V\s*,\s*float4\s*\(\s*positionData\.xyz\s*,\s*1\.0\s*\)\s*\)\.xyz"));
        Assert.That(shader, Does.Match(@"mul\s*\(\s*\(float3x3\)UNITY_MATRIX_V\s*,\s*normalData\.xyz\s*\)"));
    }

    [Test]
    public void SSGITracePass_CountsMissesAsZeroValuedMonteCarloSamples()
    {
        string shader = ReadAsset("Scripts", "Shaders", "SSGI", "ScreenSpaceGlobalIllumination.shader");

        Assert.That(shader, Does.Not.Contain("validHitCount"));
        Assert.That(shader, Does.Not.Match(@"-surfaceDepth"));
        Assert.That(shader, Does.Match(@"irradiance\s*/\s*rayCount\s*\*\s*_SSGIIntensity"));
        Assert.That(shader, Does.Match(@"return\s+float4\s*\(\s*outgoingRadiance\s*,\s*surfaceDepth\s*\)"));
        Assert.That(shader, Does.Match(
            @"!hasCurrentObservation[\s\S]*temporal\.valid[\s\S]*temporal\.irradianceDepth\.rgb"));
    }

    [Test]
    public void SSGITracePass_RejectsNearOriginSelfIntersections()
    {
        string shader = ReadAsset("Scripts", "Shaders", "SSGI", "ScreenSpaceGlobalIllumination.shader");

        Assert.That(shader, Does.Match(
            @"minimumHitDistance\s*=\s*max\s*\(\s*_SSGIOriginBias\s*\*\s*2\.0\s*,\s*_SSGIThickness\s*\)"));
        Assert.That(shader, Does.Match(
            @"hitDistance\s*=\s*distance\s*\(\s*hit\.hitPosVS\s*,\s*rayStartVS\s*\)"));
        Assert.That(shader, Does.Match(
            @"hitDistance\s*<=\s*minimumHitDistance[\s\S]*continue"));
    }

    [Test]
    public void ScreenSpaceProjection_RoundTripsTextureYConvention()
    {
        string marcher = ReadAsset("Scripts", "Shaders", "GI", "ScreenSpaceRayMarch.hlsl");
        string blur = ReadAsset("Scripts", "Shaders", "GI", "SSGIBlur.hlsl");

        Assert.That(marcher, Does.Match(
            @"ProjectVStoUV[\s\S]*UNITY_UV_STARTS_AT_TOP[\s\S]*uv\.y\s*=\s*1\.0\s*-\s*uv\.y"));
        Assert.That(blur, Does.Match(@"texelSize\s*=\s*1\.0\s*/\s*_ScreenParams"));
        Assert.That(blur, Does.Not.Contain("_BlitTextureSize"));
    }

    [Test]
    public void SSGIRandomLibrary_ProvidesBothR2AndHashRNGsBehindKeyword()
    {
        string randomLibrary = ReadAsset("Scripts", "Shaders", "GI", "SSGIRandom.hlsl");

        // R2 low-discrepancy sequence (legacy, deterministic, prefix-independent).
        Assert.That(randomLibrary, Does.Contain("SSGI_R2_STEP24"));
        Assert.That(randomLibrary, Does.Contain("SSGIHashRandom"));
        Assert.That(randomLibrary, Does.Contain("SSGIBuildHemisphereDirection"));

        // Hash + frame-counter RNG (UnitySSGIURP parity).
        Assert.That(randomLibrary, Does.Contain("SSGIGenerateRandomValueHash"));
        Assert.That(randomLibrary, Does.Contain("_SSGIRngFrameIndex"));
        Assert.That(randomLibrary, Does.Contain("_SSGIRngSeed"));
        Assert.That(randomLibrary, Does.Match(@"GenerateHashedRandomFloat\s*\(\s*uint3"));
        Assert.That(randomLibrary, Does.Match(@"#if\s+defined\s*\(\s*_SSGI_RNG_HASH\s*\)"));
        Assert.That(randomLibrary, Does.Match(@"#else"));

        // Both RNGs feed the same cosine-weighted hemisphere distribution.
        Assert.That(randomLibrary, Does.Match(@"SampleHemisphereCosine\s*\(\s*xiX\s*,\s*xiY\s*,\s*normal\s*\)"));
        Assert.That(randomLibrary, Does.Match(@"SampleHemisphereCosine\s*\(\s*xi\.x\s*,\s*xi\.y\s*,\s*normal\s*\)"));

        string feature = ReadAsset("Scripts", "Runtime", "Features", "ScreenSpaceGlobalIlluminationFeature.cs");
        Assert.That(feature, Does.Contain("public enum SSGIRngMode"));
        Assert.That(feature, Does.Match(@"R2\s*=\s*0"));
        Assert.That(feature, Does.Match(@"Hash\s*=\s*1"));
        Assert.That(feature, Does.Match(@"public\s+SSGIRngMode\s+rngMode"));
        Assert.That(feature, Does.Match(@"settings\.rngMode\s*==\s*SSGIRngMode\.Hash"));
        Assert.That(feature, Does.Contain("\"_SSGI_RNG_HASH\""));
        Assert.That(feature, Does.Contain("_SSGIRngFrameIndex"));
        Assert.That(feature, Does.Contain("_SSGIRngSeed"));
    }

    [Test]
    public void HalfResolutionSetting_DrivesTraceAndCompositePipeline()
    {
        Type settingsType = typeof(ScreenSpaceGlobalIlluminationFeature.SSGISettings);
        FieldInfo halfResolution = settingsType.GetField("halfResolution");
        Assert.That(halfResolution, Is.Not.Null);
        Assert.That(halfResolution.GetValue(Activator.CreateInstance(settingsType)), Is.EqualTo(true));

        string feature = ReadAsset("Scripts", "Runtime", "Features", "ScreenSpaceGlobalIlluminationFeature.cs");
        Assert.That(feature, Does.Match(@"public\s+bool\s+halfResolution\s*=\s*true"));
        Assert.That(feature, Does.Match(@"if\s*\(\s*settings\.halfResolution\s*\)"));
        Assert.That(feature, Does.Match(@"hash\s*=\s*hash\s*\*\s*31\s*\+\s*\(value\.halfResolution\s*\?\s*1\s*:\s*0\)"));

        // Working textures use the half-res descriptor; applyRT stays full-res.
        Assert.That(feature, Does.Match(@"ReAllocateIfNeeded\s*\(\s*ref\s+state\.radianceRT\s*,\s*traceDesc"));
        Assert.That(feature, Does.Match(@"ReAllocateIfNeeded\s*\(\s*ref\s+state\.blurA\s*,\s*traceDesc"));
        Assert.That(feature, Does.Match(@"ReAllocateIfNeeded\s*\(\s*ref\s+state\.blurB\s*,\s*traceDesc"));
        Assert.That(feature, Does.Match(@"ReAllocateIfNeeded\s*\(\s*ref\s+state\.historyA\s*,\s*traceDesc"));
        Assert.That(feature, Does.Match(@"ReAllocateIfNeeded\s*\(\s*ref\s+state\.historyB\s*,\s*traceDesc"));
        Assert.That(feature, Does.Match(@"ReAllocateIfNeeded\s*\(\s*ref\s+state\.historySampleA\s*,\s*sampleCountDesc"));
        Assert.That(feature, Does.Match(@"ReAllocateIfNeeded\s*\(\s*ref\s+state\.historySampleB\s*,\s*sampleCountDesc"));
        Assert.That(feature, Does.Match(@"ReAllocateIfNeeded\s*\(\s*ref\s+state\.applyRT\s*,\s*desc"));
        Assert.That(feature, Does.Match(@"state\.applyRT\?\.Release"));
    }

    [Test]
    public void CombinePass_PerformsDepthAwareUpsampleAndAlbedoModulatedComposite()
    {
        string shader = ReadAsset("Scripts", "Shaders", "SSGI", "ScreenSpaceGlobalIllumination.shader");

        Assert.That(shader, Does.Contain("Name \"Combine\""));
        Assert.That(shader, Does.Contain("SSGICombineDepthAwareUpsample"));

        // The upsample selects the closest of four half-res samples using
        // current world position and world-normal agreement.
        Assert.That(shader, Does.Match(@"float4\s+centerPosition"));
        Assert.That(shader, Does.Match(@"length\s*\(\s*p0\.xyz\s*-\s*centerPosition\.xyz\s*\)"));
        Assert.That(shader, Does.Match(@"dot\s*\(\s*n0\s*,\s*centerNormal\s*\)"));
        Assert.That(shader, Does.Match(@"bestDistance\s*=\s*min\s*\(\s*min\s*\(\s*distances\.x\s*,\s*distances\.y\s*\)\s*,\s*min\s*\(\s*distances\.z\s*,\s*distances\.w\s*\)\s*\)"));
        Assert.That(shader, Does.Match(@"_SSGIIrradianceTexture_TexelSize"));

        // Composite formula: cameraColor + indirectDiffuse * diffuse albedo.
        Assert.That(shader, Does.Match(@"diffuseReflectance\s*=\s*albedo\s*\*\s*\(\s*1\.0\s*-\s*saturate\s*\(\s*metallic\s*\)\s*\)"));
        Assert.That(shader, Does.Match(@"cameraColor\s*\+\s*indirectDiffuse\s*\*\s*diffuseReflectance"));
        Assert.That(shader, Does.Contain("TEXTURE2D_X(_SSGIGBufferAlbedo)"));
        Assert.That(shader, Does.Contain("TEXTURE2D_X(_SSGIGBufferMaterial)"));
        Assert.That(shader, Does.Contain("TEXTURE2D_X(_SSGIGBufferNormalWS)"));
        Assert.That(shader, Does.Contain("TEXTURE2D_X(_SSGIGBufferPositionWS)"));

        string feature = ReadAsset("Scripts", "Runtime", "Features", "ScreenSpaceGlobalIlluminationFeature.cs");
        Assert.That(feature, Does.Match(@"IrradianceTexelSizeID\s*=\s*Shader\.PropertyToID\s*\(\s*""_SSGIIrradianceTexture_TexelSize""\s*\)"));
        Assert.That(feature, Does.Match(@"sourceHandle\s*,\s*state\.applyRT\s*,\s*material\s*,\s*5\s*\)"));
    }
}
