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

        string shader = ReadAsset("Scripts", "Shaders", "SSGI", "ScreenSpaceGlobalIllumination.shader");
        Assert.That(shader, Does.Match(@"#pragma\s+target\s+3\.5"));
        Assert.That(shader, Does.Contain("ShaderLibrary/Sampling/Sampling.hlsl"));
        Assert.That(shader, Does.Match(@"clamp\s*\(\s*_SSGISampleOffset\s*,\s*0\s*,\s*65535\s*\)"));
        Assert.That(shader, Does.Contain("HashRandom"));
        Assert.That(shader, Does.Match(@"SSGI_R2_STEP24\s*=\s*uint2\s*\(\s*12664746u\s*,\s*9560334u\s*\)"));
        Assert.That(shader, Does.Match(@"sampleIndex\s*=\s*\(\s*\(uint\)clamp[\s\S]*&\s*0xFFFFu"));
        Assert.That(shader, Does.Match(@"phase24\s*=\s*\(\s*uint2\s*\(\s*sampleIndex\s*,\s*sampleIndex\s*\)\s*\*\s*SSGI_R2_STEP24\s*\)\s*&"));
        Assert.That(shader, Does.Match(@"frac\s*\(\s*HashRandom\s*\(\s*pixelPosition\s*\)\s*\+\s*temporalSample\s*\)"));
        Assert.That(shader, Does.Match(@"SampleHemisphereCosine\s*\(\s*xi\.x\s*,\s*xi\.y\s*,\s*normal\s*\)"));
        Assert.That(shader, Does.Match(@"BuildHemisphereDirection\s*\(\s*normal\s*,\s*input\.positionCS\.xy\s*,\s*i\s*\)"));
        Assert.That(shader, Does.Not.Match(@"sampleIndex\s*\*\s*float3"));
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
        Assert.That(shader, Does.Match(@"irradiance\s*\+=\s*hitRadiance\s*;"));
        Assert.That(shader, Does.Match(@"outgoingRadiance\s*=\s*irradiance\s*/\s*rayCount\s*;"));
        Assert.That(shader, Does.Not.Contain("SSGI_UNIFORM_HEMISPHERE_PDF"));
        Assert.That(shader, Does.Not.Match(@"float\s+ndotl\s*="));
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
        Assert.That(feature, Does.Contain("historyCount"));
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
    public void ViewChange_AccumulatesAndAdvancesSamplesWithoutDiscardingHistory()
    {
        string feature = ReadAsset("Scripts", "Runtime", "Features", "ScreenSpaceGlobalIlluminationFeature.cs");

        Assert.That(feature, Does.Match(@"if\s*\(\s*resetHistory\s*\)\s*ResetTemporalState\s*\(\s*state\s*\)"));
        Assert.That(feature, Does.Match(@"else\s+if\s*\(\s*viewChanged\s*&&\s*state\.historyValid\s*\)[\s\S]*state\.historyCount\s*=\s*Mathf\.Min\s*\([\s\S]*MotionHistoryFrameCap"));
        Assert.That(feature, Does.Match(@"MotionHistoryFrameCap\s*=\s*4"));
        Assert.That(feature, Does.Match(@"bool\s+accumulate\s*=\s*resetHistory\s*\|\|\s*isNewLogicalFrame\s*\|\|\s*viewChanged"));
        Assert.That(feature, Does.Match(@"if\s*\(\s*!resetHistory\s*&&\s*state\.historyValid\s*\)[\s\S]*sampleOffset\s*=\s*\(\s*state\.sampleOffset\s*\+\s*rayCount"));
    }

    [Test]
    public void TemporalAccumulation_UsesReprojectionValidationAndEnergyPreservingAverage()
    {
        const float previousSampleCount = 4.0f;
        const float maxHistoryFrames = 32.0f;
        const float temporalResponse = 0.05f;
        Func<float, float> currentFrameWeight = count => count < maxHistoryFrames
            ? 1.0f / (count + 1.0f)
            : temporalResponse;
        float runningWeight = currentFrameWeight(previousSampleCount);
        float cappedWeight = currentFrameWeight(maxHistoryFrames);
        Assert.That(runningWeight, Is.EqualTo(0.2f).Within(1e-6f));
        Assert.That(cappedWeight, Is.EqualTo(0.05f).Within(1e-6f));

        string shader = ReadAsset("Scripts", "Shaders", "SSGI", "ScreenSpaceGlobalIllumination.shader");
        string temporal = ReadAsset("Scripts", "Shaders", "GI", "Temporal.hlsl");
        Assert.That(shader, Does.Contain("Name \"Accumulate\""));
        Assert.That(temporal, Does.Contain("TEXTURE2D_X(_MotionVectorTexture)"));
        Assert.That(temporal, Does.Match(@"previousUV\s*=\s*uv\s*-\s*motion"));
        Assert.That(temporal, Does.Match(@"historyDepth\s*=\s*SAMPLE_TEXTURE2D_X\s*\([\s\S]*sampler_PointClamp\s*,\s*previousUV\s*\)\.a"));
        Assert.That(temporal, Does.Match(@"abs\s*\(\s*historyDepth\s*-\s*previousExpectedDepth\s*\)\s*>\s*depthThreshold"));
        Assert.That(temporal, Does.Match(@"runningAverageWeight\s*=\s*rcp\s*\(\s*_SSGIHistoryFrameCount\s*\+\s*1\.0\s*\)"));
        Assert.That(shader, Does.Not.Contain("ClampSSGIHistoryToCurrentNeighborhood"));
        Assert.That(shader, Does.Not.Contain("RGBToYCoCg"));
        Assert.That(shader, Does.Match(@"accumulated\s*=\s*lerp\s*\(\s*temporal\.irradianceDepth\.rgb\s*,\s*current\.rgb\s*,\s*currentFrameWeight\s*\)"));
        Assert.That(shader, Does.Match(@"return\s+float4\s*\(\s*accumulated\s*,\s*current\.a\s*\)"));
        Assert.That(shader, Does.Not.Contain("Name \"UpdateMoments\""));
        Assert.That(shader, Does.Not.Contain("Name \"UpdateNormalCount\""));
        Assert.That(shader, Does.Match(@"outgoingRadiance\s*\*\s*_SSGIIntensity\s*,\s*min\s*\(\s*viewPos\.z\s*,\s*65500\.0\s*\)"));
        Assert.That(Regex.Matches(shader, @"return\s+float4\s*\(\s*0\s*,\s*0\s*,\s*0\s*,\s*-1\.0\s*\)").Count, Is.EqualTo(2));
        Assert.That(shader, Does.Not.Contain("_SSGIMaxBlend"));

        string feature = ReadAsset("Scripts", "Runtime", "Features", "ScreenSpaceGlobalIlluminationFeature.cs");
        Assert.That(feature, Does.Match(@"HistoryWeightID\s*,\s*temporalResponse\s*\)"));
        Assert.That(feature, Does.Match(@"HistoryDepthThresholdID\s*,\s*temporalDepthThreshold\s*\)"));
        Assert.That(feature, Does.Match(@"HistoryFrameCountID\s*,[\s\S]*state\.historyCount"));
        Assert.That(feature, Does.Match(@"ScriptableRenderPassInput\.Motion"));
        Assert.That(feature, Does.Match(@"hash\s*=\s*hash\s*\*\s*31\s*\+\s*TemporalAlgorithmVersion"));
    }

    [Test]
    public void TemporalRenderSequence_PublishesBlurredIrradianceWithoutCompositingCameraColor()
    {
        string feature = ReadAsset("Scripts", "Runtime", "Features", "ScreenSpaceGlobalIlluminationFeature.cs");

        Assert.That(feature, Does.Match(@"sourceHandle\s*,\s*state\.radianceRT\s*\)"));
        Assert.That(feature, Does.Match(@"sourceHandle\s*,\s*state\.blurA\s*,\s*material\s*,\s*0\s*\)"));
        Assert.That(feature, Does.Match(@"state\.blurA\s*,\s*state\.HistoryWrite\s*,\s*material\s*,\s*3\s*\)"));
        Assert.That(feature, Does.Match(@"state\.HistoryWrite\s*,\s*state\.blurB\s*,\s*material\s*,\s*1\s*\)"));
        Assert.That(feature, Does.Match(@"state\.blurB\s*,\s*state\.irradianceRT\s*,\s*material\s*,\s*2\s*\)"));
        Assert.That(feature, Does.Match(@"material\s*,\s*3\s*\)[\s\S]*material\s*,\s*1\s*\)[\s\S]*material\s*,\s*2\s*\)"));
        Assert.That(feature, Does.Match(@"SetGlobalTexture\s*\(\s*RadianceTextureID\s*,\s*state\.radianceRT\s*\)"));
        Assert.That(feature, Does.Match(@"SetGlobalTexture\s*\(\s*IrradianceTextureID\s*,\s*state\.irradianceRT\s*\)"));
        Assert.That(feature, Does.Not.Contain("compositeRT"));
        Assert.That(feature, Does.Not.Match(@"BlitCameraTexture\s*\(\s*cmd\s*,\s*state\.irradianceRT\s*,\s*sourceHandle"));
        Assert.That(feature, Does.Not.Match(@"BlitCameraTexture\s*\([^;]*state\.HistoryRead\s*,\s*state\.HistoryRead"));
        Assert.That(feature, Does.Not.Contain("MomentsWrite"));
        Assert.That(feature, Does.Not.Contain("NormalCountWrite"));
    }

    [Test]
    public void PreviousIrradiance_IsBoundPerCameraBeforeOpaqueRendering()
    {
        string feature = ReadAsset("Scripts", "Runtime", "Features", "ScreenSpaceGlobalIlluminationFeature.cs");

        Assert.That(feature, Does.Contain("IrradianceHistorySetupPass"));
        Assert.That(feature, Does.Match(@"irradianceHistorySetupPass\.renderPassEvent\s*=\s*RenderPassEvent\.BeforeRenderingOpaques"));
        Assert.That(feature, Does.Match(@"EnqueuePass\s*\(\s*irradianceHistorySetupPass\s*\)"));
        Assert.That(feature, Does.Match(@"irradianceValid\s*=\s*state\.historyValid\s*&&\s*configurationMatches"));
        Assert.That(feature, Does.Not.Contain("poseMatches"));
        Assert.That(feature, Does.Match(@"SetGlobalFloat\s*\(\s*IrradianceValidID\s*,\s*irradianceValid\s*\?\s*1\.0f\s*:\s*0\.0f\s*\)"));
        Assert.That(feature, Does.Match(@"SetGlobalTexture\s*\(\s*IrradianceTextureID\s*,\s*state\.irradianceRT\s*\)"));
        Assert.That(feature, Does.Match(@"SetGlobalMatrix\s*\(\s*PreviousViewProjectionMatrixID\s*,\s*state\.previousViewProjectionMatrix\s*\)"));
        Assert.That(feature, Does.Match(@"SetGlobalMatrix\s*\(\s*PreviousWorldToCameraMatrixID\s*,\s*state\.previousWorldToCameraMatrix\s*\)"));
    }

    [Test]
    public void PerCameraTargets_AreAllReleasedAndStateDictionaryIsCleared()
    {
        string feature = ReadAsset("Scripts", "Runtime", "Features", "ScreenSpaceGlobalIlluminationFeature.cs");

        foreach (string handle in new[]
        {
            "radianceRT", "blurA", "blurB", "historyA", "historyB", "irradianceRT"
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
        FieldInfo temporalDepthThreshold = settingsType.GetField("temporalDepthThreshold");
        FieldInfo disocclusionFallback = settingsType.GetField("disocclusionFallback");

        Assert.That(sampleOffset, Is.Not.Null);
        Assert.That(originBias, Is.Not.Null);
        Assert.That(temporalResponse, Is.Not.Null);
        Assert.That(temporalDepthThreshold, Is.Not.Null);
        Assert.That(disocclusionFallback, Is.Not.Null);
        object settings = Activator.CreateInstance(settingsType);
        Assert.That(sampleOffset.GetValue(settings), Is.EqualTo(0));
        Assert.That(originBias.GetValue(settings), Is.EqualTo(0.01f));
        Assert.That(temporalResponse.GetValue(settings), Is.EqualTo(0.05f));
        Assert.That(temporalDepthThreshold.GetValue(settings), Is.EqualTo(0.1f));
        Assert.That(disocclusionFallback.GetValue(settings), Is.EqualTo(0.35f));
        Assert.That(sampleOffset.GetCustomAttribute<UnityEngine.RangeAttribute>(), Is.Not.Null);
        Assert.That(originBias.GetCustomAttribute<UnityEngine.RangeAttribute>(), Is.Not.Null);
        Assert.That(temporalResponse.GetCustomAttribute<UnityEngine.RangeAttribute>(), Is.Not.Null);
        Assert.That(temporalDepthThreshold.GetCustomAttribute<UnityEngine.MinAttribute>(), Is.Not.Null);
        Assert.That(disocclusionFallback.GetCustomAttribute<UnityEngine.RangeAttribute>(), Is.Not.Null);

        string feature = ReadAsset("Scripts", "Runtime", "Features", "ScreenSpaceGlobalIlluminationFeature.cs");
        Assert.That(feature, Does.Contain("_SSGISampleOffset"));
        Assert.That(feature, Does.Contain("_SSGIOriginBias"));
        Assert.That(feature, Does.Contain("_SSGIHistoryDepthThreshold"));
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
        Assert.That(feature, Does.Contain("Mathf.Clamp(settings.maxHistoryFrames, 2, 64)"));
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
        Assert.That(blur, Does.Match(@"exp\s*\(\s*-SSGI_FILTER_DEPTH_FALLOFF\s*\*\s*abs\s*\(\s*centerDepth\s*-\s*sampleDepth\s*\)\s*\)"));
        Assert.That(blur, Does.Match(@"saturate\s*\(\s*dot\s*\(\s*centerNormal\s*,\s*sampleNormal\s*\)\s*\)"));
        Assert.That(blur, Does.Not.Contain("_SSGIMomentsTexture"));
        Assert.That(blur, Does.Not.Contain("_SSGINormalCountTexture"));
        Assert.That(blur, Does.Match(@"SSGI_FILTER_OFFSETS\s*\[\s*i\s*\]\s*\*\s*texelSize\s*\*\s*filterRadius"));
        Assert.That(blur, Does.Match(@"weight\s*=\s*depthWeight\s*\*\s*normalWeight"));
        Assert.That(blur, Does.Not.Contain("luminanceWeight"));
        Assert.That(blur, Does.Match(@"max\s*\(\s*0\.001\s*,\s*weightSum\s*\)"));
        Assert.That(blur, Does.Match(@"return\s+float4\s*\(\s*filtered\s*,\s*center\.a\s*\)"));
        Assert.That(blur, Does.Match(@"sampleUV\.x\s*<\s*0\.0[\s\S]*sampleUV\.x\s*>\s*1\.0"));
        Assert.That(blur, Does.Match(@"sample\.a\s*<\s*0\.0"));
        Assert.That(blur, Does.Not.Contain("Gaussian"));
        Assert.That(blur, Does.Not.Contain("GI_BLUR_WEIGHT"));
        Assert.That(sharedBlur, Does.Contain("SampleGaussianBlurHorizontal"));
        Assert.That(sharedBlur, Does.Contain("SampleGaussianBlurVertical"));
        Assert.That(shader, Does.Contain("../GI/SSGIBlur.hlsl"));

        Assert.That(Regex.Matches(shader, "DeclareDepthTexture.hlsl").Count, Is.EqualTo(4));
        Assert.That(Regex.Matches(shader, "../GI/Commond.hlsl").Count, Is.EqualTo(3));
        Assert.That(shader, Does.Match(@"FragBlurH[\s\S]*SampleSSGIEdgeAwareFilter\s*\(\s*input\.texcoord\s*,\s*_BlurSpread\s*\)"));
        Assert.That(shader, Does.Match(@"FragBlurV[\s\S]*SampleSSGIEdgeAwareFilter\s*\(\s*input\.texcoord\s*,\s*_BlurSpread\s*\*\s*2\.0\s*\)"));
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
    public void MainLightDirect_ReprojectsWithPointDepthAndFillsDisocclusions()
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
        Assert.That(feature, Does.Match(@"cameraMoved\s*=\s*state\.hasPreviousPose"));
        Assert.That(feature, Does.Match(@"SetGlobalFloat\s*\(\s*ReprojectIrradianceID\s*,\s*cameraMoved\s*\?\s*1\.0f\s*:\s*0\.0f\s*\)"));
    }
}
