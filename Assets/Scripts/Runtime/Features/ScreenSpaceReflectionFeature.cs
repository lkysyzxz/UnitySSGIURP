using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Experimental.Rendering;

public class ScreenSpaceReflectionFeature : ScriptableRendererFeature
{
    [System.Serializable]
    public class SSRSettings
    {
        [Header("Ray March")]
        [Tooltip("Maximum number of ray march steps")]
        [Range(1, 1024)]
        public int maxSteps = 32;

        [Tooltip("Maximum ray travel distance")]
        [Range(0.1f, 1024.0f)]
        public float maxDistance = 10.0f;

        [Tooltip("Geometry thickness for hit detection")]
        [Range(0.001f, 1.0f)]
        public float thickness = 0.1f;

        [Header("Blur")]
        [Tooltip("Gaussian blur spread in pixels (0 = no blur)")]
        [Range(0f, 8f)]
        public float blurSpread = 1.5f;

        [Header("Jitter")]
        [Tooltip("Enable jitter dither to break regular sampling and reduce banding")]
        public bool jitterDither = true;
    }

    [SerializeField] private SSRSettings settings = new SSRSettings();
    [SerializeField] private Shader ssrShader;

    private Material ssrMaterial;
    private ScreenSpaceReflectionPass ssrPass;

    public override void Create()
    {
        ssrPass = new ScreenSpaceReflectionPass(settings);
        ssrPass.renderPassEvent = RenderPassEvent.BeforeRenderingPostProcessing;
    }

    public override void SetupRenderPasses(ScriptableRenderer renderer, in RenderingData renderingData)
    {
        if (ssrMaterial == null) return;
        ssrPass.Setup(ssrMaterial, settings, renderer.cameraColorTargetHandle);
    }

    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        if (ssrShader == null) return;
        if (ssrMaterial == null)
            ssrMaterial = CoreUtils.CreateEngineMaterial(ssrShader);
        if (ssrMaterial == null) return;
        renderer.EnqueuePass(ssrPass);
    }

    protected override void Dispose(bool disposing)
    {
        ssrPass?.Dispose();
        CoreUtils.Destroy(ssrMaterial);
        ssrMaterial = null;
    }

    // --- Inner Pass Class ---
    internal class ScreenSpaceReflectionPass : ScriptableRenderPass
    {
        private Material material;
        private SSRSettings settings;
        private RTHandle sourceHandle;
        private RTHandle rtA;   // SSR 结果 / BlurV 结果
        private RTHandle rtB;   // BlurH 结果 / Composite 结果

        private static readonly ProfilingSampler profilingSampler =
            new ProfilingSampler("Screen Space Reflection");

        private static readonly int UVToViewPosID     = Shader.PropertyToID("_UVToViewPos");
        private static readonly int MaxStepsID        = Shader.PropertyToID("_SSRMaxSteps");
        private static readonly int MaxDistanceID     = Shader.PropertyToID("_SSRMaxDistance");
        private static readonly int ThicknessID       = Shader.PropertyToID("_SSRThickness");
        private static readonly int BlurSpreadID      = Shader.PropertyToID("_BlurSpread");
        private static readonly int OriginalTextureID = Shader.PropertyToID("_OriginalTexture");
        private const string JitterKeyword = "_JITTER_ON";

        public ScreenSpaceReflectionPass(SSRSettings settings)
        {
            this.settings = settings;
        }

        public void Setup(Material mat, SSRSettings ssrSettings, RTHandle source)
        {
            material = mat;
            settings = ssrSettings;
            sourceHandle = source;
            ConfigureInput(ScriptableRenderPassInput.Depth | ScriptableRenderPassInput.Color);
        }

        public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
        {
            var desc = renderingData.cameraData.cameraTargetDescriptor;
            desc.depthBufferBits = 0;
            desc.msaaSamples = 1;
            desc.graphicsFormat = GraphicsFormat.R16G16B16A16_SFloat;

            RenderingUtils.ReAllocateIfNeeded(ref rtA, desc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: "_SSR_RT_A");
            RenderingUtils.ReAllocateIfNeeded(ref rtB, desc, FilterMode.Bilinear, TextureWrapMode.Clamp, name: "_SSR_RT_B");
        }

        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            if (material == null) return;

            CommandBuffer cmd = CommandBufferPool.Get("SSR Pass");

            using (new ProfilingScope(cmd, profilingSampler))
            {
                Camera cam = renderingData.cameraData.camera;
                material.SetVector(UVToViewPosID, GIUtility.ComputeUVToViewPos(cam));
                material.SetInt(MaxStepsID, settings.maxSteps);
                material.SetFloat(MaxDistanceID, settings.maxDistance);
                material.SetFloat(ThicknessID, settings.thickness);
                material.SetFloat(BlurSpreadID, settings.blurSpread);

                // Jitter dither 开关
                if (settings.jitterDither)
                    material.EnableKeyword(JitterKeyword);
                else
                    material.DisableKeyword(JitterKeyword);

                // Pipeline（2 个临时 RT 乒乓，避免读写同一目标）:
                //   source ─[Pass0 SSR]─→ rtA   (反射色 + 混合系数 alpha)
                //   rtA    ─[Pass1 BlurH]─→ rtB
                //   rtB    ─[Pass2 BlurV]─→ rtA   (模糊后反射 + alpha)
                //   把原图绑给 _OriginalTexture
                //   rtA    ─[Pass3 Composite: lerp(原图, 反射, alpha)]─→ rtB
                //   rtB    ─[拷贝]─→ source

                Blitter.BlitCameraTexture(cmd, sourceHandle, rtA, material, 0);  // SSR
                Blitter.BlitCameraTexture(cmd, rtA, rtB, material, 1);           // BlurH
                Blitter.BlitCameraTexture(cmd, rtB, rtA, material, 2);           // BlurV

                cmd.SetGlobalTexture(OriginalTextureID, sourceHandle);            // 原图

                Blitter.BlitCameraTexture(cmd, rtA, rtB, material, 3);           // Composite
                Blitter.BlitCameraTexture(cmd, rtB, sourceHandle);               // 拷回 source
            }

            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
        }

        public override void OnCameraCleanup(CommandBuffer cmd)
        {
        }

        public void Dispose()
        {
            rtA?.Release();
            rtB?.Release();
            rtA = null;
            rtB = null;
        }
    }
}
