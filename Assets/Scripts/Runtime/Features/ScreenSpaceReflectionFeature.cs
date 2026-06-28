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
        [Tooltip("Distance of each ray march step")]
        [Range(0.01f, 1.0f)]
        public float stepSize = 0.1f;

        [Tooltip("Maximum number of ray march steps")]
        [Range(1, 1024)]
        public int maxSteps = 32;

        [Tooltip("Maximum ray travel distance")]
        [Range(0.1f, 1024.0f)]
        public float maxDistance = 10.0f;

        [Tooltip("Geometry thickness for hit detection")]
        [Range(0.001f, 1.0f)]
        public float thickness = 0.1f;
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
        private RTHandle tempTarget;

        private static readonly ProfilingSampler profilingSampler =
            new ProfilingSampler("Screen Space Reflection");

        private static readonly int UVToViewPosID = Shader.PropertyToID("_UVToViewPos");
        private static readonly int StepSizeID = Shader.PropertyToID("_SSRStepSize");
        private static readonly int MaxStepsID = Shader.PropertyToID("_SSRMaxSteps");
        private static readonly int MaxDistanceID = Shader.PropertyToID("_SSRMaxDistance");
        private static readonly int ThicknessID = Shader.PropertyToID("_SSRThickness");

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

            RenderingUtils.ReAllocateIfNeeded(ref tempTarget, desc, FilterMode.Point, TextureWrapMode.Clamp, name: "_SSRTempTarget");
        }

        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            if (material == null) return;

            CommandBuffer cmd = CommandBufferPool.Get("SSR Pass");

            using (new ProfilingScope(cmd, profilingSampler))
            {
                Camera cam = renderingData.cameraData.camera;
                Vector4 uvToView = GIUtility.ComputeUVToViewPos(cam);
                material.SetVector(UVToViewPosID, uvToView);

                material.SetFloat(StepSizeID, settings.stepSize);
                material.SetInt(MaxStepsID, settings.maxSteps);
                material.SetFloat(MaxDistanceID, settings.maxDistance);
                material.SetFloat(ThicknessID, settings.thickness);

                // cmd.Blit(sourceHandle, tempTarget, material, 0);
                // cmd.Blit(tempTarget, sourceHandle);
                Blitter.BlitCameraTexture(cmd, sourceHandle, tempTarget, material, 0);
                Blitter.BlitCameraTexture(cmd, tempTarget, sourceHandle);
            }

            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
        }

        public override void OnCameraCleanup(CommandBuffer cmd)
        {
        }

        public void Dispose()
        {
            tempTarget?.Release();
            tempTarget = null;
        }
    }
}
