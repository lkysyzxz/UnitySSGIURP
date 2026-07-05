using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Experimental.Rendering;

public class HierarchicalZBufferFeature : ScriptableRendererFeature
{
    [System.Serializable]
    public class HiZSettings
    {
        [Tooltip("HiZ 层数（0 = 自动，最大 8）")]
        [Range(0, 8)]
        public int mipCount = 0;
    }

    [SerializeField] private HiZSettings settings = new HiZSettings();
    [SerializeField] private Shader hiZShader;

    private Material hiZMaterial;
    private HiZPass hiZPass;

    public override void Create()
    {
        hiZPass = new HiZPass(settings);
        hiZPass.renderPassEvent = RenderPassEvent.AfterRenderingOpaques;
    }

    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        if (hiZShader == null) return;
        if (hiZMaterial == null)
            hiZMaterial = CoreUtils.CreateEngineMaterial(hiZShader);
        if (hiZMaterial == null) return;
        hiZPass.Setup(hiZMaterial, settings);
        renderer.EnqueuePass(hiZPass);
    }

    protected override void Dispose(bool disposing)
    {
        hiZPass?.Dispose();
        CoreUtils.Destroy(hiZMaterial);
        hiZMaterial = null;
    }

    internal class HiZPass : ScriptableRenderPass
    {
        private const int MaxMipCount = 8;
        private Material material;
        private HiZSettings settings;
        private int runtimeMipCount;
        private RTHandle[] hiZTextures = new RTHandle[MaxMipCount];

        private static readonly ProfilingSampler profilingSampler =
            new ProfilingSampler("Hierarchical Z-Buffer");

        private static readonly int HiZMipCountID = Shader.PropertyToID("_HiZMipCount");
        private static readonly int HiZSrcSizeID = Shader.PropertyToID("_HiZSrcSize");
        private static readonly int HiZMaxMipID = Shader.PropertyToID("_HiZMaxMip");

        public HiZPass(HiZSettings settings) { this.settings = settings; }

        public void Setup(Material mat, HiZSettings s)
        {
            material = mat;
            settings = s;
            ConfigureInput(ScriptableRenderPassInput.Depth);
        }

        public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
        {
            var camDesc = renderingData.cameraData.cameraTargetDescriptor;
            int w = camDesc.width;
            int h = camDesc.height;

            int maxMip = Mathf.Max(1, (int)Mathf.Floor(Mathf.Log(Mathf.Max(w, h), 2f)) + 1);
            runtimeMipCount = settings.mipCount > 0
                ? Mathf.Min(settings.mipCount, maxMip)
                : maxMip;
            runtimeMipCount = Mathf.Min(runtimeMipCount, MaxMipCount);

            // 每层独立 RT（尺寸递减），不用 mip chain（URP 14 的 mip chain 创建不稳定）
            for (int i = 0; i < runtimeMipCount; i++)
            {
                int mipW = Mathf.Max(w >> i, 1);
                int mipH = Mathf.Max(h >> i, 1);
                var desc = new RenderTextureDescriptor(mipW, mipH, GraphicsFormat.R32_SFloat, 0);
                desc.msaaSamples = 1;
                RenderingUtils.ReAllocateIfNeeded(ref hiZTextures[i], desc,
                    FilterMode.Point, TextureWrapMode.Clamp, name: "_HiZTexture_" + i);
            }
        }

        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            if (material == null || runtimeMipCount <= 0) return;

            CommandBuffer cmd = CommandBufferPool.Get("HiZ");

            using (new ProfilingScope(cmd, profilingSampler))
            {
                var colorTarget = renderingData.cameraData.renderer.cameraColorTargetHandle;

                // 1. CopyDepth → hiZTextures[0]
                Blitter.BlitCameraTexture(cmd, colorTarget, hiZTextures[0], material, 0);

                // 2. 降采样链：hiZTextures[i-1] → hiZTextures[i]
                for (int i = 1; i < runtimeMipCount; i++)
                {
                    // 传源（上一层）尺寸给 shader（Blitter 不设 _BlitTextureSize）
                    cmd.SetGlobalVector(HiZSrcSizeID, new Vector4(
                        hiZTextures[i - 1].referenceSize.x,
                        hiZTextures[i - 1].referenceSize.y, 0, 0));
                    Blitter.BlitCameraTexture(cmd, hiZTextures[i - 1], hiZTextures[i], material, 1);
                }

                // 3. 各层暴露为全局纹理 _HiZTexture_0..N
                for (int i = 0; i < runtimeMipCount; i++)
                {
                    cmd.SetGlobalTexture("_HiZTexture_" + i, hiZTextures[i]);
                }
                cmd.SetGlobalInt(HiZMipCountID, runtimeMipCount);
                cmd.SetGlobalFloat(HiZMaxMipID, runtimeMipCount - 1);
            }

            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
        }

        public override void OnCameraCleanup(CommandBuffer cmd) { }

        public void Dispose()
        {
            for (int i = 0; i < MaxMipCount; i++)
            {
                hiZTextures[i]?.Release();
                hiZTextures[i] = null;
            }
        }
    }
}
