// RealtimeDotsFragment.kt  (핵심 변경: 모델 tier/경로 인자, HUD 타이틀/모델 라벨 설정)
package cc.ggrip.movenet.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Rational
import android.view.*
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.ViewPort
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import cc.ggrip.movenet.R
import cc.ggrip.movenet.bench.ModelAssets
import cc.ggrip.movenet.bench.Tier
import cc.ggrip.movenet.tflite.MoveNetProcessor
import cc.ggrip.movenet.util.FpsGovernor
import cc.ggrip.movenet.util.LatencyMeter
import java.util.concurrent.Executors

class RealtimeDotsFragment : Fragment() {

    companion object {
        private const val ARG_FPS = "target_fps"
        private const val ARG_TIER = "tier"

        fun newInstance(targetFps: Double, tier: Tier) = RealtimeDotsFragment().apply {
            arguments = Bundle().apply {
                putDouble(ARG_FPS, targetFps)
                putString(ARG_TIER, tier.name)
            }
        }
    }

    private lateinit var previewView: PreviewView
    private lateinit var overlay: DotsOverlay
    private lateinit var processor: MoveNetProcessor
    private lateinit var fpsGov: FpsGovernor
    private lateinit var latencyMeter: LatencyMeter
    private var targetFps = 30.0
    private var chosenTier: Tier = Tier.MID

    private var cameraProvider: ProcessCameraProvider? = null
    private var analysis: ImageAnalysis? = null
    private val analyzerExecutor = Executors.newSingleThreadExecutor()

    private val requestPerm =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { res ->
            if (res.values.all { it }) startCamera()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        targetFps = arguments?.getDouble(ARG_FPS, 30.0) ?: 30.0
        chosenTier = arguments?.getString(ARG_TIER)?.let { Tier.valueOf(it) } ?: Tier.MID
        fpsGov = FpsGovernor(targetFps)
        latencyMeter = LatencyMeter()
    }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, saved: Bundle?): View {
        return inflater.inflate(R.layout.fragment_realtime_dots, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        previewView = view.findViewById(R.id.previewView)
        previewView.implementationMode = PreviewView.ImplementationMode.PERFORMANCE
        previewView.scaleType = PreviewView.ScaleType.FILL_CENTER

        overlay = DotsOverlay(
            context = requireContext(),
            targetFps = targetFps,
            meter = latencyMeter
        ).also {
            (view as ViewGroup).addView(it, ViewGroup.LayoutParams(MATCH_PARENT, MATCH_PARENT))
            it.bringToFront()
            it.setMirrorFlip(mirrorX = true, flipY = false) // 전면 카메라 기준
            it.setEngineLabel("MoveNet")
            it.setModelLabel(
                when (chosenTier) {
                    Tier.LIGHT -> "lightning"
                    Tier.MID, Tier.HEAVY -> "thunder"
                }
            )
        }

        val assetPath = ModelAssets.movenetPath(chosenTier)
        processor = MoveNetProcessor(requireContext(), { frame ->
            frame?.let { overlay.post { overlay.update(it) } }
        }, assetPath)

        overlay.setAcceleratorLabel(processor.currentDelegate())
        ensurePerm()
    }

    private fun ensurePerm() {
        val need = arrayOf(Manifest.permission.CAMERA)
        if (need.any { ContextCompat.checkSelfPermission(requireContext(), it) != PackageManager.PERMISSION_GRANTED }) {
            requestPerm.launch(need)
        } else startCamera()
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(requireContext())
        providerFuture.addListener({
            cameraProvider = providerFuture.get()

            val rotation = requireView().display?.rotation ?: Surface.ROTATION_0

            val preview = Preview.Builder()
                .setTargetRotation(rotation)
                .build().also { it.setSurfaceProvider(previewView.surfaceProvider) }

            analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .setTargetRotation(rotation)
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build().also { ia ->
                    ia.setAnalyzer(analyzerExecutor) { imageProxy ->
                        val tsNs = imageProxy.imageInfo.timestamp
                        if (!fpsGov.shouldAccept(tsNs)) { imageProxy.close(); return@setAnalyzer }

                        val rot = imageProxy.imageInfo.rotationDegrees
                        val srcW = if (rot % 180 == 0) imageProxy.width  else imageProxy.height
                        val srcH = if (rot % 180 == 0) imageProxy.height else imageProxy.width
                        overlay.setSourceSize(srcW, srcH)

                        processor.process(imageProxy)
                    }
                }

            val selector = CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_FRONT).build()

            if (previewView.width == 0 || previewView.height == 0) {
                previewView.post { startCamera() }; return@addListener
            }
            val vp = ViewPort.Builder(Rational(previewView.width, previewView.height), rotation)
                .setScaleType(ViewPort.FILL_CENTER).build()

            val group = UseCaseGroup.Builder()
                .setViewPort(vp)
                .addUseCase(preview)
                .addUseCase(analysis!!)
                .build()

            cameraProvider?.unbindAll()
            cameraProvider?.bindToLifecycle(viewLifecycleOwner, selector, group)

        }, ContextCompat.getMainExecutor(requireContext()))
    }

    override fun onDestroyView() {
        super.onDestroyView()
        try { processor.close() } catch (_: Exception) {}
    }
}
