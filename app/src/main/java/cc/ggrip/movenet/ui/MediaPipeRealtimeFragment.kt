// MediaPipeRealtimeFragment.kt
package cc.ggrip.movenet.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Rational
import android.view.*
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.widget.RadioGroup
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import cc.ggrip.movenet.R
import cc.ggrip.movenet.bench.ModelAssets
import cc.ggrip.movenet.bench.Tier
import cc.ggrip.movenet.smoothing.SmoothingMode
import cc.ggrip.movenet.tflite.MediaPipePoseProcessor
import cc.ggrip.movenet.util.FpsGovernor
import cc.ggrip.movenet.util.LatencyMeter
import cc.ggrip.movenet.analysis.SwingStateAnalyzer
import java.util.concurrent.Executors

class MediaPipeRealtimeFragment : Fragment() {

    companion object {
        private const val ARG_FPS = "target_fps"
        private const val ARG_TIER = "tier"

        fun newInstance(targetFps: Double, tier: Tier) = MediaPipeRealtimeFragment().apply {
            arguments = Bundle().apply {
                putDouble(ARG_FPS, targetFps)
                putString(ARG_TIER, tier.name)
            }
        }
    }

    private lateinit var previewView: PreviewView
    private lateinit var overlay: DotsOverlay
    private lateinit var processor: MediaPipePoseProcessor
    private lateinit var fpsGov: FpsGovernor
    private lateinit var latencyMeter: LatencyMeter
    private lateinit var swingAnalyzer: SwingStateAnalyzer
    private var smoothingModeGroup: RadioGroup? = null
    private var targetFps = 30.0
    private var chosenTier: Tier = Tier.LIGHT

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
        chosenTier = arguments?.getString(ARG_TIER)?.let { Tier.valueOf(it) } ?: Tier.LIGHT
        fpsGov = FpsGovernor(targetFps)
        latencyMeter = LatencyMeter()
        swingAnalyzer = SwingStateAnalyzer()
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
            it.setMirrorFlip(mirrorX = true, flipY = false)
            it.setEngineLabel("MediaPipe")
            it.setModelLabel(
                when (chosenTier) {
                    Tier.LIGHT -> "lite"
                    Tier.MID   -> "full"
                    Tier.HEAVY -> "heavy"
                }
            )
        }

        // Processor 생성
        val assetPath = ModelAssets.mpTaskPath(chosenTier)
        processor = MediaPipePoseProcessor(requireContext(), assetPath) { frame ->

            overlay.setAcceleratorLabel(processor.currentDelegate())

            frame?.let {
                val swingAnalysis = swingAnalyzer.analyzeSwingState(it)
                overlay.post {
                    overlay.update(it)
                    overlay.updateSwingState(swingAnalysis)
                }
            }
        }

        // 가속기 라벨 & 초기 엔진 라벨 동기화
        overlay.setAcceleratorLabel(processor.currentDelegate())
        overlay.setEngineLabel(
            when (SmoothingMode.RAW) {
                SmoothingMode.RAW -> "MediaPipe (Raw)"
                SmoothingMode.EMA -> "MediaPipe (EMA)"
                SmoothingMode.FLK -> "MediaPipe (FLK)"
            }
        )

        // 스무딩 모드 라디오 리스너 (overlay 생성 이후 등록)
        smoothingModeGroup = view.findViewById(R.id.smoothingModeGroup)
        smoothingModeGroup?.setOnCheckedChangeListener { _, checkedId ->
            val mode = when (checkedId) {
                R.id.modeRaw -> SmoothingMode.RAW
                R.id.modeEMA -> SmoothingMode.EMA
                R.id.modeFLK -> SmoothingMode.FLK
                else -> SmoothingMode.RAW
            }
            processor.smoothingMode = mode // 내부에서 resetSmoothing() 호출됨

            overlay.setEngineLabel(
                when (mode) {
                    SmoothingMode.RAW -> "MediaPipe (Raw)"
                    SmoothingMode.EMA -> "MediaPipe (EMA)"
                    SmoothingMode.FLK -> "MediaPipe (FLK)"
                }
            )
        }

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
                        // 프레임 스로틀링(필요하다면 FpsGovernor에서 거부)
                        if (!fpsGov.shouldAccept(tsNs)) { imageProxy.close(); return@setAnalyzer }

                        val rotDeg = imageProxy.imageInfo.rotationDegrees
                        val srcW = if (rotDeg % 180 == 0) imageProxy.width else imageProxy.height
                        val srcH = if (rotDeg % 180 == 0) imageProxy.height else imageProxy.width
                        overlay.setSourceSize(srcW, srcH)

                        processor.process(imageProxy)
                    }
                }

            val selector = CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                .build()

            // PreviewView 크기 준비 안됐으면 재시도
            if (previewView.width == 0 || previewView.height == 0) {
                previewView.post { startCamera() }
                return@addListener
            }
            val vp = ViewPort.Builder(Rational(previewView.width, previewView.height), rotation)
                .setScaleType(ViewPort.FILL_CENTER)
                .build()

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
        try { analysis?.clearAnalyzer() } catch (_: Exception) {}
        try { cameraProvider?.unbindAll() } catch (_: Exception) {}
        try { processor.close() } catch (_: Exception) {}
    }
}
