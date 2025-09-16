// RealtimeDotsFragment.kt
package cc.ggrip.movenet.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Rational
import android.util.Size
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
import cc.ggrip.movenet.tflite.MoveNetProcessor
import cc.ggrip.movenet.util.FpsGovernor
import cc.ggrip.movenet.util.LatencyMeter
import java.util.concurrent.Executors

class RealtimeDotsFragment : Fragment() {

    companion object {
        private const val ARG_FPS = "target_fps"
        fun newInstance(targetFps: Double) = RealtimeDotsFragment().apply {
            arguments = Bundle().apply { putDouble(ARG_FPS, targetFps) }
        }
    }

    private lateinit var previewView: PreviewView
    private lateinit var overlay: DotsOverlay
    private lateinit var processor: MoveNetProcessor
    private lateinit var fpsGov: FpsGovernor
    private lateinit var latencyMeter: LatencyMeter
    private var targetFps = 30.0

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
            // 전면 카메라 기준: 좌우 미러 + 상하 플립(머리가 화면 위쪽)
            it.setMirrorFlip(mirrorX = true, flipY = false)
        }

        processor = MoveNetProcessor(requireContext()) { frame ->
            frame?.let { overlay.post { overlay.update(it) } }
        }

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
                .setTargetRotation(rotation) // ★ rotation 통일
                .build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .setTargetRotation(rotation)                 // ★ rotation 통일
                .setTargetAspectRatio(AspectRatio.RATIO_4_3) // ★ 해상도 강제 대신 AR만 통일
                .build().also { ia ->
                    ia.setAnalyzer(analyzerExecutor) { imageProxy ->
                        val tsNs = imageProxy.imageInfo.timestamp
                        if (!fpsGov.shouldAccept(tsNs)) { imageProxy.close(); return@setAnalyzer }

                        val rot = imageProxy.imageInfo.rotationDegrees
                        val srcW = if (rot % 180 == 0) imageProxy.width  else imageProxy.height
                        val srcH = if (rot % 180 == 0) imageProxy.height else imageProxy.width
                        overlay.setSourceSize(srcW, srcH)

                        processor.process(imageProxy) // 내부에서 close()
                    }
                }

            val selector = CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                .build()

            // ★★★ Preview와 Analysis에 동일한 ViewPort를 적용
            //     (PreviewView의 가시 영역과 동일한 FILL_CENTER 스케일/크롭을 카메라 파이프라인이 공유)
            if (previewView.width == 0 || previewView.height == 0) {
                // 레이아웃 직후에 다시 시도
                previewView.post { startCamera() }
                return@addListener
            }
            val vp = ViewPort.Builder(
                Rational(previewView.width, previewView.height),
                rotation
            ).setScaleType(ViewPort.FILL_CENTER).build()

            val group = UseCaseGroup.Builder()
                .setViewPort(vp)
                .addUseCase(preview)
                .addUseCase(analysis!!)
                .build()

            cameraProvider?.unbindAll()
            cameraProvider?.bindToLifecycle(viewLifecycleOwner, selector, group)

        }, ContextCompat.getMainExecutor(requireContext()))
    }
}
