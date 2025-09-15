// RealtimeDotsFragment.kt
package cc.ggrip.movenet.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.*
import android.util.Log
import android.view.*
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import cc.ggrip.movenet.R
import cc.ggrip.movenet.pose.PoseFrame
import cc.ggrip.movenet.tflite.MoveNetProcessor
import cc.ggrip.movenet.util.FpsGovernor
import cc.ggrip.movenet.util.LatencyMeter
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class RealtimeDotsFragment : Fragment() {

    companion object {
        private const val ARG_FPS = "target_fps"
        fun newInstance(targetFps: Double) = RealtimeDotsFragment().apply {
            arguments = Bundle().apply { putDouble(ARG_FPS, targetFps) }
        }
        private const val TAG = "RealtimeDotsFragment"
    }

    private lateinit var previewView: PreviewView
    private lateinit var overlay: DotsOverlay
    private lateinit var processor: MoveNetProcessor
    private lateinit var fpsGov: FpsGovernor
    private lateinit var latencyMeter: LatencyMeter
    private var targetFps = 30.0

    private var cameraProvider: ProcessCameraProvider? = null
    private var analysisExecutor: ExecutorService? = null

    private val requestPerm =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { res ->
            if (res.values.all { it }) startCamera() else Log.e(TAG, "Camera permission denied")
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

        overlay = DotsOverlay(requireContext(), targetFps, latencyMeter).apply {
            setMirror(true) // 전면 카메라이므로 좌우 반전
        }
        (view as ViewGroup).addView(overlay, ViewGroup.LayoutParams(MATCH_PARENT, MATCH_PARENT))
        overlay.bringToFront()

        processor = MoveNetProcessor(requireContext()) { frame ->
            frame?.let { overlay.post { overlay.update(it) } }
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
        val ctx = requireContext()
        val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
        cameraProviderFuture.addListener({
            val provider = cameraProviderFuture.get()
            cameraProvider = provider

            // 전면 카메라
            val selector = CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                .build()

            // 프리뷰
            val preview = Preview.Builder()
                .setTargetRotation(requireActivity().windowManager.defaultDisplay?.rotation
                    ?: Surface.ROTATION_0)
                .build().also { it.setSurfaceProvider(previewView.surfaceProvider) }

            // 이미지 분석
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .setTargetRotation(requireActivity().windowManager.defaultDisplay?.rotation
                    ?: Surface.ROTATION_0)
                .build()

            analysisExecutor = Executors.newSingleThreadExecutor()
            analysis.setAnalyzer(analysisExecutor!!) { imageProxy ->
                val tsNs = imageProxy.imageInfo.timestamp
                if (!fpsGov.shouldAccept(tsNs)) {
                    imageProxy.close(); return@setAnalyzer
                }
                processor.process(imageProxy) // 내부에서 imageProxy.close()
            }

            try {
                provider.unbindAll()
                provider.bindToLifecycle(viewLifecycleOwner, selector, preview, analysis)
            } catch (t: Throwable) {
                Log.e(TAG, "bind failed", t)
            }
        }, ContextCompat.getMainExecutor(ctx))
    }

    override fun onPause() {
        super.onPause()
        // 필요 시 일시정지 처리
    }

    override fun onDestroyView() {
        try { cameraProvider?.unbindAll() } catch (_: Exception) {}
        try { analysisExecutor?.shutdownNow() } catch (_: Exception) {}
        try { processor.close() } catch (_: Exception) {}
        super.onDestroyView()
    }

    // ====== 오버레이 ======
    private class DotsOverlay(
        context: android.content.Context,
        private val targetFps: Double,
        private val meter: LatencyMeter
    ) : View(context) {
        @Volatile private var frame: PoseFrame? = null
        @Volatile private var mirror: Boolean = false

        fun setMirror(m: Boolean) { mirror = m }

        private val dotPaint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
            color = 0xFFFFCC00.toInt(); style = android.graphics.Paint.Style.FILL
        }
        private val hudPaint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
            color = 0xFFFFFFFF.toInt(); textSize = 36f
            setShadowLayer(4f, 1f, 1f, 0x80000000.toInt())
        }
        private val boxPaint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
            color = 0x66000000; style = android.graphics.Paint.Style.FILL
        }

        fun update(f: PoseFrame) { frame = f; postInvalidateOnAnimation() }

        override fun onDraw(canvas: android.graphics.Canvas) {
            super.onDraw(canvas)
            val f = frame ?: return

            val p = f.screen2d
            val W = width.toFloat(); val H = height.toFloat()
            // MoveNet 17포인트
            for (i in 0 until 17) {
                var x = p[i * 2]
                val y = p[i * 2 + 1]
                if (mirror) x = 1f - x
                canvas.drawCircle(x * W, y * H, 10f, dotPaint)
            }

            // 지연 통계
            val nowMs = SystemClock.elapsedRealtime()
            if (f.srcTsMs > 0) {
                val e2e = nowMs - f.srcTsMs
                val algo = if (f.algoDoneTsMs > 0) f.algoDoneTsMs - f.srcTsMs else -1L
                meter.push(algo, e2e)
            }
            val stats = meter.snapshot()

            val frameInterval = 1000.0 / targetFps
            val eAvgF = if (!stats.e2eAvg.isNaN()) stats.e2eAvg / frameInterval else Double.NaN
            val eP95F = if (!stats.e2eP95.isNaN()) stats.e2eP95 / frameInterval else Double.NaN

            fun fmtMs(d: Double) = if (d.isNaN()) "-" else "%.1f".format(d)
            fun fmtFr(d: Double) = if (d.isNaN()) "-" else "%.2f".format(d)

            val lines = listOf(
                "MoveNet • 목표 FPS ${"%.0f".format(targetFps)}",
                "알고리즘 지연 평균/95퍼센타일: ${fmtMs(stats.algoAvg)} / ${fmtMs(stats.algoP95)} ms",
                "종단 지연(E2E) 평균/95퍼센타일: ${fmtMs(stats.e2eAvg)} / ${fmtMs(stats.e2eP95)} ms",
                "프레임 지연(환산): ${fmtFr(eAvgF)} 프레임(평균) | ${fmtFr(eP95F)} 프레임(95퍼센타일)"
            )
            val pad = 12f
            val boxW = lines.maxOf { hudPaint.measureText(it) } + pad * 2
            val boxH = hudPaint.textSize * lines.size + pad * 2
            canvas.drawRoundRect(16f, 16f, 16f + boxW, 16f + boxH, 18f, 18f, boxPaint)
            var y0 = 16f + pad + hudPaint.textSize
            for (ln in lines) { canvas.drawText(ln, 16f + pad, y0, hudPaint); y0 += hudPaint.textSize }
        }
    }
}