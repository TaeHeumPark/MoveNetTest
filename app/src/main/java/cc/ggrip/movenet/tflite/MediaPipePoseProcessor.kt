// MediaPipePoseProcessor.kt
package cc.ggrip.movenet.mediapipe

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy
import cc.ggrip.movenet.pose.PoseFrame
import cc.ggrip.movenet.tflite.YuvToRgb
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker.PoseLandmarkerOptions

private const val TAG_MP = "MPPose"

class MediaPipePoseProcessor(
    context: Context,
    private val assetPath: String,
    private val onResult: (PoseFrame?) -> Unit
) {
    private val context: Context = context.applicationContext
    private val yuv = YuvToRgb(context)
    private var landmarker: PoseLandmarker? = null
    @Volatile private var delegateLabel: String = "CPU"
    fun currentDelegate(): String = delegateLabel
    private val tsQueue = java.util.concurrent.ConcurrentLinkedQueue<Long>()

    // 백프레셔 상태
    @Volatile private var inFlight = false
    @Volatile private var lastSrcTsMs: Long = -1L
    @Volatile private var inFlightSince: Long = 0L

    // 연속 에러 카운트 (GPU→CPU 폴백 트리거)
    @Volatile private var errCount = 0

    init {
        // heavy는 CPU 우선 (원하면 true로 바꿔 GPU 먼저 시도)
        val isHeavy = false
        landmarker = if (isHeavy) tryCreate(Delegate.CPU) else tryCreate(Delegate.GPU) ?: tryCreate(Delegate.CPU)
    }

    private fun recreateWithCpu() {
        try { landmarker?.close() } catch (_: Exception) {}
        landmarker = tryCreate(Delegate.CPU)
    }

    private fun tryCreate(delegate: Delegate): PoseLandmarker? {
        return try {
            val base = BaseOptions.builder()
                .setModelAssetPath(assetPath)
                .setDelegate(delegate)
                .build()

            val opts = PoseLandmarkerOptions.builder()
                .setBaseOptions(base)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setNumPoses(1)
                .setMinPoseDetectionConfidence(0.3f)
                .setMinPosePresenceConfidence(0.3f)
                .setMinTrackingConfidence(0.3f)
                // ★ 2-인자 리스너 (result, inputImage)
                .setResultListener { result, _: MPImage ->
                    val srcTs = lastSrcTsMs
                    val algoDone = android.os.SystemClock.elapsedRealtime()

                    val lmList = result.landmarks().firstOrNull()
                    if (!lmList.isNullOrEmpty() && srcTs > 0) {
                        val n = lmList.size
                        val screen2d = FloatArray(n * 2)
                        for (i in 0 until n) {
                            val lm = lmList[i]
                            screen2d[i*2]     = lm.x()
                            screen2d[i*2 + 1] = lm.y()
                        }
                        onResult(
                            PoseFrame(
                                tMillis = algoDone,
                                world = floatArrayOf(),
                                screen2d = screen2d,
                                visibility = null,
                                srcTsMs = srcTs,
                                algoDoneTsMs = algoDone
                            )
                        )
                    } else {
                        onResult(null)
                    }

                    // 성공 → 상태 초기화
                    errCount = 0
                    lastSrcTsMs = -1L
                    inFlight = false
                }
                // ✅ 에러 리스너: 막히면 풀고, GPU면 CPU로 폴백
                .setErrorListener { e ->
                    errCount++
                    lastSrcTsMs = -1L
                    inFlight = false
                    if (delegate == Delegate.GPU && errCount >= 1) {
                        recreateWithCpu()
                        delegateLabel = "CPU"
                    }
                }
                .build()

            return try {
                val inst = PoseLandmarker.createFromOptions(context, opts)
                delegateLabel = if (delegate == Delegate.GPU) "GPU" else "CPU"
                errCount = 0
                inst
            } catch (_: Throwable) {
                null
            }
        } catch (t: Throwable) {
            Log.w(TAG_MP, "create(delegate=$delegate) failed: ${t.message}")
            null
        }
    }

    @OptIn(ExperimentalGetImage::class)
    fun process(imageProxy: ImageProxy) {
        val tsMs = imageProxy.imageInfo.timestamp / 1_000_000L
        try {
            // ✅ watchdog: 콜백이 너무 오래 안 오면 해제
            if (inFlight && android.os.SystemClock.elapsedRealtime() - inFlightSince > 1200) {
                inFlight = false
                lastSrcTsMs = -1L
            }
            if (inFlight) { imageProxy.close(); return }

            // YUV→RGB, 회전, 중앙 정사각형 크롭
            val srcBmp = Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)
            yuv.yuvToRgb(imageProxy.image!!, srcBmp)
            val rot = imageProxy.imageInfo.rotationDegrees
            val mat = Matrix().apply { postRotate(rot.toFloat()) }
            val rotated = Bitmap.createBitmap(srcBmp, 0, 0, srcBmp.width, srcBmp.height, mat, false)
            val size = minOf(rotated.width, rotated.height)
            val left = (rotated.width - size) / 2
            val top  = (rotated.height - size) / 2
            val square = Bitmap.createBitmap(rotated, left, top, size, size)
            val mpImg = BitmapImageBuilder(square).build()

            // 호출 직전 상태 세팅
            lastSrcTsMs = tsMs
            inFlight = true
            inFlightSince = android.os.SystemClock.elapsedRealtime()

            landmarker?.detectAsync(mpImg, tsMs)
        } catch (_: Throwable) {
            lastSrcTsMs = -1L
            inFlight = false
            onResult(null)
        } finally {
            imageProxy.close()
        }
    }

    fun close() { try { landmarker?.close() } catch (_: Exception) {} }
}
