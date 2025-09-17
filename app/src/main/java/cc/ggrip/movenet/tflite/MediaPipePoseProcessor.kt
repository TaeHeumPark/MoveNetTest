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

    // 백프레셔 상태
    @Volatile private var inFlight = false
    @Volatile private var lastFrameReceivedTsMs: Long = -1L  // 앱이 프레임을 받은 시각(boottime ms)
    @Volatile private var inFlightSince: Long = 0L           // inFlight가 시작된 시각
    @Volatile private var lastAlgoStartTsMs: Long = -1L      // 추론 시작 시각(boottime ms)

    // 연속 에러 카운트 (GPU→CPU 폴백 트리거)
    @Volatile private var errCount = 0

    init {
        // 무거운 모델에서 CPU를 우선 사용하고 싶으면 isHeavy = true 로 변경
        val isHeavy = false
        landmarker = if (isHeavy) {
            tryCreate(Delegate.CPU)
        } else {
            tryCreate(Delegate.GPU) ?: tryCreate(Delegate.CPU)
        }
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
                // ★ 2-인자 리스너(result, inputImage)
                .setResultListener { result, _: MPImage ->
                    val frameTs = lastFrameReceivedTsMs
                    val algoStartTs = lastAlgoStartTsMs
                    val algoDone = SystemClock.elapsedRealtime()

                    val lmList = result.landmarks().firstOrNull()
                    if (!lmList.isNullOrEmpty() && frameTs > 0 && algoStartTs > 0) {
                        val n = lmList.size
                        val screen2d = FloatArray(n * 2)
                        for (i in 0 until n) {
                            val lm = lmList[i]
                            screen2d[i * 2]     = lm.x()
                            screen2d[i * 2 + 1] = lm.y()
                        }
                        onResult(
                            PoseFrame(
                                tMillis = algoDone,
                                world = floatArrayOf(),
                                screen2d = screen2d,
                                visibility = null,
                                // E2E 시작: 앱이 프레임을 받은 시각(boottime ms)
                                frameReceivedTsMs = frameTs,
                                // 알고리즘 지연: 추론 시작/종료(boottime ms)
                                algoStartTsMs = algoStartTs,
                                algoDoneTsMs = algoDone
                            )
                        )
                    } else {
                        onResult(null)
                    }

                    // 백프레셔 상태 초기화
                    errCount = 0
                    lastFrameReceivedTsMs = -1L
                    lastAlgoStartTsMs = -1L
                    inFlight = false
                }
                // 에러 리스너: 막힘 해제, GPU 사용 중이면 CPU로 폴백
                .setErrorListener { e ->
                    errCount++
                    lastFrameReceivedTsMs = -1L
                    lastAlgoStartTsMs = -1L
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
        val frameReceivedTs = SystemClock.elapsedRealtime()             // 앱에서 프레임을 받은 시각(E2E 시작점)
        val tsMs = imageProxy.imageInfo.timestamp / 1_000_000L         // MediaPipe LIVE_STREAM용 입력 타임스탬프(ms)
        try {
            // 워치독: 콜백이 1.2초 넘게 안 오면 리셋
            if (inFlight && SystemClock.elapsedRealtime() - inFlightSince > 1200) {
                inFlight = false
                lastFrameReceivedTsMs = -1L
                lastAlgoStartTsMs = -1L
            }
            if (inFlight) { imageProxy.close(); return }

            // YUV→RGB, 회전 보정, 중앙 정사각형 크롭
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
            lastFrameReceivedTsMs = frameReceivedTs
            val algoStart = SystemClock.elapsedRealtime()
            lastAlgoStartTsMs = algoStart
            inFlight = true
            inFlightSince = algoStart

            // 추론 요청 (라이브 스트림 타임스탬프 전달)
            landmarker?.detectAsync(mpImg, tsMs)
        } catch (_: Throwable) {
            lastFrameReceivedTsMs = -1L
            lastAlgoStartTsMs = -1L
            inFlight = false
            onResult(null)
        } finally {
            imageProxy.close()
        }
    }

    fun close() {
        try { landmarker?.close() } catch (_: Exception) {}
        try { yuv.release() } catch (_: Exception) {}
    }
}
