// MediaPipePoseProcessor.kt
package cc.ggrip.movenet.tflite

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy
import cc.ggrip.movenet.pose.PoseFrame
import cc.ggrip.movenet.smoothing.FLKProcessor
import cc.ggrip.movenet.smoothing.PoseSmoother
import cc.ggrip.movenet.smoothing.SmoothingMode
import cc.ggrip.movenet.smoothing.Vec3
import cc.ggrip.movenet.tflite.YuvToRgb
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker.PoseLandmarkerOptions

private const val TAG_MP = "MPPose"

// MediaPipe 33 → FLK 12 관절 매핑
// FLK 순서: [ RHip, RKnee, RAnkle, LHip, LKnee, LAnkle, LShoulder, LElbow, LWrist, RShoulder, RElbow, RWrist ]
private val MP_33_TO_FLK12 = intArrayOf(
    24, 26, 28,   // RHip, RKnee, RAnkle
    23, 25, 27,   // LHip, LKnee, LAnkle
    11, 13, 15,   // LShoulder, LElbow, LWrist
    12, 14, 16    // RShoulder, RElbow, RWrist
)

private fun pickJoints3D(src33x3: FloatArray, idxs: IntArray): FloatArray {
    val out = FloatArray(idxs.size * 3)
    var o = 0
    for (id in idxs) {
        val b = id * 3
        out[o++] = src33x3[b]
        out[o++] = src33x3[b + 1]
        out[o++] = src33x3[b + 2]
    }
    return out
}

private fun pickVisibility(src33: FloatArray, idxs: IntArray): FloatArray {
    val out = FloatArray(idxs.size)
    for (i in idxs.indices) out[i] = src33[idxs[i]]
    return out
}

private fun mergeBack3D(dst33x3: FloatArray, patch12x3: FloatArray, idxs: IntArray) {
    var p = 0
    for (id in idxs) {
        val b = id * 3
        dst33x3[b]     = patch12x3[p++]
        dst33x3[b + 1] = patch12x3[p++]
        dst33x3[b + 2] = patch12x3[p++]
    }
}

class MediaPipePoseProcessor(
    context: Context,
    private val assetPath: String,
    private val onResult: (PoseFrame?) -> Unit
) {
    private val context: Context = context.applicationContext
    private val yuv = YuvToRgb(context)
    private var landmarker: PoseLandmarker? = null

    // 2D 표시용 EMA 상태 (joint별 x,y)
    private var ema2d: FloatArray? = null
    private var ema2dInit = false
    private val EMA2D_ALPHA = 0.35f  // 0.2~0.5 사이에서 취향 조정

    // 스무딩
    private val poseSmoother = PoseSmoother()
    private val flkProcessor = FLKProcessor(context)
    var smoothingMode: SmoothingMode = SmoothingMode.RAW
        set(value) {
            if (field != value) {
                Log.d(TAG_MP, "Smoothing mode changed from $field to $value")
                field = value
                // 모드 변경 시 내부 상태 초기화
                resetSmoothing()
            }
        }

    // 디바이스/백프레셔 & 타임스탬프
    @Volatile private var delegateLabel: String = "CPU"
    fun currentDelegate(): String = delegateLabel

    @Volatile private var inFlight = false
    @Volatile private var lastFrameReceivedTsMs: Long = -1L
    @Volatile private var inFlightSince: Long = 0L
    @Volatile private var lastAlgoStartTsMs: Long = -1L
    @Volatile private var errCount = 0

    // dt 계산용
    private var lastProcessTime = SystemClock.elapsedRealtime()

    init {
        // GPU 우선, 실패 시 CPU
        landmarker = tryCreate(Delegate.GPU) ?: tryCreate(Delegate.CPU)
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
                // result, inputImage (LIVE_STREAM)
                .setResultListener { result, _: MPImage ->
                    val frameTs = lastFrameReceivedTsMs
                    val algoStartTs = lastAlgoStartTsMs
                    val algoDone = SystemClock.elapsedRealtime()

                    val lm2d = result.landmarks().firstOrNull()
                    val lm3d = result.worldLandmarks().firstOrNull()

                    if (!lm2d.isNullOrEmpty() && !lm3d.isNullOrEmpty() && frameTs > 0 && algoStartTs > 0) {
                        val n = lm3d.size // 33
                        val screen2d = FloatArray(n * 2)
                        val world3d  = FloatArray(n * 3)
                        val visibility = FloatArray(n)

                        // dt 계산
                        val now = algoDone
                        val dt = ((now - lastProcessTime) / 1000f).coerceIn(0.001f, 1f)
                        lastProcessTime = now

                        // 추출
                        for (i in 0 until n) {
                            // 2D 정규화 좌표
                            val l2 = lm2d[i]
                            screen2d[i * 2]     = l2.x()
                            screen2d[i * 2 + 1] = l2.y()
                            visibility[i] = l2.visibility().orElse(1f)

                            // 3D 월드 좌표 (m 단위)
                            val l3 = lm3d[i]
                            world3d[i * 3]     = l3.x()
                            world3d[i * 3 + 1] = l3.y()
                            world3d[i * 3 + 2] = l3.z()
                        }

                        // EMA 준비(33관절 전체에 적용)
                        val ema33 = if (smoothingMode != SmoothingMode.RAW) {
                            val out = FloatArray(n * 3)
                            for (i in 0 until n) {
                                val b = i * 3
                                val v = Vec3(world3d[b], world3d[b + 1], world3d[b + 2])
                                val s = poseSmoother.smooth(i, v, dt, visibility[i])
                                out[b]     = s.x
                                out[b + 1] = s.y
                                out[b + 2] = s.z
                            }
                            out
                        } else world3d

                        val processedWorld33 = when (smoothingMode) {
                            SmoothingMode.RAW -> {
                                Log.d(TAG_MP, "Mode: RAW")
                                world3d
                            }
                            SmoothingMode.EMA -> {
                                Log.d(TAG_MP, "Mode: EMA, dt=$dt")
                                // 디버그: 양쪽 손목
                                val lW = 15 * 3; val rW = 16 * 3
                                Log.d(TAG_MP, "Wrist RAW L(${world3d[lW]}, ${world3d[lW+1]}, ${world3d[lW+2]}), " +
                                        "R(${world3d[rW]}, ${world3d[rW+1]}, ${world3d[rW+2]})")
                                Log.d(TAG_MP, "Wrist EMA L(${ema33[lW]}, ${ema33[lW+1]}, ${ema33[lW+2]}), " +
                                        "R(${ema33[rW]}, ${ema33[rW+1]}, ${ema33[rW+2]})")
                                ema33
                            }
                            SmoothingMode.FLK -> {
                                Log.d(TAG_MP, "Mode: FLK - Processing frame")
                                // 33→12 추출
                                val in12 = pickJoints3D(world3d, MP_33_TO_FLK12)
                                val vis12 = pickVisibility(visibility, MP_33_TO_FLK12)
                                Log.d(TAG_MP, "FLK input12 size: ${in12.size}, vis12 size: ${vis12.size}")

                                val out12 = flkProcessor.processFrame(in12, vis12)
                                val applied = ema33.copyOf() // FLK 없으면 EMA로 폴백

                                if (out12 != null && out12.size == in12.size) {
                                    Log.d(TAG_MP, "FLK output received, size: ${out12.size}")
                                    mergeBack3D(applied, out12, MP_33_TO_FLK12)
                                    val lW = 15 * 3; val rW = 16 * 3
                                    Log.d(TAG_MP, "Wrist RAW L(${world3d[lW]}, ${world3d[lW+1]}, ${world3d[lW+2]}), " +
                                            "R(${world3d[rW]}, ${world3d[rW+1]}, ${world3d[rW+2]})")
                                    Log.d(TAG_MP, "Wrist FLK L(${applied[lW]}, ${applied[lW+1]}, ${applied[lW+2]}), " +
                                            "R(${applied[rW]}, ${applied[rW+1]}, ${applied[rW+2]})")
                                } else {
                                    Log.d(TAG_MP, "FLK warmup → EMA fallback (out12: ${out12?.size ?: "null"})")
                                }
                                applied
                            }
                        }

                        // 화면 표시용 2D 투영(간단 Orthographic)
//                        val processed2d = if (smoothingMode == SmoothingMode.RAW) {
//                            screen2d
//                        } else {
//                            val proj = FloatArray(n * 2)
//                            for (i in 0 until n) {
//                                val b3 = i * 3
//                                val b2 = i * 2
//                                // 임시 정규화: X,Y만 사용, Y 반전
//                                proj[b2]     = (processedWorld33[b3] + 0.5f).coerceIn(0f, 1f)
//                                proj[b2 + 1] = (1f - processedWorld33[b3 + 1]).coerceIn(0f, 1f)
//                            }
//                            proj
//                        }
                        val processed2d = when (smoothingMode) {
                            SmoothingMode.RAW -> screen2d
                            SmoothingMode.EMA,
                            SmoothingMode.FLK -> smooth2D(screen2d)  // 2D에도 가벼운 EMA 적용
                        }

                        onResult(
                            PoseFrame(
                                tMillis = algoDone,
                                world = processedWorld33,
                                screen2d = processed2d,
                                visibility = visibility,
                                frameReceivedTsMs = frameTs,
                                algoStartTsMs = algoStartTs,
                                algoDoneTsMs = algoDone
                            )
                        )
                    } else {
                        onResult(null)
                    }

                    // 백프레셔 해제
                    errCount = 0
                    lastFrameReceivedTsMs = -1L
                    lastAlgoStartTsMs = -1L
                    inFlight = false
                }
                .setErrorListener { e ->
                    Log.w(TAG_MP, "PoseLandmarker error: ${e.message}")
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

            val inst = PoseLandmarker.createFromOptions(context, opts)
            delegateLabel = if (delegate == Delegate.GPU) "GPU" else "CPU"
            errCount = 0
            inst
        } catch (t: Throwable) {
            Log.w(TAG_MP, "create(delegate=$delegate) failed: ${t.message}")
            null
        }
    }

    private fun smooth2D(input: FloatArray, alpha: Float = EMA2D_ALPHA): FloatArray {
        if (!ema2dInit || ema2d == null || ema2d!!.size != input.size) {
            ema2d = input.copyOf()
            ema2dInit = true
            return ema2d!!
        }
        val out = ema2d!!
        for (i in input.indices) {
            out[i] = out[i] * (1f - alpha) + input[i] * alpha

            // ↓ 추가 (0..1 범위 보장)
            if ((i and 1) == 0) out[i] = out[i].coerceIn(0f, 1f)      // x
            else                 out[i] = out[i].coerceIn(0f, 1f)      // y
        }
        return out
    }

    @OptIn(ExperimentalGetImage::class)
    fun process(imageProxy: ImageProxy) {
        val frameReceivedTs = SystemClock.elapsedRealtime()
        val tsMs = imageProxy.imageInfo.timestamp / 1_000_000L // ns→ms
        try {
            // 콜백 타임아웃(워치독)
            if (inFlight && SystemClock.elapsedRealtime() - inFlightSince > 1200) {
                inFlight = false
                lastFrameReceivedTsMs = -1L
                lastAlgoStartTsMs = -1L
            }
            if (inFlight) { imageProxy.close(); return }

            // YUV → RGB
            val srcBmp = Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)
            imageProxy.image?.let { yuv.yuvToRgb(it, srcBmp) }

            // 회전 보정 후 정사각형 크롭
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

            // 비동기 추론
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

    fun resetSmoothing() {
        poseSmoother.reset()
        flkProcessor.reset()

        ema2d = null
        ema2dInit = false
    }

    fun close() {
        try { landmarker?.close() } catch (_: Exception) {}
        try { yuv.release() } catch (_: Exception) {}
        try { flkProcessor.close() } catch (_: Exception) {}
    }
}
