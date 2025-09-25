package cc.ggrip.movenet.smoothing

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.sqrt
import kotlinx.coroutines.*
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.Executors
import android.os.Process
import kotlin.math.abs

/**
 * FLK TFLite 러너 (레포 GRU.h5 → TFLite 변환본과 호환)
 *
 * ▶ 입력 계약: 12관절 × 3D (=36) 순서로 들어와야 함.
 *    순서: [ RHip, RKnee, RAnkle, LHip, LKnee, LAnkle, LShoulder, LElbow, LWrist, RShoulder, RElbow, RWrist ]
 *    (33→12 추출과 12→33 되붙이기는 MediaPipePoseProcessor 쪽에서 수행)
 *
 * ▶ 정규화: 윈도우 첫 프레임의 힙센터를 모든 프레임에서 빼고,
 *            윈도우 전체 플랫 값의 μ/σ(스칼라)를 구해 (x-μ)/σ 로 표준화.
 *            추론 후 y*σ+μ 로 복원하고 힙센터 재적용.
 *
 * ▶ 모델 입출력: Input (1, 64, 36) → Output (1, 36)
 */
class FLKProcessor(context: Context) {

    companion object {
        private const val TAG = "FLKProcessor"

        // 앱 자산 경로: app/src/main/assets/flk_gru_from_repo.tflite
        private const val MODEL_PATH = "flk_gru_from_repo.tflite"

        private const val WINDOW_SIZE = 64          // 변환 로그: T=64 (모델이 요구하는 크기)
        private const val NUM_JOINTS = 12           // 변환 로그: D=36 → J=12
        private const val DIMS_PER_JOINT = 3
        private const val INPUT_DIM = NUM_JOINTS * DIMS_PER_JOINT // 36

        // FLK 12관절 순서 내 힙 인덱스 (삼중 시작 인덱스)
        private const val RHIP_BASE = 0 * 3         // RHip x,y,z
        private const val LHIP_BASE = 3 * 3         // LHip x,y,z
    }

    private var interpreter: Interpreter? = null
    private val ring: ArrayDeque<FloatArray> = ArrayDeque(WINDOW_SIZE)

    // 성능 최적화: 버퍼 재사용
    private val windowBuffer = FloatArray(WINDOW_SIZE * INPUT_DIM)
    private val inputBuffer = ByteBuffer.allocateDirect(4 * WINDOW_SIZE * INPUT_DIM).order(ByteOrder.nativeOrder())
    private val outputBuffer = ByteBuffer.allocateDirect(4 * INPUT_DIM).order(ByteOrder.nativeOrder())
    private val outputArray = FloatArray(INPUT_DIM)

    // 프레임 스킵 및 보간
    private var frameCounter = 0
    private val FRAME_SKIP = 3 // 3프레임마다 1번만 처리 (30fps -> 10Hz)
    private var lastInferenceResult: FloatArray? = null
    private var previousInferenceResult: FloatArray? = null
    private var lerpAlpha = 0f

    // 속도 기반 게이팅
    private var previousFrameTime = 0L
    private var previousPositions: FloatArray? = null
    private val VELOCITY_THRESHOLD = 0.05f // 속도 임계치
    private val SWING_ZONE_MIN_SPEED = 0.1f // 스윙 구간 최소 속도
    private var isInSwingZone = false

    // 고성능 백그라운드 스레드
    private val executorService = Executors.newSingleThreadExecutor { r ->
        Thread(r).apply {
            name = "FLK-Inference-Thread"
            priority = Thread.MAX_PRIORITY - 1 // 높은 우선순위
        }
    }
    private val inferenceScope = CoroutineScope(executorService.asCoroutineDispatcher())
    private val isInferenceRunning = AtomicBoolean(false)

    init {
        Log.d(TAG, "Starting FLK model initialization...")
        try {
            Log.d(TAG, "Loading model file from assets: $MODEL_PATH")
            val modelBuffer = loadModelFile(context)
            Log.d(TAG, "Model file loaded successfully, size: ${modelBuffer.capacity()} bytes")

            val options = Interpreter.Options()

            // GRU 모델은 Select TF Ops가 필요하므로 GPU는 사용하지 않음
            // GPU Delegate와 Select TF Ops는 호환되지 않을 수 있음
            Log.d(TAG, "Configuring CPU-only execution for GRU model with Select TF Ops")
            options.setNumThreads(4) // CPU 최적화: 4 스레드 사용
            options.setUseXNNPACK(true) // XNNPACK 가속 사용 (CPU 최적화)

            Log.d(TAG, "Creating TFLite interpreter with CPU optimization...")
            interpreter = Interpreter(modelBuffer, options)

            if (interpreter == null) {
                Log.e(TAG, "Interpreter creation returned null!")
            } else {
                Log.d(TAG, "FLK model loaded successfully from: $MODEL_PATH (CPU with XNNPACK)")

                interpreter?.let {
                    val inShape = it.getInputTensor(0).shape()
                    val outShape = it.getOutputTensor(0).shape()
                    Log.d(TAG, "Input shape: ${inShape.contentToString()}  (expect [1,$WINDOW_SIZE,$INPUT_DIM])")
                    Log.d(TAG, "Output shape: ${outShape.contentToString()} (expect [1,$INPUT_DIM])")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load FLK model from: $MODEL_PATH", e)
            Log.e(TAG, "Error details: ${e.message}")
            e.printStackTrace()
            interpreter = null
        }
    }

    private fun loadModelFile(context: Context): MappedByteBuffer {
        Log.d(TAG, "Attempting to load model from: $MODEL_PATH")
        val afd = context.assets.openFd(MODEL_PATH)
        Log.d(TAG, "AssetFileDescriptor obtained, offset: ${afd.startOffset}, length: ${afd.declaredLength}")
        FileInputStream(afd.fileDescriptor).use { fis ->
            return fis.channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
        }
    }

    /**
     * @param world12x3 길이 36 = 12관절×3D, FLK 순서로 들어와야 함
     * @param visibility12 길이 12 (선택), 현재는 사용하지 않음
     * @return 스무딩된 12×3 (길이 36). 아직 윈도우 미만이면 null
     */
    fun processFrame(world12x3: FloatArray, visibility12: FloatArray? = null): FloatArray? {
        // 속도 계산 및 게이팅
        val currentTime = System.currentTimeMillis()
        if (previousPositions != null && previousFrameTime > 0) {
            val dt = (currentTime - previousFrameTime) / 1000f // 초 단위
            val velocity = calculateVelocity(previousPositions!!, world12x3, dt)

            // 스윙 구간 판단 (속도가 높을 때만 FLK 활성화)
            isInSwingZone = velocity > SWING_ZONE_MIN_SPEED

            if (!isInSwingZone) {
                // 정지/슬로우 구간에서는 FLK 사용 안 함
                previousPositions = world12x3.copyOf()
                previousFrameTime = currentTime
                return world12x3 // 원본 데이터 그대로 반환
            }
        }
        previousPositions = world12x3.copyOf()
        previousFrameTime = currentTime

        if (interpreter == null) {
            Log.e(TAG, "FLK interpreter is null, model not loaded")
            return null
        }
        val itp = interpreter!!

        if (world12x3.size != INPUT_DIM) {
            Log.e(TAG, "Invalid input size: ${world12x3.size}, expected $INPUT_DIM")
            return null
        }

        if (ring.size == WINDOW_SIZE) ring.removeFirst()
        ring.addLast(world12x3.copyOf())

        if (ring.size < WINDOW_SIZE) {
            // 윈도우가 찰 때까지는 호출측에서 RAW/EMA를 쓰도록 null 반환
            Log.d(TAG, "FLK Warmup ${ring.size}/$WINDOW_SIZE")
            return null
        }

        // 프레임 스킵 및 보간 처리
        frameCounter++

        // 이미 추론 중이면 보간된 결과 반환
        if (isInferenceRunning.get()) {
            return interpolateResults()
        }

        // 프레임 스킵 체크 - 스킵 시 보간 사용
        if (frameCounter % FRAME_SKIP != 0) {
            lerpAlpha = (frameCounter % FRAME_SKIP).toFloat() / FRAME_SKIP
            return interpolateResults()
        }

        // 새로운 추론 시작
        lerpAlpha = 0f

        // 1) 힙센터(첫 프레임) 계산
        val first = ring.first()
        val hipX = 0.5f * (first[RHIP_BASE + 0] + first[LHIP_BASE + 0])
        val hipY = 0.5f * (first[RHIP_BASE + 1] + first[LHIP_BASE + 1])
        val hipZ = 0.5f * (first[RHIP_BASE + 2] + first[LHIP_BASE + 2])

        // 비동기로 추론 실행
        runInferenceAsync(ring.toList(), hipX, hipY, hipZ)

        // 비동기 처리 중 마지막 결과 반환
        return lastInferenceResult ?: ring.last()

    }

    private fun runInferenceAsync(windowFrames: List<FloatArray>, hipX: Float, hipY: Float, hipZ: Float) {
        if (!isInferenceRunning.compareAndSet(false, true)) return

        inferenceScope.launch {
            try {
                // 스레드 우선순위 설정
                Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)

                val startTime = System.currentTimeMillis()
                val itp = interpreter ?: return@launch

                // 2) 윈도우 전체에서 힙센터 제거
                var w = 0
                for (t in 0 until WINDOW_SIZE) {
                    val f = windowFrames[t]
                    var j = 0
                    while (j < INPUT_DIM) {
                        windowBuffer[w++] = f[j]     - hipX
                        windowBuffer[w++] = f[j + 1] - hipY
                        windowBuffer[w++] = f[j + 2] - hipZ
                        j += 3
                    }
                }

                // 3) μ/σ(스칼라) 표준화
                val mu = mean(windowBuffer)
                val sd = std(windowBuffer, mu).let { if (it < 1e-6f) 1f else it }
                for (i in windowBuffer.indices) windowBuffer[i] = (windowBuffer[i] - mu) / sd

                // 4) TFLite 입력 [1, T, D]
                inputBuffer.rewind()
                for (v in windowBuffer) inputBuffer.putFloat(v)
                inputBuffer.rewind()

                // 5) 출력 [1, D]
                outputBuffer.rewind()
                itp.run(inputBuffer, outputBuffer)

                // 6) 복원
                outputBuffer.rewind()
                val result = FloatArray(INPUT_DIM)
                for (i in 0 until INPUT_DIM) result[i] = outputBuffer.getFloat()

                var k = 0
                while (k < INPUT_DIM) {
                    result[k]     = result[k]     * sd + mu + hipX
                    result[k + 1] = result[k + 1] * sd + mu + hipY
                    result[k + 2] = result[k + 2] * sd + mu + hipZ
                    k += 3
                }

                // 이전 결과 저장 후 새 결과로 업데이트
                previousInferenceResult = lastInferenceResult
                lastInferenceResult = result
                lerpAlpha = 0f // 새 추론 완료 시 보간 지수 리셋

                val elapsedTime = System.currentTimeMillis() - startTime
                if (elapsedTime > 50) {
                    Log.w(TAG, "FLK CPU inference took ${elapsedTime}ms")
                } else {
                    Log.d(TAG, "FLK CPU inference completed in ${elapsedTime}ms")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Async inference failed", e)
            } finally {
                isInferenceRunning.set(false)
            }
        }
    }

    // 보간 함수 (Linear Interpolation)
    private fun interpolateResults(): FloatArray? {
        val current = lastInferenceResult ?: return ring.last()
        val previous = previousInferenceResult ?: return current

        // lerp: result = previous * (1 - alpha) + current * alpha
        val interpolated = FloatArray(INPUT_DIM)
        for (i in 0 until INPUT_DIM) {
            interpolated[i] = previous[i] * (1f - lerpAlpha) + current[i] * lerpAlpha
        }

        // 스무딩 품질 메트릭 계산 (주요 관절만)
        if (ring.size >= 2) {
            val raw = ring.last()
            calculateSmoothingMetrics(raw, interpolated)
        }

        return interpolated
    }

    // 스무딩 품질 메트릭 계산
    private fun calculateSmoothingMetrics(raw: FloatArray, smoothed: FloatArray) {
        // 손목 관절 (가장 움직임이 큰 부위)
        val leftWristIdx = 8 * 3  // LWrist
        val rightWristIdx = 11 * 3 // RWrist

        // 지터(떨림) 계산 - 연속 프레임 간 차이
        var rawJitter = 0f
        var smoothedJitter = 0f

        if (previousPositions != null) {
            // 왼손목 지터
            for (i in 0..2) {
                val rawDiff = abs(raw[leftWristIdx + i] - previousPositions!![leftWristIdx + i])
                val smoothDiff = abs(smoothed[leftWristIdx + i] -
                    (lastInferenceResult?.get(leftWristIdx + i) ?: smoothed[leftWristIdx + i]))
                rawJitter += rawDiff
                smoothedJitter += smoothDiff
            }

            // 오른손목 지터
            for (i in 0..2) {
                val rawDiff = abs(raw[rightWristIdx + i] - previousPositions!![rightWristIdx + i])
                val smoothDiff = abs(smoothed[rightWristIdx + i] -
                    (lastInferenceResult?.get(rightWristIdx + i) ?: smoothed[rightWristIdx + i]))
                rawJitter += rawDiff
                smoothedJitter += smoothDiff
            }

            // 지터 감소율 계산
            val jitterReduction = if (rawJitter > 0) {
                ((rawJitter - smoothedJitter) / rawJitter) * 100f
            } else 0f

            // 스무딩 강도 (원본과의 차이)
            var smoothingStrength = 0f
            for (i in 0..2) {
                smoothingStrength += abs(smoothed[leftWristIdx + i] - raw[leftWristIdx + i])
                smoothingStrength += abs(smoothed[rightWristIdx + i] - raw[rightWristIdx + i])
            }

            // 10프레임마다 메트릭 출력
            if (frameCounter % 10 == 0) {
                Log.d(TAG, "=== FLK Smoothing Metrics ===")
                Log.d(TAG, "Raw Jitter: %.4f, Smoothed Jitter: %.4f".format(rawJitter, smoothedJitter))
                Log.d(TAG, "Jitter Reduction: %.1f%%".format(jitterReduction))
                Log.d(TAG, "Smoothing Strength: %.4f".format(smoothingStrength))
                Log.d(TAG, "===========================")
            }
        }
    }

    // 속도 계산 (RMS of joint velocities)
    private fun calculateVelocity(prev: FloatArray, curr: FloatArray, dt: Float): Float {
        if (dt <= 0) return 0f

        var sumSquared = 0f
        var count = 0

        // 주요 관절(손목, 팔꿈치)의 속도만 계산
        val keyJoints = intArrayOf(8, 10, 11) // LWrist, RElbow, RWrist 인덱스
        for (jointIdx in keyJoints) {
            val base = jointIdx * 3
            val dx = curr[base] - prev[base]
            val dy = curr[base + 1] - prev[base + 1]
            val dz = curr[base + 2] - prev[base + 2]
            val distSquared = dx * dx + dy * dy + dz * dz
            sumSquared += distSquared
            count++
        }

        return if (count > 0) {
            sqrt(sumSquared / count) / dt
        } else 0f
    }

    private fun mean(arr: FloatArray): Float {
        var s = 0.0
        for (v in arr) s += v
        return (s / arr.size).toFloat()
    }

    private fun std(arr: FloatArray, mu: Float): Float {
        var s = 0.0
        for (v in arr) {
            val d = v - mu
            s += d * d
        }
        return sqrt((s / arr.size).toFloat())
    }

    fun reset() {
        Log.d(TAG, "FLK reset called, clearing ${ring.size} frames")
        ring.clear()
        frameCounter = 0
        lastInferenceResult = null
        previousInferenceResult = null
        previousPositions = null
        previousFrameTime = 0L
        isInSwingZone = false
        lerpAlpha = 0f
        isInferenceRunning.set(false)
    }

    fun close() {
        inferenceScope.cancel()
        executorService.shutdown()
        try {
            interpreter?.close()
        } catch (_: Throwable) {}
        interpreter = null
    }
}

/**
 * 스무딩 모드 enum
 */
enum class SmoothingMode {
    RAW,      // 원본 데이터
    EMA,      // Adaptive EMA 스무딩
    FLK       // FLK GRU 모델
}