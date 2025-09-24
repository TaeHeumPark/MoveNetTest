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

    // 프레임 스킵
    private var frameCounter = 0
    private val FRAME_SKIP = 8 // 8프레임마다 1번만 처리 (더 공격적인 스킵)

    // 비동기 처리
    private val inferenceScope = CoroutineScope(Dispatchers.Default)
    private val isInferenceRunning = AtomicBoolean(false)
    private var lastInferenceResult: FloatArray? = null

    init {
        try {
            val modelBuffer = loadModelFile(context)
            val options = Interpreter.Options().apply {
                // FLK GRU 모델은 Select TF Ops가 필요할 수 있음
                setNumThreads(2) // 스레드를 줄여서 메인 스레드에 리소스 할당
                // NNAPI는 Select TF Ops와 호환되지 않을 수 있으므로 비활성화
                // setUseNNAPI(true)
                // Select TF Ops는 dependency만 추가하면 자동으로 활성화됨
            }
            interpreter = Interpreter(modelBuffer, options)
            Log.d(TAG, "FLK model loaded successfully from: $MODEL_PATH")

            interpreter?.let {
                val inShape = it.getInputTensor(0).shape()
                val outShape = it.getOutputTensor(0).shape()
                Log.d(TAG, "Input shape: ${inShape.contentToString()}  (expect [1,$WINDOW_SIZE,$INPUT_DIM])")
                Log.d(TAG, "Output shape: ${outShape.contentToString()} (expect [1,$INPUT_DIM])")
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

        // 프레임 스킵으로 성능 개선
        frameCounter++

        // 이미 추론 중이면 마지막 결과 반환
        if (isInferenceRunning.get()) {
            return lastInferenceResult ?: ring.last()
        }

        // 프레임 스킵 체크
        if (frameCounter % FRAME_SKIP != 0) {
            return lastInferenceResult ?: ring.last()
        }

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

                lastInferenceResult = result

                val elapsedTime = System.currentTimeMillis() - startTime
                if (elapsedTime > 50) {
                    Log.w(TAG, "FLK async inference took ${elapsedTime}ms")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Async inference failed", e)
            } finally {
                isInferenceRunning.set(false)
            }
        }
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
        isInferenceRunning.set(false)
    }

    fun close() {
        inferenceScope.cancel()
        try { interpreter?.close() } catch (_: Throwable) {}
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