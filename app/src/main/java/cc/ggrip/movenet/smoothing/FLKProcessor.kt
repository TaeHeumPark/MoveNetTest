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

        // 앱 자산 경로: app/src/main/assets/flk_gru.tflite
        private const val MODEL_PATH = "flk_gru.tflite"

        private const val WINDOW_SIZE = 64          // 변환 로그: T=64
        private const val NUM_JOINTS = 12           // 변환 로그: D=36 → J=12
        private const val DIMS_PER_JOINT = 3
        private const val INPUT_DIM = NUM_JOINTS * DIMS_PER_JOINT // 36

        // FLK 12관절 순서 내 힙 인덱스 (삼중 시작 인덱스)
        private const val RHIP_BASE = 0 * 3         // RHip x,y,z
        private const val LHIP_BASE = 3 * 3         // LHip x,y,z
    }

    private var interpreter: Interpreter? = null
    private val ring: ArrayDeque<FloatArray> = ArrayDeque(WINDOW_SIZE)

    init {
        try {
            val modelBuffer = loadModelFile(context)
            val options = Interpreter.Options().apply {
                // Select TF Ops 사용 시 보통 별도 delegate 추가 없이 동작
                // setNumThreads(2) // 필요 시
            }
            interpreter = Interpreter(modelBuffer, options)
            Log.d(TAG, "FLK model loaded")

            interpreter?.let {
                val inShape = it.getInputTensor(0).shape()
                val outShape = it.getOutputTensor(0).shape()
                Log.d(TAG, "Input shape: ${inShape.contentToString()}  (expect [1,$WINDOW_SIZE,$INPUT_DIM])")
                Log.d(TAG, "Output shape: ${outShape.contentToString()} (expect [1,$INPUT_DIM])")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load FLK model", e)
        }
    }

    private fun loadModelFile(context: Context): MappedByteBuffer {
        val afd = context.assets.openFd(MODEL_PATH)
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
        val itp = interpreter ?: return null
        require(world12x3.size == INPUT_DIM) { "FLKProcessor expects D=$INPUT_DIM (12 joints × 3D)" }

        if (ring.size == WINDOW_SIZE) ring.removeFirst()
        ring.addLast(world12x3.copyOf())

        if (ring.size < WINDOW_SIZE) {
            // 윈도우가 찰 때까지는 호출측에서 RAW/EMA를 쓰도록 null 반환
            Log.d(TAG, "Warmup ${ring.size}/$WINDOW_SIZE")
            return null
        }

        // 1) 힙센터(첫 프레임) 계산
        val first = ring.first()
        val hipX = 0.5f * (first[RHIP_BASE + 0] + first[LHIP_BASE + 0])
        val hipY = 0.5f * (first[RHIP_BASE + 1] + first[LHIP_BASE + 1])
        val hipZ = 0.5f * (first[RHIP_BASE + 2] + first[LHIP_BASE + 2])

        // 2) 윈도우 전체에서 힙센터 제거
        val window = FloatArray(WINDOW_SIZE * INPUT_DIM)
        var w = 0
        for (t in 0 until WINDOW_SIZE) {
            val f = ring.elementAt(t)
            var j = 0
            while (j < INPUT_DIM) {
                window[w++] = f[j]     - hipX
                window[w++] = f[j + 1] - hipY
                window[w++] = f[j + 2] - hipZ
                j += 3
            }
        }

        // 3) μ/σ(스칼라) 표준화 (윈도우 전체 평탄화 기준)
        val mu = mean(window)
        val sd = std(window, mu).let { if (it < 1e-6f) 1f else it }
        for (i in window.indices) window[i] = (window[i] - mu) / sd

        // 4) TFLite 입력 [1, T, D]
        val inBuf = ByteBuffer.allocateDirect(4 * WINDOW_SIZE * INPUT_DIM).order(ByteOrder.nativeOrder())
        for (v in window) inBuf.putFloat(v)
        inBuf.rewind()

        // 5) 출력 [1, D]
        val outBuf = ByteBuffer.allocateDirect(4 * INPUT_DIM).order(ByteOrder.nativeOrder())
        outBuf.rewind()
        try {
            itp.run(inBuf, outBuf)
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e)
            return ring.last() // 실패 시 최근 프레임 그대로 (힙센터 붙인 원본)
        }

        // 6) 복원: y*σ + μ 후 힙센터 되돌리기
        outBuf.rewind()
        val y = FloatArray(INPUT_DIM)
        for (i in 0 until INPUT_DIM) y[i] = outBuf.getFloat()

        var k = 0
        while (k < INPUT_DIM) {
            y[k]     = y[k]     * sd + mu + hipX
            y[k + 1] = y[k + 1] * sd + mu + hipY
            y[k + 2] = y[k + 2] * sd + mu + hipZ
            k += 3
        }
        return y
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
        ring.clear()
    }

    fun close() {
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