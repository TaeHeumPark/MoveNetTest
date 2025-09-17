// MoveNetProcessor.kt
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
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList          // GPU 호환성 리스트
import org.tensorflow.lite.gpu.GpuDelegate               // GPU Delegate
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

private const val TAG = "MoveNet"

class MoveNetProcessor(
    context: Context,
    private val onResult: (PoseFrame?) -> Unit,
    assetPath: String
) {
    private val interpreter: Interpreter
    private val yuv = YuvToRgb(context)

    // GPU delegate 핸들 (AutoCloseable)
    private var gpuDelegate: GpuDelegate? = null
    @Volatile private var delegateLabel: String = "CPU(XNNPACK)"
    fun currentDelegate(): String = delegateLabel

    private lateinit var inputImage: TensorImage
    private lateinit var imageProcessor: ImageProcessor

    init {
        // 모델 로드
        // val model = loadModel(context, "models/movenet_thunder_fp16.tflite")
        val model = loadModel(context, assetPath)

        // 디바이스 능력에 따라: GPU 지원 시 GPU Delegate, 아니면 CPU(XNNPACK)
        val compat = try { CompatibilityList() } catch (t: Throwable) {
            Log.w(TAG, "GPU CompatibilityList not available: ${t.message}")
            null
        }

        val opts = Interpreter.Options()
        val gpuSupported = compat?.isDelegateSupportedOnThisDevice == true

        if (gpuSupported) {
            try {
                // 권장 옵션으로 GPU delegate 생성
                val dOpts = compat!!.bestOptionsForThisDevice
                // 추가 최적화(가능한 경우): FP16/양자화 모델 허용
                try { dOpts.setPrecisionLossAllowed(true) } catch (_: Throwable) {}
                try { dOpts.setQuantizedModelsAllowed(true) } catch (_: Throwable) {}

                gpuDelegate = GpuDelegate(dOpts)
                opts.addDelegate(gpuDelegate)
                delegateLabel = "GPU"
                Log.i(TAG, "Using GPU delegate")
            } catch (t: Throwable) {
                Log.w(TAG, "GPU delegate init failed, fallback to CPU: ${t.message}", t)
                // GPU 실패 시 CPU(XNNPACK)로 폴백
                opts.setUseXNNPACK(true)
                opts.setNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))
                delegateLabel = "CPU(XNNPACK)"
            }
        } else {
            // GPU 미지원: CPU(XNNPACK)
            opts.setUseXNNPACK(true)
            opts.setNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))
            delegateLabel = "CPU(XNNPACK)"
            Log.i(TAG, "GPU not supported on this device. Using CPU(XNNPACK).")
        }

        interpreter = try {
            Interpreter(model, opts)
        } catch (t: Throwable) {
            Log.w(TAG, "Interpreter init failed with $delegateLabel: ${t.message}. Falling back to pure CPU.")
            // 최종 폴백: 순수 CPU
            val cpuOnly = Interpreter.Options().apply {
                setUseXNNPACK(true)
                setNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))
            }
            // GPU delegate 자원 해제
            try { gpuDelegate?.close() } catch (_: Exception) {}
            gpuDelegate = null
            delegateLabel = "CPU(XNNPACK)"
            Interpreter(model, cpuOnly)
        }

        Log.i(TAG, "TFLite interpreter ready. delegate=$delegateLabel")

        // 입력 텐서/전처리 파이프라인 준비
        val inTensor = interpreter.getInputTensor(0)
        val inType = inTensor.dataType()
        val inShape = inTensor.shape() // [1, H, W, 3]
        inputImage = TensorImage(inType)

        val ipBuilder = ImageProcessor.Builder()
            .add(ResizeOp(inShape[1], inShape[2], ResizeOp.ResizeMethod.BILINEAR))
        if (inType == DataType.FLOAT32) {
            // [-1, 1] 정규화(일반적인 FP16/Float 입력)
            ipBuilder.add(NormalizeOp(127.5f, 127.5f))
        }
        imageProcessor = ipBuilder.build()

        Log.i(TAG, "Model input: type=$inType shape=${inShape.contentToString()}")
        val outTensor = interpreter.getOutputTensor(0)
        Log.i(TAG, "Model output: type=${outTensor.dataType()} shape=${outTensor.shape().contentToString()}")
    }

    fun close() {
        try { interpreter.close() } catch (_: Exception) {}
        try { gpuDelegate?.close() } catch (_: Exception) {} // GPU delegate 해제
        gpuDelegate = null
    }

    @OptIn(ExperimentalGetImage::class)
    fun process(imageProxy: ImageProxy) {
        // 앱이 프레임을 받은 시각(E2E 시작점; boottime ms)
        val frameReceivedTs = SystemClock.elapsedRealtime()
        try {
            // YUV → ARGB
            val srcBmp = Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)
            yuv.yuvToRgb(imageProxy.image!!, srcBmp)

            // 회전 보정
            val rot = imageProxy.imageInfo.rotationDegrees
            val mat = Matrix().apply { postRotate(rot.toFloat()) }
            val rotated = Bitmap.createBitmap(srcBmp, 0, 0, srcBmp.width, srcBmp.height, mat, false)

            // 중앙 정사각형 크롭
            val size = minOf(rotated.width, rotated.height)
            val left = (rotated.width  - size) / 2
            val top  = (rotated.height - size) / 2
            val square = Bitmap.createBitmap(rotated, left, top, size, size)

            // 텐서 로드 + 전처리
            inputImage.load(square)
            val inputBuffer = imageProcessor.process(inputImage).buffer
            inputBuffer.rewind()

            // 출력 버퍼 준비
            val outTensor = interpreter.getOutputTensor(0)
            val outShape = outTensor.shape()
            val outType  = outTensor.dataType()
            val outBuf   = TensorBuffer.createFixedSize(outShape, outType)

            // 알고리즘 지연: 추론 시작(카메라/전처리 제외) → 추론 종료
            val algoStart = SystemClock.elapsedRealtime()
            interpreter.run(inputBuffer, outBuf.buffer.rewind())
            val algoDone = SystemClock.elapsedRealtime()

            // MoveNet 출력: [1,1,17,3] (y, x, score)
            val floats: FloatArray = when (outType) {
                DataType.FLOAT32 -> outBuf.floatArray
                DataType.UINT8 -> {
                    val qp = outTensor.quantizationParams()
                    val bb = outBuf.buffer; bb.rewind()
                    FloatArray(bb.remaining()) {
                        val u = (bb.get().toInt() and 0xFF)
                        (u - qp.zeroPoint) * qp.scale
                    }
                }
                else -> {
                    Log.e(TAG, "Unsupported output type: $outType"); onResult(null); return
                }
            }

            // 화면 정규화 좌표(x,y)로 변환 (크롭 공간 기준)
            val screenCropNorm = FloatArray(17 * 2)
            for (i in 0 until 17) {
                val b = i * 3
                val y = floats[b].coerceIn(0f, 1f)
                val x = floats[b + 1].coerceIn(0f, 1f)
                screenCropNorm[i*2] = x
                screenCropNorm[i*2+1] = y
            }

            onResult(
                PoseFrame(
                    tMillis = algoDone,
                    world = floatArrayOf(),
                    screen2d = screenCropNorm,
                    visibility = null,
                    // E2E 시작: 앱이 프레임을 받은 시각
                    frameReceivedTsMs = frameReceivedTs,
                    // 알고리즘 지연: 추론 시작/종료
                    algoStartTsMs = algoStart,
                    algoDoneTsMs = algoDone
                )
            )
        } catch (t: Throwable) {
            Log.e(TAG, "inference failed: ${t.message}", t)
            onResult(null)
        } finally {
            imageProxy.close()
        }
    }

    // ---------- 모델 로딩 ----------
    private fun loadModel(context: Context, assetPath: String): ByteBuffer {
        return try {
            // AssetFileDescriptor 경로(권장): 메모리 매핑으로 로드
            context.assets.openFd(assetPath).use { fd ->
                fd.createInputStream().channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    fd.startOffset,
                    fd.declaredLength
                ).order(ByteOrder.nativeOrder())
            }
        } catch (e: Throwable) {
            // AFD가 안 될 때: 일반 스트림으로 로드
            Log.w(TAG, "openFd() failed ($assetPath). Falling back to stream load")
            context.assets.open(assetPath).use { ins ->
                val bytes = ins.readBytes()
                ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply {
                    put(bytes); rewind()
                }
            }
        }
    }
}
