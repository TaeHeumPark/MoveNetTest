// MoveNetProcessor.kt
package cc.ggrip.movenet.tflite

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageProxy
import cc.ggrip.movenet.pose.PoseFrame
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MoveNetProcessor(
    context: Context,
    private val onResult: (PoseFrame?) -> Unit
) : Closeable {

    companion object { private const val TAG = "MoveNetProcessor" }

    private val interpreter: Interpreter
    private val yuv = YuvToRgb(context)

    // 입력 텐서 타입/전처리 체인을 **모델에서 자동 감지**해 셋업
    private val inputDataType: DataType
    private var inputImage: TensorImage
    private var baseProcessor: ImageProcessor
    private var rotProcessor: ImageProcessor
    private var lastRot90 = -1

    // 재사용 비트맵
    private var rgbBitmap: Bitmap? = null

    init {
        val model = loadModelBuffer(context, "models/movenet_lightning_fp16.tflite")
        val opts = Interpreter.Options()
        val compat = CompatibilityList()

        var gpuAdded = false
        if (compat.isDelegateSupportedOnThisDevice) {
            try { opts.addDelegate(GpuDelegate()); gpuAdded = true } catch (_: Throwable) {}
        }
        if (!gpuAdded) { opts.setNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4)) }
        interpreter = Interpreter(model, opts)

        // 입력 텐서 타입을 모델에서 조회
        inputDataType = interpreter.getInputTensor(0).dataType()
        inputImage = TensorImage(inputDataType)

        // 전처리: 데이터 타입별로 다르게
        //  - UINT8: 0~255 그대로
        //  - FLOAT32: [-1, 1]로 정규화(일반적인 MoveNet float 모델)
        baseProcessor = ImageProcessor.Builder()
            .add(ResizeOp(192, 192, ResizeOp.ResizeMethod.BILINEAR))
            .apply {
                if (inputDataType == DataType.FLOAT32) {
                    add(NormalizeOp(127.5f, 127.5f)) // (x - 127.5) / 127.5  → [-1,1]
                }
            }
            .build()

        // 회전은 매 프레임 입력 값에 따라 갱신
        rotProcessor = ImageProcessor.Builder().add(Rot90Op(0)).build()

        Log.i(TAG, "TFLite ready. inputDataType=$inputDataType, gpu=$gpuAdded")
    }

    override fun close() {
        try { interpreter.close() } catch (_: Exception) {}
        try { yuv.close() } catch (_: Exception) {}
        try { rgbBitmap?.recycle() } catch (_: Exception) {}
        rgbBitmap = null
    }

    fun process(imageProxy: ImageProxy) {
        val srcTsMs = imageProxy.imageInfo.timestamp / 1_000_000L
        try {
            val img = imageProxy.image ?: run { imageProxy.close(); return }
            val w = imageProxy.width
            val h = imageProxy.height

            var bmp = rgbBitmap
            if (bmp == null || bmp.width != w || bmp.height != h) {
                bmp?.recycle()
                bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                rgbBitmap = bmp
            }

            // YUV → ARGB
            yuv.yuvToRgb(img, bmp!!)

            // 회전 적용
            val rotDeg = ((imageProxy.imageInfo.rotationDegrees % 360) + 360) % 360
// Rot90Op는 "반시계(CCW) k회" 회전이므로, 시계방향 rotDeg를 CCW로 뒤집어 준다.
            val ccwTurns = when (rotDeg) { 0 -> 0; 90 -> 3; 180 -> 2; 270 -> 1; else -> 0 }
            if (ccwTurns != lastRot90) {
                rotProcessor = ImageProcessor.Builder().add(Rot90Op(ccwTurns)).build()
                lastRot90 = ccwTurns
            }

            // TensorImage 로드 + 전처리
            inputImage.load(bmp)
            val rotated = rotProcessor.process(inputImage)
            val preprocessed = baseProcessor.process(rotated)
            val inputBuffer = preprocessed.buffer

            // 출력 버퍼 생성 (동적 shape 대응)
            val outTensor = interpreter.getOutputTensor(0)
            val outShape = outTensor.shape() // 예: [1,1,17,3] 또는 [1,17,3]
            val out = TensorBuffer.createFixedSize(outShape, DataType.FLOAT32)

            val t0 = SystemClock.elapsedRealtime()
            interpreter.run(inputBuffer, out.buffer.rewind())
            val t1 = SystemClock.elapsedRealtime()

            // 결과 파싱
            val arr = out.floatArray
            val kpts = if (outShape.size >= 2) outShape[outShape.size - 2] else 17
            val chans = if (outShape.isNotEmpty()) outShape.last() else 3
            val step = chans
            val screen = FloatArray(kpts * 2)
            var j = 0
            var anyFinite = false
            for (i in 0 until kpts) {
                val base = i * step
                val y = arr[base + 0]
                val x = arr[base + 1]
                val xx = x.coerceIn(0f, 1f)
                val yy = y.coerceIn(0f, 1f)
                screen[j++] = xx
                screen[j++] = yy
                if (xx.isFinite() && yy.isFinite()) anyFinite = true
            }

            if (!anyFinite) {
                // 비정상 출력 방어
                onResult(null)
            } else {
                onResult(
                    PoseFrame(
                        tMillis = t1,
                        world = floatArrayOf(),
                        screen2d = screen,
                        visibility = null,
                        srcTsMs = srcTsMs,
                        algoDoneTsMs = t1
                    )
                )
            }
        } catch (t: Throwable) {
            Log.e(TAG, "inference failed", t)
            onResult(null)
        } finally {
            imageProxy.close()
        }
    }

    private fun loadModelBuffer(context: Context, assetPath: String): ByteBuffer {
        return try {
            context.assets.openFd(assetPath).use { fd ->
                fd.createInputStream().channel.map(
                    java.nio.channels.FileChannel.MapMode.READ_ONLY,
                    fd.startOffset, fd.declaredLength
                )
            }
        } catch (_: Throwable) {
            context.assets.open(assetPath).use { input ->
                val bytes = input.readBytes()
                val bb = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
                bb.put(bytes); bb.rewind(); bb
            }
        }
    }
}
