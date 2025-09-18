package cc.ggrip.movenet.analysis

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import cc.ggrip.movenet.pose.PoseFrame
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

const val TAG_ANALYZER = "VideoAnalyzer"

interface PoseFrameAnalyzer : Closeable {
    fun analyzeFrame(source: Bitmap, timestampMs: Long): PoseFrame?
}

class MoveNetVideoAnalyzer(
    context: Context,
    assetPath: String
) : PoseFrameAnalyzer {

    private val interpreter: Interpreter
    private val inputImage: TensorImage
    private val imageProcessor: ImageProcessor
    private val outputBuffer: TensorBuffer

    init {
        val model = loadModel(context, assetPath)
        val opts = Interpreter.Options().apply {
            setUseXNNPACK(true)
            setNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))
        }
        interpreter = Interpreter(model, opts)

        val inTensor = interpreter.getInputTensor(0)
        val inType = inTensor.dataType()
        val inShape = inTensor.shape()
        inputImage = TensorImage(inType)
        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(inShape[1], inShape[2], ResizeOp.ResizeMethod.BILINEAR))
            .apply { if (inType == DataType.FLOAT32) add(NormalizeOp(127.5f, 127.5f)) }
            .build()

        val outTensor = interpreter.getOutputTensor(0)
        outputBuffer = TensorBuffer.createFixedSize(outTensor.shape(), outTensor.dataType())
    }

    override fun analyzeFrame(source: Bitmap, timestampMs: Long): PoseFrame? {
        val square = cropCenterSquare(source)
        inputImage.load(square)
        if (square !== source) square.recycle()

        val processed = imageProcessor.process(inputImage)
        processed.buffer.rewind()
        outputBuffer.buffer.rewind()

        val algoStart = SystemClock.elapsedRealtime()
        interpreter.run(processed.buffer, outputBuffer.buffer)
        val algoDone = SystemClock.elapsedRealtime()

        val floats: FloatArray = when (outputBuffer.dataType) {
            DataType.FLOAT32 -> outputBuffer.floatArray
            DataType.UINT8 -> {
                val qp = interpreter.getOutputTensor(0).quantizationParams()
                val bb = outputBuffer.buffer
                bb.rewind()
                FloatArray(bb.remaining()) {
                    val u = bb.get().toInt() and 0xFF
                    (u - qp.zeroPoint) * qp.scale
                }
            }
            else -> return null
        }

        val screenCropNorm = FloatArray(17 * 2)
        for (i in 0 until 17) {
            val base = i * 3
            if (base + 1 >= floats.size) break
            val y = floats[base].coerceIn(0f, 1f)
            val x = floats[base + 1].coerceIn(0f, 1f)
            screenCropNorm[i * 2] = x
            screenCropNorm[i * 2 + 1] = y
        }

        return PoseFrame(
            tMillis = algoDone,
            world = floatArrayOf(),
            screen2d = screenCropNorm,
            visibility = null,
            frameReceivedTsMs = timestampMs,
            algoStartTsMs = algoStart,
            algoDoneTsMs = algoDone
        )
    }

    override fun close() {
        try {
            interpreter.close()
        } catch (t: Throwable) {
            Log.w(TAG_ANALYZER, "MoveNet interpreter close failed: ${t.message}")
        }
    }

    private fun cropCenterSquare(bitmap: Bitmap): Bitmap {
        val size = minOf(bitmap.width, bitmap.height)
        val left = (bitmap.width - size) / 2
        val top = (bitmap.height - size) / 2
        return if (size == bitmap.width && size == bitmap.height) {
            bitmap
        } else {
            Bitmap.createBitmap(bitmap, left, top, size, size)
        }
    }

    companion object {
        private fun loadModel(context: Context, assetPath: String): ByteBuffer {
            return try {
                context.assets.openFd(assetPath).use { fd ->
                    fd.createInputStream().channel.map(
                        FileChannel.MapMode.READ_ONLY,
                        fd.startOffset,
                        fd.declaredLength
                    ).order(ByteOrder.nativeOrder())
                }
            } catch (e: Throwable) {
                context.assets.open(assetPath).use { ins ->
                    val bytes = ins.readBytes()
                    ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply {
                        put(bytes)
                        rewind()
                    }
                }
            }
        }
    }
}

class MediaPipeVideoAnalyzer(
    context: Context,
    assetPath: String,
    delegate: Delegate = Delegate.CPU
) : PoseFrameAnalyzer {

    private val landmarker: PoseLandmarker

    init {
        val base = BaseOptions.builder()
            .setModelAssetPath(assetPath)
            .setDelegate(delegate)
            .build()

        val opts = PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.VIDEO)
            .setNumPoses(1)
            .setMinPoseDetectionConfidence(0.3f)
            .setMinPosePresenceConfidence(0.3f)
            .setMinTrackingConfidence(0.3f)
            .build()

        landmarker = PoseLandmarker.createFromOptions(context, opts)
    }

    override fun analyzeFrame(source: Bitmap, timestampMs: Long): PoseFrame? {
        val square = cropCenterSquare(source)
        val mpImage = BitmapImageBuilder(square).build()

        return try {
            val algoStart = SystemClock.elapsedRealtime()
            val result = landmarker.detectForVideo(mpImage, timestampMs)
            val algoDone = SystemClock.elapsedRealtime()

            val lmList = result.landmarks().firstOrNull() ?: return null
            val n = lmList.size
            val screen2d = FloatArray(n * 2)
            for (i in 0 until n) {
                val lm = lmList[i]
                screen2d[i * 2] = lm.x()
                screen2d[i * 2 + 1] = lm.y()
            }

            PoseFrame(
                tMillis = algoDone,
                world = floatArrayOf(),
                screen2d = screen2d,
                visibility = null,
                frameReceivedTsMs = timestampMs,
                algoStartTsMs = algoStart,
                algoDoneTsMs = algoDone
            )
        } finally {
            mpImage.close()
            if (square !== source) {
                square.recycle()
            }
        }
    }

    override fun close() {
        try {
            landmarker.close()
        } catch (t: Throwable) {
            Log.w(TAG_ANALYZER, "PoseLandmarker close failed: ${t.message}")
        }
    }

    private fun cropCenterSquare(bitmap: Bitmap): Bitmap {
        val size = minOf(bitmap.width, bitmap.height)
        val left = (bitmap.width - size) / 2
        val top = (bitmap.height - size) / 2
        return if (size == bitmap.width && size == bitmap.height) {
            bitmap
        } else {
            Bitmap.createBitmap(bitmap, left, top, size, size)
        }
    }
}
