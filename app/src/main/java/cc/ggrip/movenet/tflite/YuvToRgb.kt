// YuvToRgb.kt
package cc.ggrip.movenet.tflite

import android.content.Context
import android.graphics.Bitmap
import android.renderscript.*
import java.io.Closeable

@Suppress("DEPRECATION")
class YuvToRgb(context: Context) : Closeable {
    private val rs = RenderScript.create(context)
    private val script = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs))
    private var yuvAllocation: Allocation? = null
    private var rgbAllocation: Allocation? = null
    private var cachedNv21: ByteArray? = null
    private var lastW = -1
    private var lastH = -1

    fun yuvToRgb(image: android.media.Image, output: Bitmap) {
        val nv21 = ImageUtil.yuv420ThreePlanesToNV21(
            image.planes, image.width, image.height, cachedNv21
        )
        cachedNv21 = nv21

        if (yuvAllocation == null || lastW != image.width || lastH != image.height) {
            yuvAllocation?.destroy()
            yuvAllocation = Allocation.createTyped(
                rs,
                Type.Builder(rs, Element.U8(rs)).setX(nv21.size).create(),
                Allocation.USAGE_SCRIPT
            )
        }
        if (rgbAllocation == null ||
            rgbAllocation!!.type.x != output.width || rgbAllocation!!.type.y != output.height
        ) {
            rgbAllocation?.destroy()
            rgbAllocation = Allocation.createTyped(
                rs,
                Type.Builder(rs, Element.RGBA_8888(rs))
                    .setX(output.width).setY(output.height).create(),
                Allocation.USAGE_SCRIPT
            )
        }
        lastW = image.width; lastH = image.height

        yuvAllocation!!.copyFrom(nv21)
        script.setInput(yuvAllocation)
        script.forEach(rgbAllocation)
        rgbAllocation!!.copyTo(output)
    }

    override fun close() {
        try { yuvAllocation?.destroy() } catch (_: Exception) {}
        try { rgbAllocation?.destroy() } catch (_: Exception) {}
        try { script.destroy() } catch (_: Exception) {}
        try { rs.destroy() } catch (_: Exception) {}
    }
}

object ImageUtil {
    fun yuv420ThreePlanesToNV21(
        planes: Array<android.media.Image.Plane>,
        width: Int,
        height: Int,
        reuse: ByteArray? = null
    ): ByteArray {
        val imageSize = width * height
        val outSize = imageSize + 2 * (imageSize / 4)
        val out = if (reuse != null && reuse.size >= outSize) reuse else ByteArray(outSize)

        val yPlane = planes[0].buffer
        val uPlane = planes[1].buffer
        val vPlane = planes[2].buffer

        yPlane.rewind()
        yPlane.get(out, 0, imageSize)

        val chromaHeight = height / 2
        val chromaWidth = width / 2
        var offset = imageSize
        val uRowStride = planes[1].rowStride
        val vRowStride = planes[2].rowStride
        val uPixelStride = planes[1].pixelStride
        val vPixelStride = planes[2].pixelStride
        for (row in 0 until chromaHeight) {
            val uRow = row * uRowStride
            val vRow = row * vRowStride
            for (col in 0 until chromaWidth) {
                out[offset++] = vPlane.get(vRow + col * vPixelStride)
                out[offset++] = uPlane.get(uRow + col * uPixelStride)
            }
        }
        return out
    }
}












