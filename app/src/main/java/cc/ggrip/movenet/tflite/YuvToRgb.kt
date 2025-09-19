// YuvToRgb.kt
package cc.ggrip.movenet.tflite

import android.content.Context
import android.graphics.Bitmap
import android.media.Image
import android.renderscript.*

// RenderScript는 deprecated지만 기존 프로젝트 호환을 위해 유지합니다.
@Suppress("DEPRECATION")
class YuvToRgb(context: Context) {
    private val rs = RenderScript.create(context)
    private val script = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs))

    private var yuvBuffer: ByteArray? = null
    private var inAllocation: Allocation? = null
    private var outAllocation: Allocation? = null
    private var cachedYuvSize = 0
    private var cachedOutWidth = 0
    private var cachedOutHeight = 0

    fun yuvToRgb(image: Image, output: Bitmap) {
        val requiredSize = image.width * image.height * 3 / 2
        val buffer = if (yuvBuffer?.size == requiredSize) {
            yuvBuffer!!
        } else {
            ByteArray(requiredSize).also {
                yuvBuffer = it
                cachedYuvSize = requiredSize
            }
        }

        ImageUtil.yuv420ThreePlanesToNV21(image.planes, image.width, image.height, buffer)
        ensureAllocations(buffer.size, output.width, output.height)

        inAllocation!!.copyFrom(buffer)
        script.setInput(inAllocation)
        script.forEach(outAllocation)
        outAllocation!!.copyTo(output)
    }

    private fun ensureAllocations(yuvSize: Int, outWidth: Int, outHeight: Int) {
        if (inAllocation == null || cachedYuvSize != yuvSize) {
            inAllocation?.destroy()
            val yuvType = Type.Builder(rs, Element.U8(rs)).setX(yuvSize).create()
            inAllocation = Allocation.createTyped(rs, yuvType, Allocation.USAGE_SCRIPT)
            cachedYuvSize = yuvSize
        }

        if (outAllocation == null || cachedOutWidth != outWidth || cachedOutHeight != outHeight) {
            outAllocation?.destroy()
            val rgbaType = Type.Builder(rs, Element.RGBA_8888(rs))
                .setX(outWidth)
                .setY(outHeight)
                .create()
            outAllocation = Allocation.createTyped(rs, rgbaType, Allocation.USAGE_SCRIPT)
            cachedOutWidth = outWidth
            cachedOutHeight = outHeight
        }
    }

    fun release() {
        try { inAllocation?.destroy() } catch (_: Exception) {}
        try { outAllocation?.destroy() } catch (_: Exception) {}
        try { script.destroy() } catch (_: Exception) {}
        try { rs.destroy() } catch (_: Exception) {}
    }
}

object ImageUtil {
    fun yuv420ThreePlanesToNV21(
        planes: Array<Image.Plane>,
        width: Int,
        height: Int,
        out: ByteArray
    ) {
        val imageSize = width * height
        val expectedSize = imageSize + 2 * (imageSize / 4)
        require(out.size == expectedSize) { "out buffer has wrong size: ${out.size} != $expectedSize" }

        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer
        yBuffer.position(0)
        uBuffer.position(0)
        vBuffer.position(0)

        val yPos = yBuffer.position()
        val uPos = uBuffer.position()
        val vPos = vBuffer.position()

        var outPos = 0
        if (yPlane.rowStride == width) {
            yBuffer.get(out, 0, imageSize)
            outPos += imageSize
        } else {
            var y = 0
            while (y < height) {
                yBuffer.position(y * yPlane.rowStride)
                yBuffer.get(out, outPos, width)
                outPos += width
                y++
            }
        }

        val chromaHeight = height / 2
        val chromaWidth = width / 2
        val chromaRowStride = uPlane.rowStride
        val chromaPixelStride = uPlane.pixelStride
        val vRowStride = vPlane.rowStride
        val vPixelStride = vPlane.pixelStride

        var row = 0
        while (row < chromaHeight) {
            var col = 0
            while (col < chromaWidth) {
                val vIndex = row * vRowStride + col * vPixelStride
                val uIndex = row * chromaRowStride + col * chromaPixelStride
                out[outPos] = vBuffer.get(vIndex)
                out[outPos + 1] = uBuffer.get(uIndex)
                outPos += 2
                col++
            }
            row++
        }

        yBuffer.position(yPos)
        uBuffer.position(uPos)
        vBuffer.position(vPos)
    }

}

