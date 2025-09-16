// YuvToRgb.kt
package cc.ggrip.movenet.tflite

import android.content.Context
import android.graphics.Bitmap
import android.renderscript.*

// RenderScript는 deprecated지만, 기존 프로젝트 호환을 위해 유지
@Suppress("DEPRECATION")
class YuvToRgb(context: Context) {
    private val rs = RenderScript.create(context)
    private val script = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs))

    fun yuvToRgb(image: android.media.Image, output: Bitmap) {
        val yuvBytes = ImageUtil.yuv420ThreePlanesToNV21(image.planes, image.width, image.height)

        // 프레임 단위 Allocation은 즉시 해제 (누수 로그 방지)
        val yuvType = Type.Builder(rs, Element.U8(rs)).setX(yuvBytes.size).create()
        val allocationYuv = Allocation.createTyped(rs, yuvType, Allocation.USAGE_SCRIPT)

        val rgbaType = Type.Builder(rs, Element.RGBA_8888(rs))
            .setX(output.width).setY(output.height).create()
        val allocationRgb = Allocation.createTyped(rs, rgbaType, Allocation.USAGE_SCRIPT)

        try {
            allocationYuv.copyFrom(yuvBytes)
            script.setInput(allocationYuv)
            script.forEach(allocationRgb)
            allocationRgb.copyTo(output)
        } finally {
            try { allocationYuv.destroy() } catch (_: Exception) {}
            try { allocationRgb.destroy() } catch (_: Exception) {}
        }
    }

    fun release() {
        try { script.destroy() } catch (_: Exception) {}
        try { rs.destroy() } catch (_: Exception) {}
    }
}

object ImageUtil {
    // 안정적인 NV21 변환 (Android 공식 샘플 패턴 기반)
    fun yuv420ThreePlanesToNV21(
        planes: Array<android.media.Image.Plane>,
        width: Int,
        height: Int
    ): ByteArray {
        val imageSize = width * height
        val out = ByteArray(imageSize + 2 * (imageSize / 4))

        // Y
        val yBuffer = planes[0].buffer
        val yRowStride = planes[0].rowStride
        var pos = 0
        if (yRowStride == width) {
            yBuffer.get(out, 0, imageSize)
            pos += imageSize
        } else {
            var y = 0
            while (y < height) {
                yBuffer.position(y * yRowStride)
                yBuffer.get(out, pos, width)
                pos += width
                y++
            }
        }

        // VU (NV21)
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer
        val chromaRowStride = planes[1].rowStride
        val chromaPixelStride = planes[1].pixelStride // usually 2

        // interleave V and U
        val chromaHeight = height / 2
        val chromaWidth = width / 2
        val vRowStride = planes[2].rowStride
        val vPixelStride = planes[2].pixelStride
        var row = 0
        while (row < chromaHeight) {
            var col = 0
            while (col < chromaWidth) {
                val vuIndex = pos
                val vIndex = row * vRowStride + col * vPixelStride
                val uIndex = row * chromaRowStride + col * chromaPixelStride
                out[vuIndex] = vBuffer.get(vIndex)
                out[vuIndex + 1] = uBuffer.get(uIndex)
                pos += 2
                col++
            }
            row++
        }
        return out
    }
}
