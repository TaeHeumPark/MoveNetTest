// DotsOverlay.kt
package cc.ggrip.movenet.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.os.SystemClock
import android.view.View
import cc.ggrip.movenet.pose.PoseFrame
import cc.ggrip.movenet.util.LatencyMeter
import kotlin.math.max
import kotlin.math.min

class DotsOverlay(
    context: Context,
    private val targetFps: Double,
    private val meter: LatencyMeter
) : View(context) {

    @Volatile private var frame: PoseFrame? = null
    @Volatile private var mirrorX: Boolean = false
    @Volatile private var flipY: Boolean = false

    @Volatile private var accelLabel: String = "CPU"
    fun setAcceleratorLabel(label: String) { accelLabel = label; postInvalidateOnAnimation() }

    @Volatile private var engineLabel: String = "MoveNet"
    fun setEngineLabel(label: String) { engineLabel = label; postInvalidateOnAnimation() }

    @Volatile private var modelLabel: String = "-"
    fun setModelLabel(label: String) { modelLabel = label; postInvalidateOnAnimation() }
    @Volatile private var firstFrameReceivedAtMs: Long = -1L
    @Volatile private var firstUiLatencyMs: Long = -1L

    @Volatile private var srcW: Int = 0
    @Volatile private var srcH: Int = 0

    fun setMirrorFlip(mirrorX: Boolean, flipY: Boolean) {
        this.mirrorX = mirrorX
        this.flipY = flipY
        invalidate()
    }

    fun setSourceSize(w: Int, h: Int) {
        if (w != srcW || h != srcH) {
            srcW = w; srcH = h
            postInvalidateOnAnimation()
        }
    }

    fun update(f: PoseFrame) {
        if (firstFrameReceivedAtMs < 0 && f.frameReceivedTsMs > 0) {
            firstFrameReceivedAtMs = f.frameReceivedTsMs
        }
        frame = f
        postInvalidateOnAnimation()
    }
    private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFCC00.toInt()
        style = Paint.Style.FILL
    }
    private val hudPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFFFFF.toInt()
        textSize = 36f
        setShadowLayer(4f, 1f, 1f, 0x80000000.toInt())
    }
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x66000000
        style = Paint.Style.FILL
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val W = width.toFloat()
        val H = height.toFloat()
        val f = frame

        if (f != null && srcW > 0 && srcH > 0) {
            // 화면에 맞게 원본을 letterbox/columnbox로 배치
            val scale = max(W / srcW, H / srcH)
            val offX = (W - srcW * scale) / 2f
            val offY = (H - srcH * scale) / 2f

            // 입력이 정사각형으로 크롭되어 온 경우를 고려한 매핑
            val cropSize = min(srcW, srcH).toFloat()
            val cropL = (srcW - cropSize) / 2f
            val cropT = (srcH - cropSize) / 2f

            val m = Matrix().apply {
                // 키포인트가 [0,1] 정규화 좌표라고 가정하고 원본 좌표로 투영
                setRectToRect(
                    RectF(0f, 0f, 1f, 1f),
                    RectF(cropL, cropT, cropL + cropSize, cropT + cropSize),
                    Matrix.ScaleToFit.FILL
                )
                // 화면 스케일/이동 적용
                postScale(scale, scale)
                postTranslate(offX, offY)
                // 미러/플립 옵션
                if (mirrorX || flipY) {
                    val px = offX + srcW * scale / 2f
                    val py = offY + srcH * scale / 2f
                    postScale(if (mirrorX) -1f else 1f, if (flipY) -1f else 1f, px, py)
                }
            }

            // 키포인트 그리기
            val p = f.screen2d
            val n = p.size / 2  // 17(MoveNet) 또는 33(MediaPipe) 등
            val tmp = FloatArray(2)
            for (i in 0 until n) {
                tmp[0] = p[i * 2]
                tmp[1] = p[i * 2 + 1]
                m.mapPoints(tmp)
                canvas.drawCircle(tmp[0], tmp[1], 10f, dotPaint)
            }

            // HUD: 지연 통계
            val nowMs = SystemClock.elapsedRealtime()
            val startTs = f.frameReceivedTsMs  // 앱이 프레임을 받은 시각(boottime ms) 사용
            var latestE2eMs = -1L
            if (startTs > 0) {
                val e2e = nowMs - startTs
                latestE2eMs = e2e
                val algo = if (f.algoStartTsMs > 0 && f.algoDoneTsMs > 0) {
                    f.algoDoneTsMs - f.algoStartTsMs
                } else -1L
                if (firstUiLatencyMs < 0 && firstFrameReceivedAtMs >= 0) {
                    firstUiLatencyMs = nowMs - firstFrameReceivedAtMs
                }
                meter.push(algo, e2e)
            }

            val stats = meter.snapshot()
            val frameInterval = 1000.0 / targetFps
            val eAvgF = if (!stats.e2eAvg.isNaN()) stats.e2eAvg / frameInterval else Double.NaN
            val eP95F = if (!stats.e2eP95.isNaN()) stats.e2eP95 / frameInterval else Double.NaN

            fun fmtMs(d: Double) = if (d.isNaN()) "-" else "%.1f".format(d)
            fun fmtFr(d: Double) = if (d.isNaN()) "-" else "%.2f".format(d)
            fun fmtMsLong(value: Long) = if (value < 0) "-" else fmtMs(value.toDouble())

            val lines = listOf(
                "$engineLabel • $modelLabel • 목표 ${"%.0f".format(targetFps)} FPS",
                "가속기: $accelLabel",
                "알고리즘 지연 평균/95퍼: ${fmtMs(stats.algoAvg)} / ${fmtMs(stats.algoP95)} ms",
                "E2E 평균/95퍼: ${fmtMs(stats.e2eAvg)} / ${fmtMs(stats.e2eP95)} ms",
                "카메라->UI(최근): ${fmtMsLong(latestE2eMs)} ms",
                "첫 프레임 지연: ${fmtMsLong(firstUiLatencyMs)} ms",
                "프레임 지연: ${fmtFr(eAvgF)}프 (평균) | ${fmtFr(eP95F)}프 (95p)"
            )

            val pad = 12f
            val boxW = lines.maxOf { hudPaint.measureText(it) } + pad * 2
            val boxH = hudPaint.textSize * lines.size + pad * 2
            canvas.drawRoundRect(16f, 16f, 16f + boxW, 16f + boxH, 18f, 18f, boxPaint)

            var yText = 16f + pad + hudPaint.textSize
            for (ln in lines) {
                canvas.drawText(ln, 16f + pad, yText, hudPaint)
                yText += hudPaint.textSize
            }
        } else {
            // 대기 메시지
            val msg = listOf(
                if (srcW == 0 || srcH == 0) "소스 크기 대기…" else "키포인트 대기…"
            )
            val pad = 12f
            val boxW = msg.maxOf { hudPaint.measureText(it) } + pad * 2
            val boxH = hudPaint.textSize * msg.size + pad * 2
            canvas.drawRoundRect(16f, 16f, 16f + boxW, 16f + boxH, 18f, 18f, boxPaint)

            var yText = 16f + pad + hudPaint.textSize
            for (ln in msg) {
                canvas.drawText(ln, 16f + pad, yText, hudPaint)
                yText += hudPaint.textSize
            }
        }
    }
}
