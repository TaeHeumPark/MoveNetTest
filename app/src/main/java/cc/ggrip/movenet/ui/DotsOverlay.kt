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
import cc.ggrip.movenet.analysis.SwingPhaseAnalysis
import cc.ggrip.movenet.analysis.GolfSwingPhase
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

    @Volatile private var swingAnalysis: SwingPhaseAnalysis? = null
    fun updateSwingState(analysis: SwingPhaseAnalysis) { swingAnalysis = analysis; postInvalidateOnAnimation() }

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

    // 스윙 상태 신호등 UI를 위한 페인트
    private val swingStatePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = 20f
        textAlign = Paint.Align.CENTER
    }
    private val swingIndicatorPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val swingIndicatorStrokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f
        color = 0xFFFFFFFF.toInt()
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

            // 스윙 상태 신호등 그리기
            drawSwingStateIndicator(canvas, W, swingAnalysis)

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

    private fun drawSwingStateIndicator(canvas: Canvas, screenWidth: Float, analysis: SwingPhaseAnalysis?) {
        // 스윙 단계들
        val phases = listOf(
            GolfSwingPhase.ADDRESS to "어드레스",
            GolfSwingPhase.TAKEAWAY to "테이크",
            GolfSwingPhase.BACKSWING to "백스윙",
            GolfSwingPhase.BACKSWING_TOP to "탑",
            GolfSwingPhase.DOWNSWING to "다운",
            GolfSwingPhase.IMPACT to "임팩트",
            GolfSwingPhase.FOLLOW_THROUGH to "팔로우",
            GolfSwingPhase.FINISH to "피니시"
        )

        val indicatorHeight = 60f
        val indicatorY = height - indicatorHeight - 40f  // 화면 하단에서 40px 위
        val padding = 16f
        val totalWidth = screenWidth - (padding * 2)
        val itemWidth = totalWidth / phases.size
        val cornerRadius = 12f

        // 배경 박스
        val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0x88000000.toInt()
            style = Paint.Style.FILL
        }
        canvas.drawRoundRect(
            padding, indicatorY - 10f,
            screenWidth - padding, indicatorY + indicatorHeight + 10f,
            cornerRadius, cornerRadius, bgPaint
        )

        phases.forEachIndexed { index, (phase, label) ->
            val x = padding + (index * itemWidth)
            val centerX = x + (itemWidth / 2)
            val centerY = indicatorY + (indicatorHeight / 2)

            // 현재 상태 확인
            val isCurrentPhase = analysis?.phase == phase
            val isCompletedPhase = analysis?.let { currentAnalysis ->
                phases.indexOfFirst { it.first == currentAnalysis.phase } >= index
            } ?: false

            // 원 그리기 (신호등)
            val circleRadius = 18f

            swingIndicatorPaint.color = when {
                isCurrentPhase -> {
                    // 현재 상태: 밝은 녹색 + 깜빡임 효과
                    val alpha = (((System.currentTimeMillis() / 300) % 2) * 127 + 128).toInt()
                    (alpha shl 24) or 0x00FF00
                }
                isCompletedPhase -> 0xFF4CAF50.toInt()  // 완료된 상태: 연한 녹색
                else -> 0xFF424242.toInt()  // 미도달 상태: 회색
            }

            canvas.drawCircle(centerX, centerY - 10f, circleRadius, swingIndicatorPaint)

            // 테두리
            if (isCurrentPhase) {
                swingIndicatorStrokePaint.strokeWidth = 3f
                canvas.drawCircle(centerX, centerY - 10f, circleRadius, swingIndicatorStrokePaint)
            }

            // 레이블
            swingStatePaint.color = if (isCurrentPhase || isCompletedPhase) {
                0xFFFFFFFF.toInt()
            } else {
                0xFF888888.toInt()
            }
            swingStatePaint.textSize = if (isCurrentPhase) 18f else 16f
            swingStatePaint.isFakeBoldText = isCurrentPhase

            canvas.drawText(label, centerX, centerY + 20f, swingStatePaint)

            // 신뢰도 표시 (현재 상태일 때만)
            if (isCurrentPhase && analysis != null) {
                val confidenceText = "${(analysis.confidence * 100).toInt()}%"
                swingStatePaint.textSize = 14f
                swingStatePaint.color = 0xFFFFCC00.toInt()
                canvas.drawText(confidenceText, centerX, centerY + 38f, swingStatePaint)
            }
        }
    }
}
