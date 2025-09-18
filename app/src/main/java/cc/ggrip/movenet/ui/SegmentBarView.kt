package cc.ggrip.movenet.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import cc.ggrip.movenet.analysis.VideoSegment

class SegmentBarView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val backgroundPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#33FFFFFF")
        style = Paint.Style.FILL
    }

    private val segmentPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FFFFCC00")
        style = Paint.Style.FILL
    }

    private var durationMs: Long = 0L
    private var segments: List<VideoSegment> = emptyList()
    private val rect = RectF()

    fun setSegments(durationMs: Long, segments: List<VideoSegment>) {
        this.durationMs = durationMs
        this.segments = segments
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val width = width.toFloat()
        val height = height.toFloat()
        if (width <= 0f || height <= 0f) return

        val left = paddingLeft.toFloat()
        val top = paddingTop.toFloat()
        val right = width - paddingRight.toFloat()
        val bottom = height - paddingBottom.toFloat()
        rect.set(left, top, right, bottom)
        val radius = (bottom - top) / 2f
        canvas.drawRoundRect(rect, radius, radius, backgroundPaint)

        if (durationMs <= 0L) return
        val denom = durationMs.toFloat()
        if (denom <= 0f) return

        val barWidth = rect.width()
        val startX = rect.left
        val segTop = rect.top
        val segBottom = rect.bottom

        segments.forEach { segment ->
            val start = segment.startTimeMs.coerceIn(0L, durationMs).toFloat() / denom
            val end = segment.endTimeMs.coerceIn(0L, durationMs).toFloat() / denom
            if (end <= start) return@forEach
            val segLeft = startX + barWidth * start
            val segRight = startX + barWidth * end
            rect.set(segLeft, segTop, segRight, segBottom)
            canvas.drawRoundRect(rect, radius, radius, segmentPaint)
        }
    }
}