package cc.ggrip.movenet.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import kotlin.math.max

class SegmentTimeBarView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    data class Segment(val startMs: Long, val endMs: Long)

    private var durationMs: Long = 0L
    private var positionMs: Long = 0L
    private val segments = mutableListOf<Segment>()

    /** 스크럽 콜백: 사용자가 바를 터치/드래그하면 호출 */
    var onScrubListener: ((Long) -> Unit)? = null

    private val bg = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF222222.toInt() }
    private val segStroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        color = Color.YELLOW
    }
    private val playhead = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        strokeWidth = 4f
    }

    fun setDuration(ms: Long) {
        durationMs = max(0L, ms)
        invalidate()
    }

    fun setPosition(ms: Long) {
        positionMs = ms.coerceIn(0, max(0L, durationMs))
        invalidate()
    }

    fun setSegments(newSegs: List<Segment>) {
        segments.clear()
        segments.addAll(newSegs)
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        canvas.drawRect(0f, 0f, w, h, bg)

        if (durationMs > 0) {
            for (s in segments) {
                val sx = (s.startMs.toFloat() / durationMs) * w
                val ex = (s.endMs.toFloat() / durationMs) * w
                val r = RectF(sx, 6f, ex, h - 6f)
                canvas.drawRoundRect(r, 10f, 10f, segStroke)
            }
            val px = (positionMs.toFloat() / durationMs) * w
            canvas.drawLine(px, 0f, px, h, playhead)
        }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (durationMs <= 0) return false
        return when (event.actionMasked) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_MOVE -> {
                val ratio = (event.x / width).coerceIn(0f, 1f)
                val ms = (ratio * durationMs).toLong()
                onScrubListener?.invoke(ms)
                true
            }
            else -> super.onTouchEvent(event)
        }
    }
}
