package cc.ggrip.movenet.analysis

import cc.ggrip.movenet.pose.PoseFrame
import kotlin.math.abs
import kotlin.math.max

data class FrameSample(
    val frameIndex: Int,
    val timestampMs: Long,
    val pose: PoseFrame?
)

data class VideoSegment(
    val startFrameIndex: Int,
    val endFrameIndex: Int,
    val startTimeMs: Long,
    val endTimeMs: Long
)

data class VideoAnalysisResult(
    val durationMs: Long,
    val frameCount: Int,
    val segments: List<VideoSegment>
)

object SwingSegmenter {
    private const val MIN_ACTIVITY_THRESHOLD = 0.015f

    fun detect(samples: List<FrameSample>): List<VideoSegment> {
        if (samples.size < 2) return emptyList()
        val valid = samples.filter { it.pose != null }
        if (valid.size < 2) return emptyList()

        val activity = FloatArray(valid.size)
        for (i in 1 until valid.size) {
            val prev = valid[i - 1].pose!!.screen2d
            val curr = valid[i].pose!!.screen2d
            val len = minOf(prev.size, curr.size)
            var sum = 0f
            var idx = 0
            while (idx < len) {
                val dx = curr[idx] - prev[idx]
                val dy = curr[idx + 1] - prev[idx + 1]
                sum += abs(dx) + abs(dy)
                idx += 2
            }
            activity[i] = sum
        }

        val peak = activity.maxOrNull() ?: 0f
        if (peak < MIN_ACTIVITY_THRESHOLD) return emptyList()
        val threshold = max(MIN_ACTIVITY_THRESHOLD, peak * 0.35f)

        val startIdx = (1 until activity.size).firstOrNull { activity[it] >= threshold } ?: return emptyList()
        val endIdx = (activity.size - 1 downTo startIdx).first { activity[it] >= threshold }

        val start = valid[startIdx]
        val end = valid[endIdx]
        if (start.timestampMs >= end.timestampMs) return emptyList()

        return listOf(
            VideoSegment(
                startFrameIndex = start.frameIndex,
                endFrameIndex = end.frameIndex,
                startTimeMs = start.timestampMs,
                endTimeMs = end.timestampMs
            )
        )
    }
}
