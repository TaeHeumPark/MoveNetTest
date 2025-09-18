package cc.ggrip.movenet.analysis

import android.content.Context
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import cc.ggrip.movenet.bench.Engine
import cc.ggrip.movenet.bench.ModelAssets
import cc.ggrip.movenet.bench.Tier
import com.google.mediapipe.tasks.core.Delegate
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.yield
import kotlin.coroutines.coroutineContext
import kotlin.math.ceil
import kotlin.math.max

class VideoAnalyzer(private val context: Context) {

    suspend fun analyze(
        uri: Uri,
        engine: Engine,
        tier: Tier,
        onProgress: (current: Int, total: Int) -> Unit = { _, _ -> }
    ): VideoAnalysisResult {
        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(context, uri)

        val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
        val frameRate = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)?.toFloatOrNull()?.takeIf { it > 0f } ?: 30f
        val durationUs = if (durationMs > 0L) durationMs * 1000 else 0L
        val frameStepUs = (1_000_000f / frameRate).toLong().coerceAtLeast(33_000L)
        val estimatedFrames = if (durationMs > 0L) {
            max(1, ceil(durationMs / (frameStepUs / 1000f)).toInt())
        } else {
            300
        }

        val analyzer = createAnalyzer(engine, tier)
        val samples = ArrayList<FrameSample>(estimatedFrames)

        try {
            var frameIndex = 0
            var timeUs = 0L
            while (durationUs == 0L || timeUs <= durationUs) {
                coroutineContext.ensureActive()
                val frameStart = SystemClock.elapsedRealtime()
                val bitmap = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST) ?: break
                val timestampMs = timeUs / 1000
                val pose = analyzer.analyzeFrame(bitmap, timestampMs)
                samples.add(FrameSample(frameIndex, timestampMs, pose))
                bitmap.recycle()
                frameIndex++
                onProgress(frameIndex, estimatedFrames)

                val next = timeUs + frameStepUs
                if (durationUs > 0 && next > durationUs) break
                timeUs = next

                if (SystemClock.elapsedRealtime() - frameStart > 32) {
                    yield()
                }
            }
        } catch (ce: CancellationException) {
            throw ce
        } catch (t: Throwable) {
            Log.e(TAG_ANALYZER, "Video analyze failed: ${t.message}", t)
        } finally {
            try {
                retriever.release()
            } catch (t: Throwable) {
                Log.w(TAG_ANALYZER, "retriever release failed: ${t.message}")
            }
            try {
                analyzer.close()
            } catch (t: Throwable) {
                Log.w(TAG_ANALYZER, "analyzer close failed: ${t.message}")
            }
        }

        val segments = SwingSegmenter.detect(samples)
        val finalDuration = if (durationMs > 0L) durationMs else samples.lastOrNull()?.timestampMs ?: 0L
        return VideoAnalysisResult(
            durationMs = finalDuration,
            frameCount = samples.size,
            segments = segments
        )
    }

    private fun createAnalyzer(engine: Engine, tier: Tier): PoseFrameAnalyzer {
        return when (engine) {
            Engine.MOVENET -> MoveNetVideoAnalyzer(context, ModelAssets.movenetPath(tier))
            Engine.MEDIAPIPE -> MediaPipeVideoAnalyzer(context, ModelAssets.mpTaskPath(tier), Delegate.CPU)
        }
    }
}