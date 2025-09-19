package cc.ggrip.movenet.analysis

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.util.Log
import cc.ggrip.movenet.bench.Engine
import cc.ggrip.movenet.bench.ModelAssets
import cc.ggrip.movenet.bench.Tier
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.isActive
import kotlin.coroutines.coroutineContext
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

private const val TAG = "SwingAnalysis"
private const val TARGET_INFER_FPS = 15
private val STEP_MS = max(1L, (1000f / TARGET_INFER_FPS).roundToInt().toLong())
private const val HIP_BAND_TOL = 0.10f
private const val PRE_IMPACT_MS = 1_000L
private const val POST_IMPACT_MS = 2_200L
private const val COOLDOWN_MS = 900L
private const val MERGE_GAP_MS = 150L
private const val RECENT_TOP_MIN_MS = 120L
private const val RECENT_TOP_MAX_MS = 1_400L
private const val MAX_LONG_EDGE = 960
private const val DEQUEUE_TIMEOUT_US = 10_000L

class SwingDetector(
    private val context: Context,
    private val engine: Engine,
    private val tier: Tier
) {
    private val mediaMetadataRetriever = MediaMetadataRetriever()

    suspend fun analyze(
        uri: Uri,
        onProgress: suspend (processedMs: Long, totalMs: Long) -> Unit = { _, _ -> }
    ): List<VideoSegment> {
        mediaMetadataRetriever.setDataSource(context, uri)
        return try {
            val durationMs = mediaMetadataRetriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            if (durationMs <= 0L) return emptyList()

            val coarseWindows = coarseMotionWindows(uri, durationMs, onProgress)
            if (coarseWindows.isEmpty()) return emptyList()

            analyzeSwingWindows(uri, durationMs, coarseWindows, onProgress)
        } finally {
            try {
                mediaMetadataRetriever.release()
            } catch (t: Throwable) {
                Log.w(TAG, "retriever release failed: ${t.message}")
            }
        }
    }

    private suspend fun coarseMotionWindows(
        uri: Uri,
        durationMs: Long,
        onProgress: suspend (processedMs: Long, totalMs: Long) -> Unit
    ): List<Pair<Long, Long>> {
        val samplingFps = 5
        val stepMs = (1_000 / samplingFps).toLong()
        val minWindowMs = 500L
        val padWindowMs = 400L
        val diffThreshold = 0.08f

        onProgress(0L, durationMs)

        val extractor = MediaExtractor()
        extractor.setDataSource(context, uri, null)
        val trackIndex = (0 until extractor.trackCount).firstOrNull { index ->
            val format = extractor.getTrackFormat(index)
            format.getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true
        } ?: return emptyList()
        extractor.selectTrack(trackIndex)

        val targetW = 160
        val targetH = 90
        val prevPixels = IntArray(targetW * targetH)
        val currPixels = IntArray(targetW * targetH)
        var prevValid = false

        val windows = mutableListOf<Pair<Long, Long>>()
        var inMotion = false
        var windowStart = 0L
        var lastMotionMs = -1L

        var t = 0L
        while (t <= durationMs && coroutineContext.isActive) {
            val bitmap = getScaledFrameAtTime(uri, t, targetW, targetH)
            if (bitmap != null) {
                bitmap.getPixels(currPixels, 0, targetW, 0, 0, targetW, targetH)
                val ratio = if (prevValid) {
                    motionRatio(prevPixels, currPixels, targetW, targetH)
                } else {
                    0f
                }

                if (ratio > diffThreshold) {
                    if (!inMotion) {
                        inMotion = true
                        windowStart = (t - stepMs).coerceAtLeast(0L)
                    }
                    lastMotionMs = t
                } else if (inMotion && lastMotionMs >= 0 && (t - lastMotionMs) >= stepMs * 2) {
                    val rawStart = windowStart
                    val rawEnd = lastMotionMs
                    if (rawEnd - rawStart >= minWindowMs) {
                        val start = (rawStart - padWindowMs).coerceAtLeast(0L)
                        val end = (rawEnd + padWindowMs).coerceAtMost(durationMs)
                        windows += start to end
                    }
                    inMotion = false
                    lastMotionMs = -1L
                }

                currPixels.copyInto(prevPixels)
                prevValid = true
                bitmap.recycle()
            }

            t += stepMs
            onProgress(t.coerceAtMost(durationMs), durationMs)
            coroutineContext.ensureActive()
        }

        if (inMotion) {
            val rawEnd = lastMotionMs.takeIf { it >= 0 } ?: durationMs
            if (rawEnd - windowStart >= minWindowMs) {
                val start = (windowStart - padWindowMs).coerceAtLeast(0L)
                val end = (rawEnd + padWindowMs).coerceAtMost(durationMs)
                windows += start to end
            }
        }

        extractor.release()
        return mergeWindows(windows, 600L)
    }

    private suspend fun analyzeSwingWindows(
        uri: Uri,
        durationMs: Long,
        windows: List<Pair<Long, Long>>,
        onProgress: suspend (processedMs: Long, totalMs: Long) -> Unit
    ): List<VideoSegment> {
        val format = obtainVideoFormat(uri) ?: return emptyList()
        val (targetWidth, targetHeight) = chooseTargetSize(format)
        val longEdge = max(targetWidth, targetHeight)

        val reusableBitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(reusableBitmap)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)

        val poseLandmarker = createPoseLandmarker()
        val rawSegments = mutableListOf<Pair<Long, Long>>()

        try {
            for (window in windows) {
                val state = SwingState()
                decodeRangeToBitmapFrames(
                    uri = uri,
                    startMs = window.first,
                    endMs = window.second,
                    stepMs = STEP_MS,
                    targetWidth = targetWidth,
                    targetHeight = targetHeight,
                    reusableBitmap = reusableBitmap,
                    canvas = canvas,
                    paint = paint
                ) { bitmap, timestampMs ->
                    val segment = processFrame(
                        poseLandmarker = poseLandmarker,
                        bitmap = bitmap,
                        timestampMs = timestampMs,
                        window = window,
                        state = state,
                        targetWidth = targetWidth,
                        targetHeight = targetHeight,
                        longEdge = longEdge
                    )
                    if (segment != null) {
                        appendSegment(rawSegments, segment.first, segment.second)
                    }
                    onProgress(timestampMs.coerceAtMost(durationMs), durationMs)
                }
            }
        } finally {
            try {
                poseLandmarker.close()
            } catch (t: Throwable) {
                Log.w(TAG, "PoseLandmarker close failed: ${t.message}")
            }
            reusableBitmap.recycle()
        }

        return finalizeSegments(rawSegments)
    }

    private suspend fun decodeRangeToBitmapFrames(
        uri: Uri,
        startMs: Long,
        endMs: Long,
        stepMs: Long,
        targetWidth: Int,
        targetHeight: Int,
        reusableBitmap: Bitmap,
        canvas: Canvas,
        paint: Paint,
        frameHandler: suspend (Bitmap, Long) -> Unit
    ) {
        val extractor = MediaExtractor()
        extractor.setDataSource(context, uri, null)
        val trackIndex = (0 until extractor.trackCount).firstOrNull { index ->
            val format = extractor.getTrackFormat(index)
            format.getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true
        } ?: run {
            extractor.release()
            return
        }
        extractor.selectTrack(trackIndex)
        extractor.seekTo(startMs * 1_000, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)

        val format = extractor.getTrackFormat(trackIndex)
        val mime = format.getString(MediaFormat.KEY_MIME) ?: run {
            extractor.release()
            return
        }

        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(format, null, null, 0)
        codec.start()

        val bufferInfo = MediaCodec.BufferInfo()
        var inputDone = false
        var outputDone = false
        var lastDeliveredMs = Long.MIN_VALUE

        try {
            while (!outputDone && coroutineContext.isActive) {
                coroutineContext.ensureActive()

                if (!inputDone) {
                    val inputIndex = codec.dequeueInputBuffer(DEQUEUE_TIMEOUT_US)
                    if (inputIndex >= 0) {
                        val inputBuffer = codec.getInputBuffer(inputIndex)
                        val sampleSize = extractor.readSampleData(inputBuffer!!, 0)
                        if (sampleSize < 0) {
                            codec.queueInputBuffer(inputIndex, 0, 0, 0L, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            val sampleTimeUs = extractor.sampleTime
                            if (sampleTimeUs < 0 || sampleTimeUs / 1_000L > endMs + stepMs) {
                                codec.queueInputBuffer(inputIndex, 0, 0, 0L, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                                inputDone = true
                            } else {
                                codec.queueInputBuffer(
                                    inputIndex,
                                    0,
                                    sampleSize,
                                    sampleTimeUs,
                                    extractor.sampleFlags
                                )
                                extractor.advance()
                            }
                        }
                    }
                }

                val outputIndex = codec.dequeueOutputBuffer(bufferInfo, DEQUEUE_TIMEOUT_US)
                when {
                    outputIndex >= 0 -> {
                        val timestampMs = (bufferInfo.presentationTimeUs / 1_000L).coerceAtLeast(0L)
                        val isEos = bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                        val image = codec.getOutputImage(outputIndex)
                        if (image != null) {
                            try {
                                if (timestampMs in startMs..endMs && shouldDeliverFrame(timestampMs, lastDeliveredMs, stepMs)) {
                                    blitImageToBitmap(image, reusableBitmap, canvas, targetWidth, targetHeight, paint)
                                    lastDeliveredMs = timestampMs
                                    frameHandler(reusableBitmap, timestampMs)
                                }
                            } catch (t: Throwable) {
                                Log.w(TAG, "frame handler failed: ${t.message}")
                            } finally {
                                image.close()
                            }
                        }
                        codec.releaseOutputBuffer(outputIndex, false)
                        if (isEos || timestampMs > endMs) {
                            outputDone = true
                        }
                    }
                    outputIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                        if (inputDone) {
                            coroutineContext.ensureActive()
                        }
                    }
                    outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED ||
                        outputIndex == MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED -> {
                        // ignore
                    }
                }
            }
        } finally {
            codec.stop()
            codec.release()
            extractor.release()
        }
    }

    private fun processFrame(
        poseLandmarker: PoseLandmarker,
        bitmap: Bitmap,
        timestampMs: Long,
        window: Pair<Long, Long>,
        state: SwingState,
        targetWidth: Int,
        targetHeight: Int,
        longEdge: Int
    ): Pair<Long, Long>? {
        if (state.lastInferMs != Long.MIN_VALUE) {
            if (timestampMs <= state.lastInferMs) return null
            if (timestampMs - state.lastInferMs < STEP_MS) return null
        }

        val mpImage = BitmapImageBuilder(bitmap).build()
        val result = try {
            poseLandmarker.detectForVideo(mpImage, timestampMs)
        } catch (t: Throwable) {
            Log.w(TAG, "pose detect failed: ${t.message}")
            return null
        } finally {
            mpImage.close()
        }

        val landmarks = result.landmarks().firstOrNull() ?: return null
        if (landmarks.size < 33) return null

        val rightWrist = landmarks[16]
        val leftShoulder = landmarks[11]
        val rightShoulder = landmarks[12]
        val leftHip = landmarks[23]
        val rightHip = landmarks[24]

        val mirrored = rightShoulder.x() < leftShoulder.x()
        val shoulderLeftX = adjustX(leftShoulder.x(), mirrored)
        val shoulderRightX = adjustX(rightShoulder.x(), mirrored)
        val wristX = adjustX(rightWrist.x(), mirrored)
        val wristY = rightWrist.y()
        val hipLeftY = leftHip.y()
        val hipRightY = rightHip.y()

        if (!shoulderLeftX.isFinite() || !shoulderRightX.isFinite() || !wristX.isFinite() || !wristY.isFinite() || !hipLeftY.isFinite() || !hipRightY.isFinite()) {
            return null
        }

        val midShoulderX = (shoulderLeftX + shoulderRightX) * 0.5f
        val midHipY = (hipLeftY + hipRightY) * 0.5f
        val wristOffsetPx = (wristX - midShoulderX) * targetWidth
        val wristYPx = wristY * targetHeight

        val topTimestamp = state.wristTopDetector.addSample(timestampMs, wristY)
        if (topTimestamp != null) {
            state.lastTopMs = topTimestamp
        }

        var impact: Pair<Long, Long>? = null
        if (state.hasPrevious()) {
            val dtSec = (timestampMs - state.lastTimestampMs) / 1_000f
            if (dtSec > 0f) {
                val vxPx = (wristOffsetPx - state.lastShoulderOffsetPx) / dtSec
                val vyPx = (wristYPx - state.lastWristYPx) / dtSec
                if (!vxPx.isFinite() || !vyPx.isFinite()) return null

                val hipBandMin = midHipY - HIP_BAND_TOL
                val hipBandMax = midHipY + HIP_BAND_TOL
                val wasAboveHip = state.lastWyNorm < hipBandMin
                val nowInsideHip = wristY in hipBandMin..hipBandMax
                val recentTop = state.lastTopMs != Long.MIN_VALUE && (timestampMs - state.lastTopMs) in RECENT_TOP_MIN_MS..RECENT_TOP_MAX_MS
                val cooldownOk = timestampMs >= state.cooldownUntilMs
                val downMinSpeedPx = max(220f, longEdge * 0.22f)
                val downwardFast = vyPx > downMinSpeedPx

                if (cooldownOk && recentTop && wasAboveHip && nowInsideHip && downwardFast) {
                    val segStart = max(window.first, timestampMs - PRE_IMPACT_MS)
                    val segEnd = min(window.second, timestampMs + POST_IMPACT_MS)
                    if (segEnd > segStart) {
                        impact = segStart to segEnd
                        state.cooldownUntilMs = timestampMs + COOLDOWN_MS
                    }
                }
            }
        }

        state.lastInferMs = timestampMs
        state.lastTimestampMs = timestampMs
        state.lastShoulderOffsetPx = wristOffsetPx
        state.lastWristYPx = wristYPx
        state.lastWyNorm = wristY

        return impact
    }

    private fun adjustX(x: Float, mirrored: Boolean): Float {
        return if (mirrored) 1f - x else x
    }

    private fun appendSegment(segments: MutableList<Pair<Long, Long>>, start: Long, end: Long) {
        if (segments.isEmpty()) {
            segments += start to end
            return
        }
        val last = segments.last()
        if (start - last.second <= MERGE_GAP_MS) {
            segments[segments.lastIndex] = last.first to max(last.second, end)
        } else {
            segments += start to end
        }
    }

    private fun finalizeSegments(rawSegments: List<Pair<Long, Long>>): List<VideoSegment> {
        if (rawSegments.isEmpty()) return emptyList()
        val sorted = rawSegments.sortedBy { it.first }
        val merged = mutableListOf<Pair<Long, Long>>()
        for (segment in sorted) {
            if (merged.isEmpty()) {
                merged += segment
            } else {
                val last = merged.last()
                if (segment.first - last.second <= MERGE_GAP_MS) {
                    merged[merged.lastIndex] = last.first to max(last.second, segment.second)
                } else {
                    merged += segment
                }
            }
        }
        return merged.map { (start, end) ->
            VideoSegment(
                startFrameIndex = -1,
                endFrameIndex = -1,
                startTimeMs = start,
                endTimeMs = end
            )
        }
    }

    private fun createPoseLandmarker(): PoseLandmarker {
        val assetPath = when (engine) {
            Engine.MOVENET -> ModelAssets.mpTaskPath(tier)
            Engine.MEDIAPIPE -> ModelAssets.mpTaskPath(tier)
        }
        return try {
            PoseLandmarker.createFromOptions(context, landmarkerOptions(assetPath, Delegate.GPU))
        } catch (gpuError: Throwable) {
            Log.w(TAG, "GPU delegate unavailable: ${gpuError.message}")
            PoseLandmarker.createFromOptions(context, landmarkerOptions(assetPath, Delegate.CPU))
        }
    }

    private fun landmarkerOptions(assetPath: String, delegate: Delegate): PoseLandmarker.PoseLandmarkerOptions {
        val base = BaseOptions.builder()
            .setModelAssetPath(assetPath)
            .setDelegate(delegate)
            .build()
        return PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.VIDEO)
            .setNumPoses(1)
            .setMinPoseDetectionConfidence(0.4f)
            .setMinPosePresenceConfidence(0.4f)
            .setMinTrackingConfidence(0.4f)
            .build()
    }

    private fun obtainVideoFormat(uri: Uri): MediaFormat? {
        val extractor = MediaExtractor()
        return try {
            extractor.setDataSource(context, uri, null)
            val trackIndex = (0 until extractor.trackCount).firstOrNull { index ->
                val format = extractor.getTrackFormat(index)
                format.getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true
            } ?: return null
            extractor.getTrackFormat(trackIndex)
        } catch (t: Throwable) {
            Log.w(TAG, "obtain format failed: ${t.message}")
            null
        } finally {
            extractor.release()
        }
    }

    private fun chooseTargetSize(format: MediaFormat, maxLongEdge: Int = MAX_LONG_EDGE): Pair<Int, Int> {
        val width = format.optionalInt(MediaFormat.KEY_WIDTH)?.coerceAtLeast(1) ?: maxLongEdge
        val height = format.optionalInt(MediaFormat.KEY_HEIGHT)?.coerceAtLeast(1) ?: width
        val longEdge = max(width, height).coerceAtLeast(1)
        val scale = min(1f, maxLongEdge.toFloat() / longEdge)
        var targetW = max(1, (width * scale).roundToInt())
        var targetH = max(1, (height * scale).roundToInt())
        if (targetW % 2 != 0) targetW += 1
        if (targetH % 2 != 0) targetH += 1
        return targetW to targetH
    }

    private fun MediaFormat.optionalInt(key: String): Int? {
        return if (containsKey(key)) getInteger(key) else null
    }

    private fun shouldDeliverFrame(currentMs: Long, lastMs: Long, stepMs: Long): Boolean {
        if (lastMs == Long.MIN_VALUE) return true
        if (currentMs <= lastMs) return false
        return currentMs - lastMs >= stepMs
    }

    private fun blitImageToBitmap(
        image: android.media.Image,
        target: Bitmap,
        canvas: Canvas,
        targetWidth: Int,
        targetHeight: Int,
        paint: Paint
    ) {
        val bitmap = imageToBitmap(image)
        val srcRect = Rect(0, 0, bitmap.width, bitmap.height)
        val dstRect = Rect(0, 0, targetWidth, targetHeight)
        canvas.drawBitmap(bitmap, srcRect, dstRect, paint)
        bitmap.recycle()
    }

    private fun imageToBitmap(image: android.media.Image): Bitmap {
        val plane = image.planes[0]
        val buffer = plane.buffer
        buffer.rewind()
        val width = image.width
        val height = image.height
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.copyPixelsFromBuffer(buffer)
        return bitmap
    }

    private fun motionRatio(prev: IntArray, curr: IntArray, width: Int, height: Int): Float {
        var changed = 0
        var total = 0
        for (y in 0 until height) {
            for (x in 0 until width) {
                val index = y * width + x
                val p = prev[index]
                val c = curr[index]
                val dr = abs((p shr 16 and 0xFF) - (c shr 16 and 0xFF))
                val dg = abs((p shr 8 and 0xFF) - (c shr 8 and 0xFF))
                val db = abs((p and 0xFF) - (c and 0xFF))
                if (dr > 15 || dg > 15 || db > 15) {
                    changed++
                }
                total++
            }
        }
        return if (total == 0) 0f else changed.toFloat() / total.toFloat()
    }

    private fun getScaledFrameAtTime(
        uri: Uri,
        timeMs: Long,
        width: Int,
        height: Int
    ): Bitmap? {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            mediaMetadataRetriever.getScaledFrameAtTime(
                timeMs * 1_000,
                MediaMetadataRetriever.OPTION_CLOSEST,
                width,
                height
            )
        } else {
            mediaMetadataRetriever.getFrameAtTime(timeMs * 1_000, MediaMetadataRetriever.OPTION_CLOSEST)?.let { bitmap ->
                Bitmap.createScaledBitmap(bitmap, width, height, false).also {
                    if (it !== bitmap) {
                        bitmap.recycle()
                    }
                }
            }
        }
    }

    private fun mergeWindows(windows: List<Pair<Long, Long>>, mergeGapMs: Long): List<Pair<Long, Long>> {
        if (windows.isEmpty()) return emptyList()
        val sorted = windows.sortedBy { it.first }
        val merged = mutableListOf<Pair<Long, Long>>()
        var current = sorted.first()
        for (i in 1 until sorted.size) {
            val candidate = sorted[i]
            if (candidate.first - current.second <= mergeGapMs) {
                current = current.first to max(current.second, candidate.second)
            } else {
                merged += current
                current = candidate
            }
        }
        merged += current
        return merged
    }
}

private class SwingState {
    val wristTopDetector = WristTopDetector()
    var lastTopMs: Long = Long.MIN_VALUE
    var lastTimestampMs: Long = Long.MIN_VALUE
    var lastShoulderOffsetPx: Float = Float.NaN
    var lastWristYPx: Float = Float.NaN
    var lastWyNorm: Float = Float.NaN
    var lastInferMs: Long = Long.MIN_VALUE
    var cooldownUntilMs: Long = Long.MIN_VALUE

    fun hasPrevious(): Boolean {
        return lastTimestampMs != Long.MIN_VALUE &&
            lastShoulderOffsetPx.isFinite() &&
            lastWristYPx.isFinite() &&
            lastWyNorm.isFinite()
    }
}

private class WristTopDetector {
    private val values = FloatArray(5) { Float.NaN }
    private val times = LongArray(5) { Long.MIN_VALUE }
    private var count = 0

    fun addSample(timeMs: Long, y: Float): Long? {
        if (!y.isFinite()) return null
        if (count < values.size) {
            values[count] = y
            times[count] = timeMs
            count++
            return null
        }

        for (i in 0 until values.size - 1) {
            values[i] = values[i + 1]
            times[i] = times[i + 1]
        }
        values[values.lastIndex] = y
        times[times.lastIndex] = timeMs

        val midY = values[2]
        if (!midY.isFinite()) return null
        for (i in values.indices) {
            if (i == 2) continue
            val sample = values[i]
            if (!sample.isFinite() || midY >= sample) {
                return null
            }
        }
        return times[2]
    }
}