package cc.ggrip.movenet.ui

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.graphics.*
import android.media.*
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.OpenableColumns
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import cc.ggrip.movenet.R
import cc.ggrip.movenet.bench.Engine
import cc.ggrip.movenet.bench.ModelAssets
import cc.ggrip.movenet.bench.Tier
import com.google.android.exoplayer2.ExoPlayer
import com.google.android.exoplayer2.SeekParameters
import com.google.android.exoplayer2.ui.PlayerView
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker.PoseLandmarkerOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.coroutines.coroutineContext
import kotlin.math.max
import kotlin.math.roundToInt

class VideoAnalysisFragment : Fragment() {

    companion object {
        @JvmStatic fun newInstance() = VideoAnalysisFragment()
    }

    // UI
    private lateinit var playerView: PlayerView
    private lateinit var segmentBar: SegmentTimeBarView
    private lateinit var btnAnalyze: Button
    private lateinit var btnPrev: Button
    private lateinit var btnNext: Button
    private lateinit var selectVideoButton: Button
    private lateinit var modelSpinner: Spinner
    private lateinit var selectedVideoText: TextView
    private lateinit var segmentSummary: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var progressText: TextView

    // Player & state
    private var player: ExoPlayer? = null
    private var videoUri: Uri? = null
    private var frameStepMs: Long = 33L
    private var analyzerJob: Job? = null
    private var progressJob: Job? = null

    // 모델 선택 항목
    private data class ModelItem(val title: String, val engine: Engine, val tier: Tier)
    private val modelItems = listOf(
        ModelItem("MoveNet - lightning", Engine.MOVENET, Tier.LIGHT),
        ModelItem("MoveNet - thunder",   Engine.MOVENET, Tier.MID),
        ModelItem("MediaPipe - lite",    Engine.MEDIAPIPE, Tier.LIGHT),
        ModelItem("MediaPipe - full",    Engine.MEDIAPIPE, Tier.MID),
        ModelItem("MediaPipe - heavy",   Engine.MEDIAPIPE, Tier.HEAVY)
    )
    private var selectedModelIndex = 1 // 기본: MoveNet thunder

    // ====== 프리패스 파라미터 (리콜↑) ======
    private val preFps = 8                    // 5 → 8 (더 촘촘)
    private val preW = 160; private val preH = 90
    private val diffThreshold = 0.05f         // 0.08 → 0.05 (민감도↑)
    private val minWindowMs = 400L            // 500 → 400
    private val padWindowMs = 800L            // 400 → 800 (전후 넉넉)
    private val mergeGapMs = 1000L            // 600 → 1000 (근접 병합)

    // 추론 프레임률(리콜↑)
    private val TARGET_INFER_FPS = 20         // 15 → 20
    private val STEP_MS = 1000L / TARGET_INFER_FPS

    // 영상 단일 선택
    private val pickVideo = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        uri?.let {
            try {
                requireContext().contentResolver.takePersistableUriPermission(
                    it, Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            } catch (_: Exception) {}
            loadVideo(it)
        }
    }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        return inflater.inflate(R.layout.fragment_video_analysis, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        // Bind UI
        playerView = view.findViewById(R.id.playerView)
        segmentBar = view.findViewById(R.id.segmentBar)
        btnAnalyze = view.findViewById(R.id.btnAnalyze)
        btnPrev = view.findViewById(R.id.btnPrevFrame)
        btnNext = view.findViewById(R.id.btnNextFrame)
        selectVideoButton = view.findViewById(R.id.selectVideoButton)
        modelSpinner = view.findViewById(R.id.modelSpinner)
        selectedVideoText = view.findViewById(R.id.selectedVideoText)
        segmentSummary = view.findViewById(R.id.segmentSummary)
        progressBar = view.findViewById(R.id.analysisProgress)
        progressText = view.findViewById(R.id.progressText)

        // Spinner
        val titles = modelItems.map { it.title }
        modelSpinner.adapter = ArrayAdapter(requireContext(), android.R.layout.simple_spinner_dropdown_item, titles)
        modelSpinner.setSelection(selectedModelIndex)
        modelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, v: View?, position: Int, id: Long) {
                selectedModelIndex = position
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        // Video picker
        selectVideoButton.setOnClickListener { pickVideo.launch(arrayOf("video/*")) }

        // Player
        player = ExoPlayer.Builder(requireContext()).build().also {
            it.setSeekParameters(SeekParameters.EXACT)
            playerView.player = it
            it.addListener(object : com.google.android.exoplayer2.Player.Listener {
                override fun onEvents(p: com.google.android.exoplayer2.Player, events: com.google.android.exoplayer2.Player.Events) {
                    segmentBar.setPosition(p.currentPosition)
                }
            })
        }

        // Seek from segment bar
        segmentBar.onScrubListener = { ms ->
            player?.seekTo(ms)
            segmentBar.setPosition(ms)
        }

        // Analyze
        btnAnalyze.setOnClickListener {
            val uri = videoUri
            if (uri == null) {
                Toast.makeText(requireContext(), R.string.video_error_select_first, Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            startAnalyze(uri)
        }

        // Frame stepping
        btnPrev.setOnClickListener {
            val p = player ?: return@setOnClickListener
            val target = (p.currentPosition - frameStepMs).coerceAtLeast(0L)
            p.pause(); p.seekTo(target); segmentBar.setPosition(target)
        }
        btnNext.setOnClickListener {
            val p = player ?: return@setOnClickListener
            val dur = p.duration.takeIf { it > 0 } ?: Long.MAX_VALUE
            val target = (p.currentPosition + frameStepMs).coerceAtMost(dur)
            p.pause(); p.seekTo(target); segmentBar.setPosition(target)
        }

        startPositionTicker()
    }

    private fun startPositionTicker() {
        progressJob?.cancel()
        progressJob = viewLifecycleOwner.lifecycleScope.launch(Dispatchers.Main.immediate) {
            while (isActive) {
                val p = player
                val analyzing = analyzerJob?.isActive == true
                if (p != null && !analyzing) segmentBar.setPosition(p.currentPosition)
                kotlinx.coroutines.delay(50L)
            }
        }
    }

    override fun onDestroyView() {
        analyzerJob?.cancel()
        progressJob?.cancel()
        progressJob = null
        playerView.player = null
        player?.release()
        player = null
        super.onDestroyView()
    }

    /** 영상 로드 및 메타 초기화 */
    private fun loadVideo(uri: Uri) {
        videoUri = uri
        val p = player ?: return
        p.setMediaItem(com.google.android.exoplayer2.MediaItem.fromUri(uri))
        p.prepare()
        p.playWhenReady = false

        val mmr = MediaMetadataRetriever()
        mmr.setDataSource(requireContext(), uri)
        val durMs = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L

        val oneHourMs = 60 * 60 * 1000L
        if (durMs > oneHourMs) {
            Toast.makeText(requireContext(), "최대 1시간 길이의 영상만 열 수 있습니다.", Toast.LENGTH_LONG).show()
            segmentBar.setDuration(0L)
            player?.clearMediaItems()
            videoUri = null
            mmr.release()
            return
        }
        segmentBar.setDuration(durMs)

        val frameCount = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toLongOrNull()
        frameStepMs = if (durMs > 0 && frameCount != null && frameCount > 0) {
            (durMs / frameCount).coerceIn(10L, 100L)
        } else 33L
        mmr.release()

        // 파일명 표시
        selectedVideoText.text = getDisplayName(uri)?.let { "선택된 영상: $it" } ?: "선택된 영상: ${uri}"
        // 초기화
        segmentBar.setSegments(emptyList())
        segmentSummary.text = getString(R.string.video_segment_placeholder)
    }

    private fun getDisplayName(uri: Uri): String? {
        return try {
            requireContext().contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
                if (c.moveToFirst()) c.getString(0) else null
            }
        } catch (_: Exception) { null }
    }

    /** 분석 시작 */
    @RequiresApi(Build.VERSION_CODES.VANILLA_ICE_CREAM)
    private fun startAnalyze(uri: Uri) {
        analyzerJob?.cancel()
        player?.pause()
        setProgress(0, "프리패스 준비중...")

        analyzerJob = viewLifecycleOwner.lifecycleScope.launch(Dispatchers.Default) {
            // 0) Detector 준비 (선택된 모델)
            val choice = modelItems[selectedModelIndex]
            val detector = try {
                createDetector(choice)
            } catch (e: Throwable) {
                withContext(Dispatchers.Main) {
                    setProgressGone()
                    Toast.makeText(requireContext(), "모델 로드 실패: ${e.message}", Toast.LENGTH_LONG).show()
                }
                return@launch
            }

            try {
                // 1) 프리패스
                val windows = coarseMotionWindows(uri) { processedMs, totalMs ->
                    withContext(Dispatchers.Main) {
                        val pct = if (totalMs > 0) (processedMs * 100 / totalMs).toInt() else 0
                        setProgress(pct.coerceIn(0, 100), "프리패스 진행...$pct%")
                        segmentBar.setPosition(processedMs)
                    }
                }

                // 2) 본분석
                val segments = analyzeSwingSegments(uri, windows) { processedMs, totalMs ->
                    withContext(Dispatchers.Main) {
                        val pct = if (totalMs > 0) (processedMs * 100 / totalMs).toInt() else 0
                        setProgress(pct.coerceIn(0, 100), "분석 중...$pct%")
                        segmentBar.setPosition(processedMs)
                    }
                }

                // 3) UI 반영
                withContext(Dispatchers.Main) {
                    setProgressGone()
                    segmentBar.setSegments(segments.map { SegmentTimeBarView.Segment(it.first, it.second) })
                    val count = segments.size
                    val text = if (count == 0) "스윙 구간이 감지되지 않았습니다."
                    else "감지된 스윙: ${count}회  " +
                            segments.joinToString { "[${formatMs(it.first)} ~ ${formatMs(it.second)}]" }
                    segmentSummary.text = text
                    Toast.makeText(requireContext(), "분석 완료: ${segments.size}개 세그먼트", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Throwable) {
                Log.e("VideoAnalyze", "analyze failed", e)
                withContext(Dispatchers.Main) {
                    setProgressGone()
                    Toast.makeText(requireContext(), e.message ?: "분석 오류", Toast.LENGTH_LONG).show()
                }
            } finally {
                runCatching { detector.close() }
            }
        }
    }

    private fun setProgress(pct: Int, text: String) {
        if (progressBar.visibility != View.VISIBLE) {
            progressBar.visibility = View.VISIBLE
            progressText.visibility = View.VISIBLE
        }
        progressBar.progress = pct
        progressText.text = text
    }
    private fun setProgressGone() {
        progressBar.visibility = View.GONE
        progressText.visibility = View.GONE
        progressText.text = ""
        progressBar.progress = 0
    }
    private fun formatMs(ms: Long): String {
        val s = ms / 1000
        val m = s / 60
        val ss = s % 60
        val ms3 = (ms % 1000).toString().padStart(3, '0')
        return "%d:%02d.%s".format(m, ss, ms3)
    }

    // -----------------------------
    // 1) 프리패스: 저해상도/저FPS로 모션 구간 찾기
    // -----------------------------
    private suspend fun coarseMotionWindows(
        uri: Uri,
        onProgress: suspend (processedMs: Long, totalMs: Long) -> Unit = { _, _ -> }
    ): List<Pair<Long, Long>> {

        val mmr = MediaMetadataRetriever()
        mmr.setDataSource(requireContext(), uri)
        val durMs = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
        val stepMs = (1000 / preFps).toLong()
        onProgress(0L, durMs)

        val windows = mutableListOf<Pair<Long, Long>>()
        var inMotion = false
        var winStart = 0L
        var lastMotionMs = -1L

        var prevPixels: IntArray? = null
        var prevW = -1
        var prevH = -1

        var t = 0L
        try {
            while (t <= durMs && isAdded && coroutineContext.isActive) {
                val bmp: Bitmap? = withContext(Dispatchers.IO) {
                    try {
                        if (Build.VERSION.SDK_INT >= 27) {
                            mmr.getScaledFrameAtTime(
                                t * 1000L,
                                MediaMetadataRetriever.OPTION_CLOSEST,
                                preW, preH
                            )
                        } else {
                            mmr.getFrameAtTime(t * 1000L, MediaMetadataRetriever.OPTION_CLOSEST)
                                ?.let { Bitmap.createScaledBitmap(it, preW, preH, false) }
                        }
                    } catch (_: Exception) { null }
                }

                if (bmp != null) {
                    val use = if (bmp.width != preW || bmp.height != preH) {
                        try { Bitmap.createScaledBitmap(bmp, preW, preH, false) } catch (_: Throwable) { null }
                    } else bmp

                    if (use != null) {
                        val w = use.width
                        val h = use.height
                        val curr = IntArray(w * h)
                        try {
                            use.getPixels(curr, 0, w, 0, 0, w, h)
                        } catch (e: IllegalArgumentException) {
                            Log.w("Prepass", "getPixels mismatch, skip: ${e.message}")
                            if (use !== bmp) use.recycle()
                            bmp.recycle()
                            onProgress(t, durMs)
                            t += stepMs
                            continue
                        }

                        val ratio = if (prevPixels != null && prevW == w && prevH == h) {
                            motionRatio(prevPixels!!, curr, w, h)
                        } else 0f

                        if (ratio > diffThreshold) {
                            if (!inMotion) {
                                inMotion = true
                                winStart = (t - stepMs).coerceAtLeast(0L)
                            }
                            lastMotionMs = t
                        } else {
                            if (inMotion && lastMotionMs >= 0 && (t - lastMotionMs) >= (stepMs * 2)) {
                                val rawStart = winStart
                                val rawEnd = lastMotionMs
                                if (rawEnd - rawStart >= minWindowMs) {
                                    windows += ((rawStart - padWindowMs).coerceAtLeast(0L)) to
                                            ((rawEnd + padWindowMs).coerceAtMost(durMs))
                                }
                                inMotion = false
                                lastMotionMs = -1L
                            }
                        }

                        prevPixels = curr
                        prevW = w
                        prevH = h

                        if (use !== bmp) use.recycle()
                    }
                    bmp.recycle()
                }

                onProgress(t, durMs)
                t += stepMs
            }

            if (inMotion) {
                val rawStart = winStart
                val rawEnd = (lastMotionMs.takeIf { it >= 0 } ?: durMs)
                if (rawEnd - rawStart >= minWindowMs) {
                    windows += ((rawStart - padWindowMs).coerceAtLeast(0L)) to
                            ((rawEnd + padWindowMs).coerceAtMost(durMs))
                }
            }
        } finally {
            mmr.release()
        }

        if (windows.size <= 1) return windows
        val merged = mutableListOf<Pair<Long, Long>>()
        var (cs, ce) = windows.first()
        for (i in 1 until windows.size) {
            val (ns, ne) = windows[i]
            if (ns - ce <= mergeGapMs) ce = max(ce, ne) else { merged += cs to ce; cs = ns; ce = ne }
        }
        merged += cs to ce
        return merged
    }

    /** 간단한 밝기 diff 비율(저샘플링) */
    private fun motionRatio(prev: IntArray, curr: IntArray, w: Int, h: Int): Float {
        var changed = 0
        var total = 0
        val step = 4
        var y = 0
        while (y < h) {
            var x = 0
            val row = y * w
            while (x < w) {
                val i = row + x
                val p = prev[i]; val c = curr[i]
                val pr = (p shr 16) and 0xFF; val pg = (p shr 8) and 0xFF; val pb = p and 0xFF
                val cr = (c shr 16) and 0xFF; val cg = (c shr 8) and 0xFF; val cb = c and 0xFF
                val py = (30*pr + 59*pg + 11*pb)
                val cy = (30*cr + 59*cg + 11*cb)
                val diff = kotlin.math.abs(py - cy)
                if (diff > 22 * 100) changed++
                total++
                x += step
            }
            y += step
        }
        return if (total == 0) 0f else changed.toFloat() / total
    }

    // -----------------------------
    // 2) 본 분석: 선택한 Detector로 포즈 추정 → 스윙 조건 감지
    // -----------------------------
    @RequiresApi(Build.VERSION_CODES.VANILLA_ICE_CREAM)
    private suspend fun analyzeSwingSegments(
        uri: Uri,
        windows: List<Pair<Long, Long>>,
        onProgress: suspend (processedMs: Long, totalMs: Long) -> Unit = { _, _ -> }
    ): List<Pair<Long, Long>> {

        // 비디오 정보 & 진행률 초기화
        val info = readVideoInfo(requireContext(), uri)
        onProgress(0L, info.durationMs)

        // 타깃 해상도 선택 (decodeRangeToBitmapFrames와 공유)
        val (targetW, targetH) = chooseTargetSize(info)

        // PoseLandmarker (GPU 우선, 실패 시 CPU) — 모델은 풀(task) 고정
        val landmarker: PoseLandmarker = run {
            fun make(delegate: Delegate) =
                PoseLandmarker.createFromOptions(
                    requireContext(),
                    PoseLandmarkerOptions.builder()
                        .setBaseOptions(
                            BaseOptions.builder()
                                .setModelAssetPath("models/pose_landmarker_full.task")
                                .setDelegate(delegate)
                                .build()
                        )
                        .setRunningMode(RunningMode.VIDEO)
                        .setNumPoses(1)
                        .setMinPoseDetectionConfidence(0.4f)
                        .setMinTrackingConfidence(0.4f)
                        .setMinPosePresenceConfidence(0.4f)
                        .build()
                )
            try { make(Delegate.GPU) } catch (_: Exception) { make(Delegate.CPU) }
        }

        // 결과 세그먼트
        val segments = mutableListOf<Pair<Long, Long>>()

        // 재사용 비트맵 버퍼 (ARGB_8888)
        val reuse = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)

        // TOP 힌트(최근 TOP 이후에만 임팩트 인지)
        val yHist = FloatArray(5)
        var yCount = 0
        fun atLocalTop(): Boolean {
            if (yCount < yHist.size) return false
            val a=yHist[0]; val b=yHist[1]; val c=yHist[2]; val d=yHist[3]; val e=yHist[4]
            return c <= a && c <= b && c <= d && c <= e
        }
        var lastTopMs = Long.MIN_VALUE
        val minTopToImpactMs = 120L
        val maxTopToImpactMs = 1400L

        // 상태
        var lastInferMs = Long.MIN_VALUE
        var prevT: Long? = null
        var prevWxRel: Float? = null   // (wx - cx)
        var prevWy: Float? = null
        var cooldownUntilMs = Long.MIN_VALUE

        // 임팩트 중심 세그먼트 길이 & 임계치
        val longEdge = max(targetW, targetH).toFloat()
        val downMinSpeedPx = max(220f, longEdge * 0.22f) // ↓속도(다운스윙) 최소
        val hipBandTol = 0.10f                           // 힙 밴드 허용 폭(정규화 y)
        val preMs  = 1000L                               // 임팩트 이전
        val postMs = 2200L                               // 임팩트 이후
        val cooldownMs = 900L

        try {
            if (windows.isEmpty()) {
                onProgress(info.durationMs, info.durationMs)
                return emptyList()
            }

            // 시작 시각 오름차순으로 처리
            val ordered = windows.sortedBy { it.first }

            for ((ws, we) in ordered) {
                decodeRangeToBitmapFrames(
                    context = requireContext(),
                    uri = uri,
                    startMs = ws,
                    endMsInclusive = we,
                    targetWidth = targetW,
                    targetHeight = targetH,
                    reuseBitmap = reuse,
                    // frameStepMs는 기본값(약 15fps) 사용해도 OK. 원본 코드도 step 없이 동작.
                    onFrame = { tMs, bmp ->
                        if (tMs <= lastInferMs) return@decodeRangeToBitmapFrames
                        lastInferMs = tMs

                        val mpImg = com.google.mediapipe.framework.image.BitmapImageBuilder(bmp).build()
                        val result = try { landmarker.detectForVideo(mpImg, tMs) } finally { mpImg.close() }
                        val pose = result?.landmarks()?.firstOrNull()
                        if (pose == null || pose.size < 33) {
                            onProgress(tMs.coerceAtMost(info.durationMs), info.durationMs)
                            return@decodeRangeToBitmapFrames
                        }

                        fun lm(i: Int) = pose[i]
                        val lShoulder = lm(11); val rShoulder = lm(12)
                        val lHip = lm(23);     val rHip = lm(24)
                        val rWrist = lm(16)

                        val midShoulderX = (lShoulder.x() + rShoulder.x()) * 0.5f
                        val midHipY = (lHip.y() + rHip.y()) * 0.5f
                        // val shoulderSpan = kotlin.math.abs(rShoulder.x() - lShoulder.x()).coerceAtLeast(1e-3f)

                        // 좌우 반전 보정(X만)
                        val isMirrored = lShoulder.x() > rShoulder.x()
                        val wx = if (isMirrored) 1f - rWrist.x() else rWrist.x()
                        val cx = if (isMirrored) 1f - midShoulderX else midShoulderX
                        val wy = rWrist.y()

                        // TOP 힌트 갱신
                        if (yCount < yHist.size) yHist[yCount++] = wy
                        else { for (i in 1 until yHist.size) yHist[i-1] = yHist[i]; yHist[yHist.lastIndex] = wy }
                        if (atLocalTop()) lastTopMs = tMs

                        val pT = prevT; val pWx = prevWxRel; val pWy = prevWy
                        if (pT != null && pWx != null && pWy != null) {
                            val dt = (tMs - pT).coerceAtLeast(1L).toFloat()
                            // val vxPx = (((wx - cx) - pWx) * targetW) * (1000f / dt)   // (참고용)
                            val vyPx = ((wy - pWy) * targetH) * (1000f / dt)            // ↓ 방향 px/s

                            // 임팩트 트리거: 최근 TOP 이후 + 힙밴드 위→안 통과 + 충분한 ↓속도
                            val hipBot = midHipY - hipBandTol
                            val hipTop = midHipY + hipBandTol
                            val hadTopRecently = (lastTopMs != Long.MIN_VALUE) && (tMs - lastTopMs in minTopToImpactMs..maxTopToImpactMs)
                            val wasAboveHip = pWy < hipBot
                            val nowInHip    = wy >= hipBot && wy <= hipTop
                            val crossingDownIntoHip = wasAboveHip && nowInHip

                            if (tMs >= cooldownUntilMs &&
                                hadTopRecently &&
                                crossingDownIntoHip &&
                                vyPx > downMinSpeedPx) {

                                val segStart = (tMs - preMs).coerceAtLeast(ws)
                                val segEnd   = (tMs + postMs).coerceAtMost(we)

                                // 인접(≤150ms) 세그먼트 병합
                                if (segments.isNotEmpty() && segStart <= segments.last().second + 150L) {
                                    val last = segments.removeLast()
                                    segments += (last.first to max(last.second, segEnd))
                                } else {
                                    segments += (segStart to segEnd)
                                }
                                cooldownUntilMs = tMs + cooldownMs
                            }
                        }

                        prevT = tMs
                        prevWxRel = (wx - cx)
                        prevWy = wy

                        onProgress(tMs.coerceAtMost(info.durationMs), info.durationMs)
                    }
                )
            }

            // 정렬(안전)
            segments.sortBy { it.first }

            // 가까운 것만 보수적으로 병합(≤150ms)
            if (segments.size > 1) {
                val merged = mutableListOf<Pair<Long, Long>>()
                var cur = segments[0]
                for (i in 1 until segments.size) {
                    val nxt = segments[i]
                    if (nxt.first <= cur.second + 150L) {
                        cur = cur.first to max(cur.second, nxt.second)
                    } else {
                        merged += cur
                        cur = nxt
                    }
                }
                merged += cur
                segments.clear()
                segments.addAll(merged)
            }
        } catch (e: Throwable) {
            Log.e("VideoAnalyze", "analyzeSwingSegments failed", e)
            withContext(Dispatchers.Main) {
                Toast.makeText(requireContext(), e.message ?: "분석 중 오류", Toast.LENGTH_LONG).show()
            }
        } finally {
            runCatching { landmarker.close() }
        }

        return segments
    }

    // -------- Decoder --------
    @SuppressLint("Recycle")
    suspend fun decodeRangeToBitmapFrames(
        context: Context,
        uri: Uri,
        startMs: Long,
        endMsInclusive: Long,
        targetWidth: Int,
        targetHeight: Int,
        reuseBitmap: Bitmap,
        frameStepMs: Long = 66L,
        onFrame: suspend (timestampMs: Long, bitmap: Bitmap) -> Unit,
        onProgress: suspend (curMs: Long, endMs: Long) -> Unit = { _, _ -> }
    ) {
        fun pickSwH264(): String {
            val prefer = listOf("c2.android.avc.decoder", "OMX.google.h264.decoder")
            val infos = MediaCodecList(MediaCodecList.ALL_CODECS).codecInfos
            return infos.firstOrNull { info ->
                !info.isEncoder &&
                        info.supportedTypes.any { it.equals("video/avc", true) } &&
                        info.name in prefer
            }?.name ?: error("No software H.264 decoder found")
        }

        val extractor = MediaExtractor()
        var codec: MediaCodec? = null

        var targetBmp =
            if (!reuseBitmap.isRecycled
                && reuseBitmap.config == Bitmap.Config.ARGB_8888
                && reuseBitmap.isMutable
                && reuseBitmap.width == targetWidth
                && reuseBitmap.height == targetHeight
            ) reuseBitmap
            else Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)

        var scratchPixels: IntArray? = null

        try {
            extractor.setDataSource(context, uri, null)

            var track = -1
            var format: MediaFormat? = null
            for (i in 0 until extractor.trackCount) {
                val f = extractor.getTrackFormat(i)
                val mime = f.getString(MediaFormat.KEY_MIME) ?: ""
                if (mime.startsWith("video/")) { track = i; format = f; break }
            }
            require(track >= 0 && format != null) { "No video track" }
            extractor.selectTrack(track)

            codec = MediaCodec.createByCodecName(pickSwH264())
            val caps = codec.codecInfo.getCapabilitiesForType("video/avc")
            if (caps.colorFormats.contains(MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)) {
                format!!.setInteger(
                    MediaFormat.KEY_COLOR_FORMAT,
                    MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible
                )
            }
            if (Build.VERSION.SDK_INT >= 23) {
                format!!.setInteger(MediaFormat.KEY_OPERATING_RATE, 240)
            }
            codec.configure(format, null, null, 0)
            codec.start()

            val endMs = endMsInclusive + 16
            extractor.seekTo(startMs * 1000L, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)

            val info = MediaCodec.BufferInfo()
            var inputDone = false
            var outputDone = false
            var lastOutputWallMs = android.os.SystemClock.elapsedRealtime()

            while (!outputDone && coroutineContext.isActive) {
                // input
                if (!inputDone) {
                    val inIx = codec.dequeueInputBuffer(10_000)
                    if (inIx >= 0) {
                        val ib = codec.getInputBuffer(inIx)!!
                        val sampleTimeUs = extractor.sampleTime
                        val curMs = if (sampleTimeUs >= 0) sampleTimeUs / 1000L else -1L

                        if (sampleTimeUs < 0 || curMs > endMs) {
                            codec.queueInputBuffer(inIx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            val sz = extractor.readSampleData(ib, 0)
                            if (sz < 0) {
                                codec.queueInputBuffer(inIx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                                inputDone = true
                            } else {
                                codec.queueInputBuffer(inIx, 0, sz, sampleTimeUs, 0)
                                extractor.advance()
                            }
                        }
                    }
                }

                // output
                val outIx = codec.dequeueOutputBuffer(info, 10_000)
                when {
                    outIx == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                        if (inputDone && android.os.SystemClock.elapsedRealtime() - lastOutputWallMs > 2000) {
                            outputDone = true
                        }
                    }
                    outIx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> Unit
                    outIx >= 0 -> {
                        lastOutputWallMs = android.os.SystemClock.elapsedRealtime()
                        val tMs = info.presentationTimeUs / 1000L
                        val isEos = (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0

                        codec.getOutputImage(outIx)?.let { img ->
                            try {
                                val crop = img.cropRect ?: Rect(0, 0, img.width, img.height)
                                val w = crop.width()
                                val h = crop.height()

                                if (scratchPixels == null || scratchPixels!!.size < w * h) {
                                    scratchPixels = IntArray(w * h)
                                }
                                val pixels = scratchPixels!!

                                val planes = img.planes
                                val y = planes[0]; val u = planes[1]; val v = planes[2]
                                val yRS = y.rowStride; val yPS = y.pixelStride
                                val uRS = u.rowStride; val uPS = u.pixelStride
                                val vRS = v.rowStride; val vPS = v.pixelStride
                                val yBuf = y.buffer;   val uBuf = u.buffer;   val vBuf = v.buffer

                                val sx = crop.left; val sy = crop.top
                                var yy = 0
                                while (yy < h) {
                                    val yOff0 = (yy + sy) * yRS + sx * yPS
                                    val yOff1 = ((yy + 1).coerceAtMost(h - 1) + sy) * yRS + sx * yPS
                                    val uvRow = ((yy + sy) / 2)
                                    val uOffRow = uvRow * uRS
                                    val vOffRow = uvRow * vRS

                                    var xx = 0
                                    while (xx < w) {
                                        val x0 = xx
                                        val x1 = (xx + 1).coerceAtMost(w - 1)

                                        val yIx00 = yOff0 + x0 * yPS
                                        val yIx01 = yOff0 + x1 * yPS
                                        val yIx10 = yOff1 + x0 * yPS
                                        val yIx11 = yOff1 + x1 * yPS

                                        val uvCol = ((xx + sx) / 2)
                                        val uIx = uOffRow + uvCol * uPS
                                        val vIx = vOffRow + uvCol * vPS

                                        val Y00 = (yBuf.get(yIx00).toInt() and 0xFF)
                                        val Y01 = (yBuf.get(yIx01).toInt() and 0xFF)
                                        val Y10 = (yBuf.get(yIx10).toInt() and 0xFF)
                                        val Y11 = (yBuf.get(yIx11).toInt() and 0xFF)
                                        val U = (uBuf.get(uIx).toInt() and 0xFF)
                                        val V = (vBuf.get(vIx).toInt() and 0xFF)

                                        fun yuv(yv: Int, u_: Int, v_: Int): Int {
                                            val c = yv - 16
                                            val d = u_ - 128
                                            val e = v_ - 128
                                            var r = (298 * c + 409 * e + 128) shr 8
                                            var g = (298 * c - 100 * d - 208 * e + 128) shr 8
                                            var b = (298 * c + 516 * d + 128) shr 8
                                            if (r < 0) r = 0 else if (r > 255) r = 255
                                            if (g < 0) g = 0 else if (g > 255) g = 255
                                            if (b < 0) b = 0 else if (b > 255) b = 255
                                            return (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                                        }

                                        val row0 = yy * w
                                        pixels[row0 + x0] = yuv(Y00, U, V)
                                        if (x1 < w) pixels[row0 + x1] = yuv(Y01, U, V)
                                        if (yy + 1 < h) {
                                            val row1 = (yy + 1) * w
                                            pixels[row1 + x0] = yuv(Y10, U, V)
                                            if (x1 < w) pixels[row1 + x1] = yuv(Y11, U, V)
                                        }

                                        xx += 2
                                    }
                                    yy += 2
                                }

                                val tmpArgb = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                                tmpArgb.setPixels(pixels, 0, w, 0, 0, w, h)

                                if (targetBmp.isRecycled
                                    || targetBmp.config != Bitmap.Config.ARGB_8888
                                    || !targetBmp.isMutable
                                    || targetBmp.width != targetWidth
                                    || targetBmp.height != targetHeight
                                ) {
                                    targetBmp = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
                                }

                                val canvas = Canvas(targetBmp)
                                if (tmpArgb.width == targetBmp.width && tmpArgb.height == targetBmp.height) {
                                    canvas.drawBitmap(tmpArgb, 0f, 0f, null)
                                } else {
                                    canvas.drawBitmap(tmpArgb, null, Rect(0, 0, targetBmp.width, targetBmp.height), null)
                                }

                                onFrame(tMs, targetBmp)
                                onProgress(tMs, endMs)

                                tmpArgb.recycle()
                            } finally {
                                img.close()
                            }
                        }

                        codec.releaseOutputBuffer(outIx, false)
                        if (isEos || tMs >= endMs) outputDone = true

                        onProgress(tMs.coerceAtMost(endMs), endMs)
                    }
                }
            }
        } finally {
            runCatching { codec?.stop() }
            runCatching { codec?.release() }
            runCatching { extractor.release() }
        }
    }

    private data class VideoInfo(
        val durationMs: Long,
        val width: Int,
        val height: Int,
        val rotation: Int
    )

    private fun readVideoInfo(ctx: Context, uri: Uri): VideoInfo {
        val mmr = MediaMetadataRetriever()
        return try {
            mmr.setDataSource(ctx, uri)
            val duration = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            val w = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
            val h = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
            val rot = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull() ?: 0
            VideoInfo(duration, w, h, rot)
        } finally {
            mmr.release()
        }
    }

    private fun chooseTargetSize(info: VideoInfo, maxLongEdge: Int = 960): Pair<Int, Int> {
        val orientedW = if (info.rotation % 180 != 0) info.height else info.width
        val orientedH = if (info.rotation % 180 != 0) info.width else info.height
        val scale = (maxLongEdge.toDouble() / max(orientedW, orientedH)).coerceAtMost(1.0)
        var tw = max(64, (orientedW * scale).roundToInt())
        var th = max(64, (orientedH * scale).roundToInt())
        if (tw % 2 != 0) tw--
        if (th % 2 != 0) th--
        return tw to th
    }

    // ============================
    // Pose Detector 추상화 + 구현
    // ============================

    // ============================
// Pose Detector 추상화 + 구현
// ============================

    private interface PoseDetector : AutoCloseable {
        /** 결과 좌표는 0..1 정규화. 누락 시 null */
        fun detect(bmp: Bitmap, tMs: Long): Landmarks?
    }

    private data class Landmarks(
        val xs: FloatArray,   // 전 랜드마크 x (0..1)
        val ys: FloatArray,   // 전 랜드마크 y (0..1)
        val lWristX: Float?, val lWristY: Float?,
        val rWristX: Float?, val rWristY: Float?,
        val lHandX: Float?,  val lHandY: Float?,   // MP Pose: 19(index),  MoveNet: wrist로 대체
        val rHandX: Float?,  val rHandY: Float?,   // MP Pose: 20(index),  MoveNet: wrist로 대체
        val isMirrored: Boolean
    )

    private fun createDetector(mi: ModelItem): PoseDetector {
        return when (mi.engine) {
            Engine.MEDIAPIPE -> MediaPipeTaskDetector(requireContext(), ModelAssets.mpTaskPath(mi.tier))
            Engine.MOVENET   -> MoveNetDetector(requireContext(), ModelAssets.movenetPath(mi.tier), mi.tier)
        }
    }

    private class MediaPipeTaskDetector(
        private val ctx: Context,
        private val assetTaskPath: String
    ) : PoseDetector {
        private val landmarker: PoseLandmarker = run {
            fun make(d: Delegate) = PoseLandmarker.createFromOptions(
                ctx,
                PoseLandmarkerOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setModelAssetPath(assetTaskPath).setDelegate(d).build())
                    .setRunningMode(RunningMode.VIDEO)
                    .setNumPoses(1)
                    .setMinPoseDetectionConfidence(0.4f)
                    .setMinPosePresenceConfidence(0.4f)
                    .setMinTrackingConfidence(0.4f)
                    .build()
            )
            try { make(Delegate.GPU) } catch (_: Throwable) { make(Delegate.CPU) }
        }

        override fun detect(bmp: Bitmap, tMs: Long): Landmarks? {
            val mpImg = BitmapImageBuilder(bmp).build()
            return try {
                val res = landmarker.detectForVideo(mpImg, tMs) ?: return null
                val pose = res.landmarks().firstOrNull() ?: return null
                val n = pose.size
                val xs = FloatArray(n) { i -> pose[i].x() }
                val ys = FloatArray(n) { i -> pose[i].y() }

                // mediapipe pose index: 15 LWrist, 16 RWrist, 19 LIndex, 20 RIndex
                fun safe(i: Int) = if (i in 0 until n) i else -1
                val iLW = safe(15); val iRW = safe(16); val iLH = safe(19); val iRH = safe(20)

                val lWx = iLW.takeIf { it >= 0 }?.let { xs[it] }
                val lWy = iLW.takeIf { it >= 0 }?.let { ys[it] }
                val rWx = iRW.takeIf { it >= 0 }?.let { xs[it] }
                val rWy = iRW.takeIf { it >= 0 }?.let { ys[it] }
                val lHx = iLH.takeIf { it >= 0 }?.let { xs[it] }
                val lHy = iLH.takeIf { it >= 0 }?.let { ys[it] }
                val rHx = iRH.takeIf { it >= 0 }?.let { xs[it] }
                val rHy = iRH.takeIf { it >= 0 }?.let { ys[it] }

                val isMirrored =
                    runCatching { xs[11] > xs[12] }.getOrElse { false } // 11 LShoulder, 12 RShoulder

                Landmarks(xs, ys, lWx, lWy, rWx, rWy, lHx, lHy, rHx, rHy, isMirrored)
            } catch (_: Throwable) { null } finally { mpImg.close() }
        }

        override fun close() { runCatching { landmarker.close() } }
    }

    private class MoveNetDetector(
        private val ctx: Context,
        private val assetModelPath: String,
        private val tier: Tier
    ) : PoseDetector {
        private val interpreter: org.tensorflow.lite.Interpreter by lazy {
            org.tensorflow.lite.Interpreter(loadModelFile(ctx, assetModelPath))
        }
        private val inputSize = if (tier == Tier.LIGHT) 192 else 256
        private val imgBuffer = FloatArray(inputSize * inputSize * 3)

        override fun detect(bmp: Bitmap, tMs: Long): Landmarks? {
            val size = minOf(bmp.width, bmp.height)
            val left = (bmp.width - size) / 2
            val top  = (bmp.height - size) / 2
            val square = Bitmap.createBitmap(bmp, left, top, size, size)
            val input = Bitmap.createScaledBitmap(square, inputSize, inputSize, false)
            square.recycle()

            val px = IntArray(input.width * input.height)
            input.getPixels(px, 0, input.width, 0, 0, input.width, input.height)
            var o = 0
            for (p in px) {
                imgBuffer[o++] = ((p shr 16) and 0xFF) / 255f
                imgBuffer[o++] = ((p shr 8) and 0xFF) / 255f
                imgBuffer[o++] = (p and 0xFF) / 255f
            }
            input.recycle()

            // Output [1,1,17,3] (y,x,score)
            val out = Array(1) { Array(1) { Array(17) { FloatArray(3) } } }
            return try {
                interpreter.run(imgBuffer, out)
                val kps = out[0][0]
                val n = kps.size
                val xs = FloatArray(n) { i -> kps[i][1].coerceIn(0f,1f) }
                val ys = FloatArray(n) { i -> kps[i][0].coerceIn(0f,1f) }

                // COCO: 9 LShoulder,10 RShoulder, 9? (movenet variants differ) — 보편 매핑:
                // 5 LShoulder, 6 RShoulder, 9 LWrist, 10 RWrist (모델별 조금 다를 수 있음)
                val iLShoulder = 5; val iRShoulder = 6
                val iLWrist = 9;   val iRWrist = 10

                val isMirrored = xs[iLShoulder] > xs[iRShoulder]
                val lWx = xs[iLWrist]; val lWy = ys[iLWrist]
                val rWx = xs[iRWrist]; val rWy = ys[iRWrist]

                // MoveNet에는 hand index가 없음 → wrist로 대체
                Landmarks(xs, ys, lWx, lWy, rWx, rWy, lWx, lWy, rWx, rWy, isMirrored)
            } catch (_: Throwable) { null }
        }

        override fun close() { runCatching { interpreter.close() } }

        private fun loadModelFile(ctx: Context, assetPath: String): java.nio.MappedByteBuffer {
            val afd = ctx.assets.openFd(assetPath)
            java.io.FileInputStream(afd.fileDescriptor).channel.use { ch ->
                return ch.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.length)
            }
        }
    }
}
