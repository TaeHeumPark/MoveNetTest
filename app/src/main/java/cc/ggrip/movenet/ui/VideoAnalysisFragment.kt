package cc.ggrip.movenet.ui

import android.content.ContentResolver
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.ProgressBar
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import cc.ggrip.movenet.R
import cc.ggrip.movenet.analysis.VideoAnalyzer
import cc.ggrip.movenet.analysis.VideoAnalysisResult
import cc.ggrip.movenet.bench.Engine
import cc.ggrip.movenet.bench.Tier
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class VideoAnalysisFragment : Fragment() {

    companion object {
        fun newInstance() = VideoAnalysisFragment()
        private const val STATE_VIDEO_URI = "video_uri"
        private const val STATE_VIDEO_NAME = "video_name"
    }

    private data class ModelOption(val title: String, val engine: Engine, val tier: Tier)

    private val modelOptions = listOf(
        ModelOption("MoveNet - lightning", Engine.MOVENET, Tier.LIGHT),
        ModelOption("MoveNet - thunder", Engine.MOVENET, Tier.MID),
        ModelOption("MediaPipe - lite", Engine.MEDIAPIPE, Tier.LIGHT),
        ModelOption("MediaPipe - full", Engine.MEDIAPIPE, Tier.MID),
        ModelOption("MediaPipe - heavy", Engine.MEDIAPIPE, Tier.HEAVY)
    )

    private val openVideoLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            requireContext().contentResolver.takePersistableUriPermission(
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION
            )
            selectedVideoUri = uri
            selectedVideoName = displayName(requireContext(), uri)
            selectedVideoText?.text = selectedVideoName ?: uri.toString()
        }
    }

    private var selectedVideoUri: Uri? = null
    private var selectedVideoName: String? = null

    private var analyzeJob: Job? = null

    private var selectButton: Button? = null
    private var analyzeButton: Button? = null
    private var selectedVideoText: TextView? = null
    private var progressBar: ProgressBar? = null
    private var progressText: TextView? = null
    private var segmentBar: SegmentBarView? = null
    private var segmentSummary: TextView? = null
    private var modelSpinner: Spinner? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (savedInstanceState != null) {
            selectedVideoUri = savedInstanceState.getParcelable(STATE_VIDEO_URI)
            selectedVideoName = savedInstanceState.getString(STATE_VIDEO_NAME)
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        return inflater.inflate(R.layout.fragment_video_analysis, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        selectButton = view.findViewById<Button>(R.id.selectVideoButton).also { button ->
            button.setOnClickListener { openVideoPicker() }
        }
        analyzeButton = view.findViewById<Button>(R.id.btnAnalyze).also { button ->
            button.setOnClickListener { startAnalysis() }
        }
        selectedVideoText = view.findViewById<TextView>(R.id.selectedVideoText).also { textView ->
            textView.text = selectedVideoName ?: getString(R.string.video_select_placeholder)
        }
        progressBar = view.findViewById(R.id.analysisProgress)
        progressText = view.findViewById(R.id.progressText)
        segmentBar = view.findViewById(R.id.segmentBar)
        modelSpinner = view.findViewById<Spinner>(R.id.modelSpinner).also { spinner ->
            val adapter = ArrayAdapter(
                requireContext(),
                android.R.layout.simple_spinner_item,
                modelOptions.map { it.title }
            )
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            spinner.adapter = adapter
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putParcelable(STATE_VIDEO_URI, selectedVideoUri)
        outState.putString(STATE_VIDEO_NAME, selectedVideoName)
    }

    override fun onDestroyView() {
        analyzeJob?.cancel()
        selectButton = null
        analyzeButton = null
        selectedVideoText = null
        progressBar = null
        progressText = null
        segmentBar = null
        segmentSummary = null
        modelSpinner = null
        super.onDestroyView()
    }

    private fun openVideoPicker() {
        openVideoLauncher.launch(arrayOf("video/*"))
    }

    private fun startAnalysis() {
        val uri = selectedVideoUri
        if (uri == null) {
            Toast.makeText(requireContext(), R.string.video_error_select_first, Toast.LENGTH_SHORT).show()
            return
        }
        val spinner = modelSpinner ?: return
        val option = modelOptions[spinner.selectedItemPosition]

        analyzeJob?.cancel()
        val progressBar = progressBar ?: return
        val progressText = progressText ?: return
        val analyzeButton = analyzeButton ?: return

        analyzeButton.isEnabled = false
        progressBar.visibility = View.VISIBLE
        progressBar.isIndeterminate = true
        progressText.visibility = View.VISIBLE
        progressText.text = ""

        val videoAnalyzer = VideoAnalyzer(requireContext())

        analyzeJob = viewLifecycleOwner.lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.Default) {
                    videoAnalyzer.analyze(uri, option.engine, option.tier) { current, total ->
                        val safeTotal = if (total <= 0) 1 else total
                        progressBar.post {
                            if (progressBar.isIndeterminate) {
                                progressBar.isIndeterminate = false
                                progressBar.max = safeTotal
                            }
                            progressBar.progress = current.coerceIn(0, safeTotal)
                            progressText.text = getString(R.string.video_progress_format, current, safeTotal)
                        }
                    }
                }
                applyResult(result)
            } catch (ce: CancellationException) {
                // cancelled
            } catch (t: Throwable) {
                Toast.makeText(requireContext(), R.string.video_error_generic, Toast.LENGTH_SHORT).show()
                Log.w("VideoAnalysis", "analysis failed", t)
            } finally {
                analyzeButton.isEnabled = true
                progressBar.visibility = View.GONE
                progressText.visibility = View.GONE
            }
        }
    }
    private fun applyResult(result: VideoAnalysisResult) {
        val bar = segmentBar ?: return
        val summaryView = segmentSummary ?: return
        bar.setSegments(result.durationMs, result.segments)
        if (result.segments.isEmpty()) {
            summaryView.text = getString(R.string.video_segment_none)
        } else {
            val segment = result.segments.first()
            val durationSec = (segment.endTimeMs - segment.startTimeMs) / 1000f
            summaryView.text = getString(
                R.string.video_segment_summary,
                segment.startTimeMs / 1000f,
                segment.endTimeMs / 1000f,
                durationSec
            )
        }
    }

    private fun displayName(context: Context, uri: Uri): String? {
        val resolver: ContentResolver = context.contentResolver
        resolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val idx = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                if (idx >= 0) {
                    return cursor.getString(idx)
                }
            }
        }
        return null
    }
}

