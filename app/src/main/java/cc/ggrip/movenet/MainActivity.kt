// MainActivity.kt
package cc.ggrip.movenet

import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.commit
import cc.ggrip.movenet.bench.Engine
import cc.ggrip.movenet.bench.Tier
import cc.ggrip.movenet.ui.MediaPipeRealtimeFragment
import cc.ggrip.movenet.ui.RealtimeDotsFragment
import com.google.android.material.dialog.MaterialAlertDialogBuilder

class MainActivity : AppCompatActivity() {

    private data class ModelItem(val title: String, val engine: Engine, val tier: Tier)

    private val modelItems = listOf(
        ModelItem("MoveNet • lightning", Engine.MOVENET, Tier.LIGHT),
        ModelItem("MoveNet • thunder",   Engine.MOVENET, Tier.MID),   // thunder
        ModelItem("MediaPipe • lite",    Engine.MEDIAPIPE, Tier.LIGHT),
        ModelItem("MediaPipe • full",    Engine.MEDIAPIPE, Tier.MID),
        ModelItem("MediaPipe • heavy",   Engine.MEDIAPIPE, Tier.HEAVY)
    )

    private var selectedIndex = 1 // 기본: MoveNet thunder

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        setSupportActionBar(findViewById(R.id.toolbar))

        if (savedInstanceState != null) {
            selectedIndex = savedInstanceState.getInt("sel", selectedIndex)
        }

        showSelectedModel()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putInt("sel", selectedIndex)
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.menu_main, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.action_model -> {
                val names = modelItems.map { it.title }.toTypedArray()
                MaterialAlertDialogBuilder(this)
                    .setTitle("모델 선택")
                    .setSingleChoiceItems(names, selectedIndex) { dialog, which ->
                        selectedIndex = which
                        dialog.dismiss()
                        showSelectedModel()
                    }
                    .setNegativeButton(android.R.string.cancel, null)
                    .show()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    private fun showSelectedModel() {
        val item = modelItems[selectedIndex]
        supportActionBar?.title = item.title

        supportFragmentManager.commit {
            val frag = when (item.engine) {
                Engine.MOVENET   -> RealtimeDotsFragment.newInstance(30.0, item.tier)
                Engine.MEDIAPIPE -> MediaPipeRealtimeFragment.newInstance(30.0, item.tier)
            }
            replace(R.id.fragmentContainer, frag)
        }
    }
}
