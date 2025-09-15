// MainActivity.kt
package cc.ggrip.movenet

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.commit
import cc.ggrip.movenet.ui.RealtimeDotsFragment

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        if (savedInstanceState == null) {
            supportFragmentManager.commit {
                replace(R.id.fragmentContainer, RealtimeDotsFragment.newInstance(30.0))
            }
        }
    }
}
