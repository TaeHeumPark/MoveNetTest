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
        try {
            val compatCls = Class.forName("org.tensorflow.lite.gpu.CompatibilityList")
            val compat = compatCls.getConstructor().newInstance()
            val isSupported = compatCls.getMethod("isDelegateSupportedOnThisDevice").invoke(compat) as Boolean
            android.util.Log.i("MoveNet", "GPU delegate supported on this device? $isSupported")
        } catch (t: Throwable) {
            android.util.Log.w("MoveNet", "GPU libs not on classpath (add dependency).", t)
        }

        if (savedInstanceState == null) {
            supportFragmentManager.commit {
                replace(R.id.fragmentContainer, RealtimeDotsFragment.newInstance(30.0))
            }
        }
    }
}
