// FpsGovernor.kt
package cc.ggrip.movenet.util

class FpsGovernor(targetFps: Double) {
    @Volatile private var intervalNs: Long = (1_000_000_000.0 / targetFps).toLong()
    @Volatile private var lastAcceptedNs: Long = 0L

    fun shouldAccept(frameTsNs: Long): Boolean {
        if (lastAcceptedNs == 0L || frameTsNs - lastAcceptedNs >= intervalNs) {
            lastAcceptedNs = frameTsNs
            return true
        }
        return false
    }

    fun setTargetFps(targetFps: Double) {
        intervalNs = (1_000_000_000.0 / targetFps).toLong()
    }
}
