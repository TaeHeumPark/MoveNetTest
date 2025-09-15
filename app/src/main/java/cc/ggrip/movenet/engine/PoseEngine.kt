// PoseEngine.kt
package cc.ggrip.movenet.engine

import androidx.camera.core.ImageProxy
import cc.ggrip.movenet.pose.PoseFrame

interface PoseEngine {
    fun process(imageProxy: ImageProxy)
    fun close()
    val isGpuEnabled: Boolean
    val modelName: String
}