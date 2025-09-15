// PoseTypes.kt
package cc.ggrip.movenet.pose

data class PoseFrame(
    val tMillis: Long,
    val world: FloatArray,      // 33*3 or empty
    val screen2d: FloatArray,   // Nx2 normalized, N=17(ML) or 33(MP)
    val visibility: FloatArray? = null,

    val srcTsMs: Long = -1L,         // camera frame timestamp (ms)
    val algoDoneTsMs: Long = -1L,    // end of inference (ms)

    // ↓ 벤치마크용 메타
    val engineName: String = "",
    val modelTier: String = ""       // LIGHT / MID / HEAVY
)

object PoseConst {
    const val MOVENET_KP = 17
    const val MEDIAPIPE_KP = 33
}
