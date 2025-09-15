// BenchmarkConfig.kt
package cc.ggrip.movenet.bench

enum class Engine { MOVENET, MEDIAPIPE }
enum class Tier { LIGHT, MID, HEAVY }

data class BenchChoice(
    val engine: Engine,
    val tier: Tier
) {
    val title: String get() = "${engine.name} • ${tier.name}"
}

// 에셋 경로 매핑
object ModelAssets {
    fun movenetPath(tier: Tier): String = when (tier) {
        Tier.LIGHT -> "models/movenet_lightning_fp16.tflite"
        Tier.MID   -> "models/movenet_thunder_fp16.tflite"
        Tier.HEAVY -> "models/movenet_thunder_fp16.tflite" // 단일포즈 최상위로 Thunder 고정
    }
    fun mpTaskPath(tier: Tier): String = when (tier) {
        Tier.LIGHT -> "models/pose_landmarker_lite.task"
        Tier.MID   -> "models/pose_landmarker_full.task"
        Tier.HEAVY -> "models/pose_landmarker_heavy.task"
    }
}
