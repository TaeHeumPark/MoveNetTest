// PoseTypes.kt
package cc.ggrip.movenet.pose

data class PoseFrame(
    val tMillis: Long,
    val world: FloatArray,      // MoveNet은 3D 없음: 빈 배열 사용
    val screen2d: FloatArray,   // 33*2 (MoveNet은 17*2만 사용)
    val visibility: FloatArray? = null,

    // 지연 계측용 타임스탬프
    val srcTsMs: Long = -1L,        // 카메라 프레임 원본 시각
    val algoDoneTsMs: Long = -1L    // 추론 완료 시각 (onResults/processor.run 이후)
)

object PoseConst {
    const val NUM_LM = 33
    const val NOSE = 0
    const val LEFT_SHOULDER = 11
    const val RIGHT_SHOULDER = 12
    const val LEFT_ELBOW = 13
    const val RIGHT_ELBOW = 14
    const val LEFT_WRIST = 15
    const val RIGHT_WRIST = 16
    const val LEFT_HIP = 23
    const val RIGHT_HIP = 24
    const val LEFT_KNEE = 25
    const val RIGHT_KNEE = 26
    const val LEFT_ANKLE = 27
    const val RIGHT_ANKLE = 28
}
