package cc.ggrip.movenet.analysis

import cc.ggrip.movenet.pose.PoseFrame
import kotlin.math.abs
import kotlin.math.sqrt

enum class GolfSwingPhase {
    ADDRESS,        // 어드레스
    TAKEAWAY,       // 테이크어웨이
    BACKSWING,      // 백스윙
    BACKSWING_TOP,  // 백스윙 탑
    DOWNSWING,      // 다운스윙
    IMPACT,         // 임팩트
    FOLLOW_THROUGH, // 팔로우쓰루
    FINISH          // 피니시
}

data class SwingPhaseAnalysis(
    val phase: GolfSwingPhase,
    val confidence: Float,
    val wristSpeed: Float,
    val wristHeight: Float,
    val shoulderRotation: Float
)

class SwingStateAnalyzer {

    private var previousFrame: PoseFrame? = null
    private var frameHistory = mutableListOf<PoseFrame>()
    private val historySize = 10
    private val mediaPipeAnalyzer = MediaPipeSwingAnalyzer()

    // 조정 가능한 임계값들 (극단적으로 완화)
    private var addressSpeedThreshold = 0.01f  // 극단적 완화
    private var takeawaySpeedThreshold = 0.02f  // 극단적 완화
    private var backswingSpeedThreshold = 0.05f  // 극단적 완화
    private var downswingSpeedThreshold = 0.08f  // 극단적 완화
    private var impactSpeedThreshold = 0.10f  // 극단적 완화
    private var finishSpeedThreshold = 0.05f  // 극단적 완화

    // 좌/우 손잡이 감지
    private var isRightHanded: Boolean? = null

    // 이전 상태 추적 (순차적 진행을 위해)
    private var previousPhase: GolfSwingPhase = GolfSwingPhase.ADDRESS
    private var phaseCounter = 0  // 같은 상태가 연속으로 감지된 횟수

    // MoveNet COCO-17 키포인트 인덱스
    private val NOSE = 0
    private val LEFT_EYE = 1
    private val RIGHT_EYE = 2
    private val LEFT_EAR = 3
    private val RIGHT_EAR = 4
    private val LEFT_SHOULDER = 5
    private val RIGHT_SHOULDER = 6
    private val LEFT_ELBOW = 7
    private val RIGHT_ELBOW = 8
    private val LEFT_WRIST = 9
    private val RIGHT_WRIST = 10
    private val LEFT_HIP = 11
    private val RIGHT_HIP = 12
    private val LEFT_KNEE = 13
    private val RIGHT_KNEE = 14
    private val LEFT_ANKLE = 15
    private val RIGHT_ANKLE = 16

    fun analyzeSwingState(frame: PoseFrame): SwingPhaseAnalysis {
        frameHistory.add(frame)
        if (frameHistory.size > historySize) {
            frameHistory.removeAt(0)
        }

        // 좌/우 손잡이 감지 (충분한 프레임이 모이면 실행)
        if (isRightHanded == null && frameHistory.size >= 5) {
            detectHandedness()
        }

        // 키포인트 수로 MoveNet(17개, 34 좌표) vs MediaPipe(33개, 66 좌표) 구분
        val analysis = when {
            frameHistory.size < 3 -> SwingPhaseAnalysis(GolfSwingPhase.ADDRESS, 0.5f, 0f, 0f, 0f)
            frame.screen2d.size >= 66 -> mediaPipeAnalyzer.analyzeSwingState(frame)  // MediaPipe
            else -> detectSwingState(frame)  // MoveNet
        }

        previousFrame = frame
        return analysis
    }

    private fun detectSwingState(frame: PoseFrame): SwingPhaseAnalysis {
        val keypoints = frame.screen2d
        if (keypoints.size < 34) return SwingPhaseAnalysis(GolfSwingPhase.ADDRESS, 0f, 0f, 0f, 0f)

        // 디버그 모드
        val DEBUG = true

        // 주요 키포인트 좌표 추출
        val leftWrist = getPoint(keypoints, LEFT_WRIST)
        val rightWrist = getPoint(keypoints, RIGHT_WRIST)
        val leftShoulder = getPoint(keypoints, LEFT_SHOULDER)
        val rightShoulder = getPoint(keypoints, RIGHT_SHOULDER)
        val leftHip = getPoint(keypoints, LEFT_HIP)
        val rightHip = getPoint(keypoints, RIGHT_HIP)

        // 오른손잡이 골퍼 기준 (왼손이 위쪽 손)
        val topWrist = if (leftWrist.y < rightWrist.y) leftWrist else rightWrist
        val bottomWrist = if (leftWrist.y < rightWrist.y) rightWrist else leftWrist

        // 1. 손목 속도 계산
        val wristSpeed = calculateWristSpeed(topWrist)

        // 2. 손목 높이 (MediaPipe와 동일하게 정규화)
        val shoulderY = (leftShoulder.y + rightShoulder.y) / 2f
        val hipY = (leftHip.y + rightHip.y) / 2f
        val bodyHeight = abs(hipY - shoulderY).coerceAtLeast(0.01f)  // 0 방지
        val wristHeight = (shoulderY - topWrist.y) / bodyHeight  // 정규화된 높이

        // 3. 어깨 회전 각도
        val shoulderRotation = calculateShoulderRotation(leftShoulder, rightShoulder)

        // 4. 상태 판별
        val state = determineStateWithProgression(wristSpeed, wristHeight, shoulderRotation, topWrist, bottomWrist, leftShoulder, rightShoulder)
        val confidence = calculateConfidence(state, wristSpeed, wristHeight, shoulderRotation)

        if (DEBUG) {
            android.util.Log.d("SwingState", "MoveNet - Speed: %.4f, Height: %.3f, Rotation: %.3f, State: %s -> %s".format(
                wristSpeed, wristHeight, shoulderRotation, previousPhase.name, state.name
            ))
        }

        return SwingPhaseAnalysis(state, confidence, wristSpeed, wristHeight, shoulderRotation)
    }

    private fun getPoint(keypoints: FloatArray, index: Int): Point2D {
        val x = keypoints[index * 2]
        val y = keypoints[index * 2 + 1]
        return Point2D(x, y)
    }

    private fun calculateWristSpeed(wrist: Point2D): Float {
        // 히스토리를 활용한 스무딩된 속도 계산
        if (frameHistory.size < 2) return 0f

        var totalSpeed = 0f
        var validFrames = 0

        for (i in frameHistory.size - 1 downTo maxOf(0, frameHistory.size - 3)) {
            if (i > 0) {
                val current = frameHistory[i]
                val previous = frameHistory[i - 1]

                if (current.screen2d.size >= 34 && previous.screen2d.size >= 34) {
                    val currWrist = getPoint(current.screen2d, if (isRightHanded == true) RIGHT_WRIST else LEFT_WRIST)
                    val prevWrist = getPoint(previous.screen2d, if (isRightHanded == true) RIGHT_WRIST else LEFT_WRIST)

                    val dx = currWrist.x - prevWrist.x
                    val dy = currWrist.y - prevWrist.y
                    totalSpeed += sqrt(dx * dx + dy * dy)
                    validFrames++
                }
            }
        }

        return if (validFrames > 0) totalSpeed / validFrames else 0f
    }

    private fun calculateShoulderRotation(leftShoulder: Point2D, rightShoulder: Point2D): Float {
        return rightShoulder.x - leftShoulder.x  // 양수면 오른쪽 어깨가 더 앞으로
    }

    private fun determineStateWithProgression(
        wristSpeed: Float,
        wristHeight: Float,
        shoulderRotation: Float,
        topWrist: Point2D,
        bottomWrist: Point2D,
        leftShoulder: Point2D,
        rightShoulder: Point2D
    ): GolfSwingPhase {

        val clubX = (topWrist.x + bottomWrist.x) / 2f
        val bodyCenter = (leftShoulder.x + rightShoulder.x) / 2f

        // 단순화된 상태 판별 - 속도와 높이만 사용
        val candidateState = when (previousPhase) {
            GolfSwingPhase.ADDRESS -> {
                // 조금만 움직이면 TAKEAWAY로 (정규화된 Height 기준)
                if (wristSpeed > 0.01f || wristHeight > -0.9f) {  // MediaPipe와 동일
                    GolfSwingPhase.TAKEAWAY
                } else {
                    GolfSwingPhase.ADDRESS
                }
            }

            GolfSwingPhase.TAKEAWAY -> {
                // 높이가 올라가면 BACKSWING으로
                if (wristHeight > -0.5f) {  // 정규화된 값
                    GolfSwingPhase.BACKSWING
                } else if (wristSpeed < 0.005f) {
                    GolfSwingPhase.ADDRESS  // 멈춤면 어드레스로
                } else {
                    GolfSwingPhase.TAKEAWAY
                }
            }

            GolfSwingPhase.BACKSWING -> {
                // 높이가 충분히 높고 속도 감소면 TOP으로
                if (wristHeight > 0.1f && wristSpeed < 0.05f) {  // 정규화된 값
                    GolfSwingPhase.BACKSWING_TOP
                } else if (wristHeight < -0.7f) {
                    GolfSwingPhase.TAKEAWAY  // 내려오면 테이크어웨이로
                } else {
                    GolfSwingPhase.BACKSWING
                }
            }

            GolfSwingPhase.BACKSWING_TOP -> {
                // 속도가 증가하면 DOWNSWING으로
                if (wristSpeed > 0.05f) {
                    GolfSwingPhase.DOWNSWING
                } else if (wristHeight < 0.05f) {  // 정규화된 값
                    GolfSwingPhase.BACKSWING  // 높이가 낮아지면 백스윙으로
                } else {
                    GolfSwingPhase.BACKSWING_TOP
                }
            }

            GolfSwingPhase.DOWNSWING -> {
                // 속도가 최고고 높이가 낮으면 IMPACT로
                if (wristSpeed > 0.08f && wristHeight < -0.3f) {  // 정규화된 값 (음수 영역)
                    GolfSwingPhase.IMPACT
                } else if (wristSpeed < 0.03f) {
                    GolfSwingPhase.BACKSWING_TOP  // 속도 감소면 탑으로
                } else {
                    GolfSwingPhase.DOWNSWING
                }
            }

            GolfSwingPhase.IMPACT -> {
                // 높이가 다시 올라가면 FOLLOW_THROUGH로
                if (wristHeight > -0.5f) {  // 정규화된 값 (음수에서 올라옴)
                    GolfSwingPhase.FOLLOW_THROUGH
                } else {
                    GolfSwingPhase.IMPACT
                }
            }

            GolfSwingPhase.FOLLOW_THROUGH -> {
                // 속도가 충분히 감소하고 높이가 높으면 FINISH로
                if (wristSpeed < 0.05f && wristHeight > 0.1f) {  // 정규화된 값
                    GolfSwingPhase.FINISH
                } else if (wristHeight < -0.7f) {
                    GolfSwingPhase.IMPACT  // 낮아지면 임팩트로
                } else {
                    GolfSwingPhase.FOLLOW_THROUGH
                }
            }

            GolfSwingPhase.FINISH -> {
                // 속도가 낮고 높이가 낮아지면 새 스윙 시작
                if (wristSpeed < 0.03f && wristHeight < -0.8f) {  // 정규화된 값 (엉덩이 근처)
                    GolfSwingPhase.ADDRESS
                } else if (wristSpeed < 0.02f && wristHeight < -0.5f) {  // 중간 단계
                    GolfSwingPhase.ADDRESS
                } else {
                    GolfSwingPhase.FINISH
                }
            }
        }

        // 상태 변화 감지 및 업데이트
        if (candidateState == previousPhase) {
            phaseCounter++
        } else {
            phaseCounter = 0
            previousPhase = candidateState
        }

        return candidateState
    }

    private fun determineState(
        wristSpeed: Float,
        wristHeight: Float,
        shoulderRotation: Float,
        topWrist: Point2D,
        bottomWrist: Point2D,
        leftShoulder: Point2D,
        rightShoulder: Point2D
    ): GolfSwingPhase {
        // 기존 로직 보존 (필요시 사용)
        return GolfSwingPhase.ADDRESS
    }

    private fun calculateConfidence(
        state: GolfSwingPhase,
        wristSpeed: Float,
        wristHeight: Float,
        shoulderRotation: Float
    ): Float {
        // 상태별 특징값이 예상 범위에 얼마나 부합하는지 계산
        return when (state) {
            GolfSwingPhase.ADDRESS -> if (wristSpeed < addressSpeedThreshold) 0.9f else 0.6f
            GolfSwingPhase.TAKEAWAY -> if (wristSpeed in addressSpeedThreshold..takeawaySpeedThreshold) 0.8f else 0.6f
            GolfSwingPhase.BACKSWING -> if (wristSpeed > 0.03f && wristHeight > 0.15f) 0.8f else 0.6f
            GolfSwingPhase.BACKSWING_TOP -> if (wristHeight > 0.25f) 0.9f else 0.7f
            GolfSwingPhase.DOWNSWING -> if (wristSpeed > downswingSpeedThreshold) 0.8f else 0.6f
            GolfSwingPhase.IMPACT -> if (wristSpeed > impactSpeedThreshold) 0.9f else 0.7f
            GolfSwingPhase.FOLLOW_THROUGH -> if (shoulderRotation > 0.05f) 0.8f else 0.6f
            GolfSwingPhase.FINISH -> if (wristSpeed < finishSpeedThreshold && wristHeight > 0.2f) 0.9f else 0.7f
        }
    }

    private fun detectHandedness() {
        // 여러 프레임의 손목 위치를 분석하여 좌/우 손잡이 판단
        var leftDominantCount = 0
        var rightDominantCount = 0

        for (frame in frameHistory.takeLast(5)) {
            val keypoints = frame.screen2d
            if (keypoints.size >= 34) {
                val leftWrist = getPoint(keypoints, LEFT_WRIST)
                val rightWrist = getPoint(keypoints, RIGHT_WRIST)
                val leftShoulder = getPoint(keypoints, LEFT_SHOULDER)
                val rightShoulder = getPoint(keypoints, RIGHT_SHOULDER)

                val shoulderCenter = (leftShoulder.x + rightShoulder.x) / 2f

                // 어느 손목이 몸 중심에서 더 멀리 있는지 확인 (리드하는 손)
                val leftDistance = abs(leftWrist.x - shoulderCenter)
                val rightDistance = abs(rightWrist.x - shoulderCenter)

                // 일반적으로 오른손잡이는 왼손이 리드, 왼손잡이는 오른손이 리드
                if (leftDistance > rightDistance * 1.1f) {
                    rightDominantCount++  // 왼손이 리드 = 오른손잡이
                } else if (rightDistance > leftDistance * 1.1f) {
                    leftDominantCount++   // 오른손이 리드 = 왼손잡이
                }
            }
        }
        isRightHanded = rightDominantCount >= leftDominantCount
    }

    fun adjustThresholds(
        addressSpeed: Float? = null,
        takeawaySpeed: Float? = null,
        backswingSpeed: Float? = null,
        downswingSpeed: Float? = null,
        impactSpeed: Float? = null,
        finishSpeed: Float? = null
    ) {
        addressSpeed?.let { addressSpeedThreshold = it }
        takeawaySpeed?.let { takeawaySpeedThreshold = it }
        backswingSpeed?.let { backswingSpeedThreshold = it }
        downswingSpeed?.let { downswingSpeedThreshold = it }
        impactSpeed?.let { impactSpeedThreshold = it }
        finishSpeed?.let { finishSpeedThreshold = it }
    }

    data class Point2D(val x: Float, val y: Float)
}