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

        // 2. 손목 높이 (어깨 대비)
        val shoulderY = (leftShoulder.y + rightShoulder.y) / 2f
        val wristHeight = shoulderY - topWrist.y  // 양수면 어깨보다 위

        // 3. 어깨 회전 각도
        val shoulderRotation = calculateShoulderRotation(leftShoulder, rightShoulder)

        // 4. 상태 판별
        val state = determineState(wristSpeed, wristHeight, shoulderRotation, topWrist, bottomWrist, leftShoulder, rightShoulder)
        val confidence = calculateConfidence(state, wristSpeed, wristHeight, shoulderRotation)

        return SwingPhaseAnalysis(state, confidence, wristSpeed, wristHeight, shoulderRotation)
    }

    private fun getPoint(keypoints: FloatArray, index: Int): Point2D {
        val x = keypoints[index * 2]
        val y = keypoints[index * 2 + 1]
        return Point2D(x, y)
    }

    private fun calculateWristSpeed(wrist: Point2D): Float {
        val prev = previousFrame?.let {
            val kp = it.screen2d
            if (kp.size >= 34) getPoint(kp, LEFT_WRIST) else null
        }

        return if (prev != null) {
            val dx = wrist.x - prev.x
            val dy = wrist.y - prev.y
            sqrt(dx * dx + dy * dy)
        } else 0f
    }

    private fun calculateShoulderRotation(leftShoulder: Point2D, rightShoulder: Point2D): Float {
        return rightShoulder.x - leftShoulder.x  // 양수면 오른쪽 어깨가 더 앞으로
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

        // 클럽 위치 (두 손목 사이의 중점으로 추정)
        val clubX = (topWrist.x + bottomWrist.x) / 2f
        val clubY = (topWrist.y + bottomWrist.y) / 2f
        val bodyCenter = (leftShoulder.x + rightShoulder.x) / 2f

        return when {
            // 1. ADDRESS: 낮은 속도, 클럽이 몸 앞쪽, 손목이 어깨 아래
            wristSpeed < 0.02f && wristHeight < 0.1f && abs(clubX - bodyCenter) < 0.15f -> {
                GolfSwingPhase.ADDRESS
            }

            // 2. TAKEAWAY: 낮은-중간 속도, 클럽이 몸에서 멀어짐, 손목이 올라가기 시작
            wristSpeed < 0.05f && wristHeight > 0.05f && (clubX - bodyCenter) > 0.1f -> {
                GolfSwingPhase.TAKEAWAY
            }

            // 3. BACKSWING: 중간 속도, 손목이 어깨보다 높이, 클럽이 뒤쪽으로
            wristSpeed > 0.03f && wristSpeed < 0.15f && wristHeight > 0.15f && shoulderRotation < -0.05f -> {
                GolfSwingPhase.BACKSWING
            }

            // 4. BACKSWING_TOP: 속도가 감소하기 시작, 손목이 가장 높은 위치
            wristSpeed < 0.08f && wristHeight > 0.25f && shoulderRotation < -0.1f -> {
                GolfSwingPhase.BACKSWING_TOP
            }

            // 5. DOWNSWING: 속도 증가, 손목이 내려오기 시작, 어깨 회전 시작
            wristSpeed > 0.1f && wristHeight > 0.1f && shoulderRotation > -0.05f -> {
                GolfSwingPhase.DOWNSWING
            }

            // 6. IMPACT: 최고 속도, 손목이 어깨 근처 높이, 몸 앞쪽
            wristSpeed > 0.2f && abs(wristHeight) < 0.1f && abs(clubX - bodyCenter) < 0.2f -> {
                GolfSwingPhase.IMPACT
            }

            // 7. FOLLOW_THROUGH: 높은 속도 유지, 손목이 다시 올라감, 어깨 회전 계속
            wristSpeed > 0.1f && wristHeight > 0.1f && shoulderRotation > 0.05f -> {
                GolfSwingPhase.FOLLOW_THROUGH
            }

            // 8. FINISH: 속도 감소, 손목이 높은 위치에서 정지
            wristSpeed < 0.05f && wristHeight > 0.2f && shoulderRotation > 0.1f -> {
                GolfSwingPhase.FINISH
            }

            else -> GolfSwingPhase.ADDRESS  // 기본값
        }
    }

    private fun calculateConfidence(
        state: GolfSwingPhase,
        wristSpeed: Float,
        wristHeight: Float,
        shoulderRotation: Float
    ): Float {
        // 상태별 특징값이 예상 범위에 얼마나 부합하는지 계산
        return when (state) {
            GolfSwingPhase.ADDRESS -> if (wristSpeed < 0.02f) 0.9f else 0.6f
            GolfSwingPhase.TAKEAWAY -> if (wristSpeed in 0.02f..0.05f) 0.8f else 0.6f
            GolfSwingPhase.BACKSWING -> if (wristSpeed > 0.03f && wristHeight > 0.15f) 0.8f else 0.6f
            GolfSwingPhase.BACKSWING_TOP -> if (wristHeight > 0.25f) 0.9f else 0.7f
            GolfSwingPhase.DOWNSWING -> if (wristSpeed > 0.1f) 0.8f else 0.6f
            GolfSwingPhase.IMPACT -> if (wristSpeed > 0.2f) 0.9f else 0.7f
            GolfSwingPhase.FOLLOW_THROUGH -> if (shoulderRotation > 0.05f) 0.8f else 0.6f
            GolfSwingPhase.FINISH -> if (wristSpeed < 0.05f && wristHeight > 0.2f) 0.9f else 0.7f
        }
    }

    data class Point2D(val x: Float, val y: Float)
}