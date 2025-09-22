package cc.ggrip.movenet.analysis

import cc.ggrip.movenet.pose.PoseFrame
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.math.atan2
import kotlin.math.PI

class MediaPipeSwingAnalyzer {

    private var previousFrame: PoseFrame? = null
    private var frameHistory = mutableListOf<PoseFrame>()
    private val historySize = 15

    // MediaPipe Pose 33개 키포인트 인덱스
    // 0-10: 얼굴 랜드마크
    private val NOSE = 0
    private val LEFT_EYE_INNER = 1
    private val LEFT_EYE = 2
    private val LEFT_EYE_OUTER = 3
    private val RIGHT_EYE_INNER = 4
    private val RIGHT_EYE = 5
    private val RIGHT_EYE_OUTER = 6
    private val LEFT_EAR = 7
    private val RIGHT_EAR = 8
    private val MOUTH_LEFT = 9
    private val MOUTH_RIGHT = 10

    // 11-22: 상체 주요 관절
    private val LEFT_SHOULDER = 11
    private val RIGHT_SHOULDER = 12
    private val LEFT_ELBOW = 13
    private val RIGHT_ELBOW = 14
    private val LEFT_WRIST = 15
    private val RIGHT_WRIST = 16
    private val LEFT_PINKY = 17
    private val RIGHT_PINKY = 18
    private val LEFT_INDEX = 19
    private val RIGHT_INDEX = 20
    private val LEFT_THUMB = 21
    private val RIGHT_THUMB = 22

    // 23-32: 하체 관절
    private val LEFT_HIP = 23
    private val RIGHT_HIP = 24
    private val LEFT_KNEE = 25
    private val RIGHT_KNEE = 26
    private val LEFT_ANKLE = 27
    private val RIGHT_ANKLE = 28
    private val LEFT_HEEL = 29
    private val RIGHT_HEEL = 30
    private val LEFT_FOOT_INDEX = 31
    private val RIGHT_FOOT_INDEX = 32

    fun analyzeSwingState(frame: PoseFrame): SwingPhaseAnalysis {
        frameHistory.add(frame)
        if (frameHistory.size > historySize) {
            frameHistory.removeAt(0)
        }

        val analysis = when {
            frameHistory.size < 3 -> SwingPhaseAnalysis(GolfSwingPhase.ADDRESS, 0.5f, 0f, 0f, 0f)
            else -> detectSwingStateMediaPipe(frame)
        }

        previousFrame = frame
        return analysis
    }

    private fun detectSwingStateMediaPipe(frame: PoseFrame): SwingPhaseAnalysis {
        val keypoints = frame.screen2d
        if (keypoints.size < 66) return SwingPhaseAnalysis(GolfSwingPhase.ADDRESS, 0f, 0f, 0f, 0f)

        // 주요 키포인트 좌표 추출
        val leftWrist = getPoint(keypoints, LEFT_WRIST)
        val rightWrist = getPoint(keypoints, RIGHT_WRIST)
        val leftShoulder = getPoint(keypoints, LEFT_SHOULDER)
        val rightShoulder = getPoint(keypoints, RIGHT_SHOULDER)
        val leftElbow = getPoint(keypoints, LEFT_ELBOW)
        val rightElbow = getPoint(keypoints, RIGHT_ELBOW)
        val leftHip = getPoint(keypoints, LEFT_HIP)
        val rightHip = getPoint(keypoints, RIGHT_HIP)

        // 손가락 키포인트 활용 (MediaPipe 고유)
        val leftIndex = getPoint(keypoints, LEFT_INDEX)
        val rightIndex = getPoint(keypoints, RIGHT_INDEX)
        val leftThumb = getPoint(keypoints, LEFT_THUMB)
        val rightThumb = getPoint(keypoints, RIGHT_THUMB)

        // 1. 고급 손목 속도 계산 (손가락 포함)
        val wristSpeed = calculateAdvancedWristSpeed(leftWrist, rightWrist, leftIndex, rightIndex)

        // 2. 정밀한 손목 높이 (어깨-엉덩이 기준)
        val shoulderY = (leftShoulder.y + rightShoulder.y) / 2f
        val hipY = (leftHip.y + rightHip.y) / 2f
        val bodyHeight = hipY - shoulderY
        val wristY = minOf(leftWrist.y, rightWrist.y)  // 더 높은 손목
        val wristHeight = (shoulderY - wristY) / bodyHeight  // 정규화된 높이

        // 3. 정밀한 어깨-엉덩이 회전각
        val shoulderRotation = calculateBodyRotation(leftShoulder, rightShoulder, leftHip, rightHip)

        // 4. 팔꿈치 각도 계산 (MediaPipe 추가 지표)
        val leftElbowAngle = calculateElbowAngle(leftShoulder, leftElbow, leftWrist)
        val rightElbowAngle = calculateElbowAngle(rightShoulder, rightElbow, rightWrist)
        val elbowExtension = minOf(leftElbowAngle, rightElbowAngle)  // 더 펼쳐진 팔

        // 5. 클럽 그립 안정성 (손가락 위치 기반)
        val gripStability = calculateGripStability(leftWrist, rightWrist, leftIndex, rightIndex, leftThumb, rightThumb)

        // 6. 상태 판별
        val state = determineStateMediaPipe(wristSpeed, wristHeight, shoulderRotation, elbowExtension, gripStability)
        val confidence = calculateConfidenceMediaPipe(state, wristSpeed, wristHeight, shoulderRotation, elbowExtension)

        return SwingPhaseAnalysis(state, confidence, wristSpeed, wristHeight, shoulderRotation)
    }

    private fun getPoint(keypoints: FloatArray, index: Int): Point2D {
        val x = keypoints[index * 2]
        val y = keypoints[index * 2 + 1]
        return Point2D(x, y)
    }

    private fun calculateAdvancedWristSpeed(
        leftWrist: Point2D, rightWrist: Point2D,
        leftIndex: Point2D, rightIndex: Point2D
    ): Float {
        val prev = previousFrame?.let {
            val kp = it.screen2d
            if (kp.size >= 66) {
                Pair(getPoint(kp, LEFT_WRIST), getPoint(kp, RIGHT_WRIST))
            } else null
        }

        return if (prev != null) {
            // 양손 속도 계산
            val leftSpeed = sqrt((leftWrist.x - prev.first.x).pow(2) + (leftWrist.y - prev.first.y).pow(2))
            val rightSpeed = sqrt((rightWrist.x - prev.second.x).pow(2) + (rightWrist.y - prev.second.y).pow(2))

            // 손가락 움직임도 고려 (더 세밀한 클럽 움직임 감지)
            val leftFingerSpeed = sqrt((leftIndex.x - leftWrist.x).pow(2) + (leftIndex.y - leftWrist.y).pow(2))
            val rightFingerSpeed = sqrt((rightIndex.x - rightWrist.x).pow(2) + (rightIndex.y - rightWrist.y).pow(2))

            // 종합 속도 (손목 + 손가락 움직임)
            maxOf(leftSpeed, rightSpeed) + (leftFingerSpeed + rightFingerSpeed) * 0.2f
        } else 0f
    }

    private fun calculateBodyRotation(
        leftShoulder: Point2D, rightShoulder: Point2D,
        leftHip: Point2D, rightHip: Point2D
    ): Float {
        // 어깨와 엉덩이 회전각의 평균
        val shoulderRotation = rightShoulder.x - leftShoulder.x
        val hipRotation = rightHip.x - leftHip.x
        return (shoulderRotation + hipRotation) / 2f
    }

    private fun calculateElbowAngle(shoulder: Point2D, elbow: Point2D, wrist: Point2D): Float {
        val vec1X = shoulder.x - elbow.x
        val vec1Y = shoulder.y - elbow.y
        val vec2X = wrist.x - elbow.x
        val vec2Y = wrist.y - elbow.y

        val angle = atan2(vec2Y, vec2X) - atan2(vec1Y, vec1X)
        return abs(angle * 180f / PI.toFloat())
    }

    private fun calculateGripStability(
        leftWrist: Point2D, rightWrist: Point2D,
        leftIndex: Point2D, rightIndex: Point2D,
        leftThumb: Point2D, rightThumb: Point2D
    ): Float {
        // 손목 간 거리 (그립 폭)
        val wristDistance = sqrt((leftWrist.x - rightWrist.x).pow(2) + (leftWrist.y - rightWrist.y).pow(2))

        // 손가락 정렬도 (클럽을 제대로 잡고 있는지)
        val leftFingerAlignment = sqrt((leftIndex.x - leftThumb.x).pow(2) + (leftIndex.y - leftThumb.y).pow(2))
        val rightFingerAlignment = sqrt((rightIndex.x - rightThumb.x).pow(2) + (rightIndex.y - rightThumb.y).pow(2))

        // 안정성 점수 (낮을수록 안정적)
        return 1f / (wristDistance + leftFingerAlignment + rightFingerAlignment + 0.1f)
    }

    private fun determineStateMediaPipe(
        wristSpeed: Float,
        wristHeight: Float,
        shoulderRotation: Float,
        elbowExtension: Float,
        gripStability: Float
    ): GolfSwingPhase {

        return when {
            // 1. ADDRESS: 정적, 낮은 위치, 안정적 그립
            wristSpeed < 0.03f && wristHeight < -0.1f && abs(shoulderRotation) < 0.1f && gripStability > 2.0f -> {
                GolfSwingPhase.ADDRESS
            }

            // 2. TAKEAWAY: 낮은 속도, 손목 올라가기 시작, 팔꿈치 펴지기 시작
            wristSpeed < 0.08f && wristHeight > -0.05f && elbowExtension > 140f && shoulderRotation < -0.05f -> {
                GolfSwingPhase.TAKEAWAY
            }

            // 3. BACKSWING: 중간 속도, 높은 위치, 백스윙 회전, 팔꿈치 많이 펴짐
            wristSpeed > 0.05f && wristSpeed < 0.2f && wristHeight > 0.2f && shoulderRotation < -0.15f && elbowExtension > 160f -> {
                GolfSwingPhase.BACKSWING
            }

            // 4. BACKSWING_TOP: 속도 감소, 최고점, 최대 백스윙 회전
            wristSpeed < 0.12f && wristHeight > 0.4f && shoulderRotation < -0.25f && elbowExtension > 170f -> {
                GolfSwingPhase.BACKSWING_TOP
            }

            // 5. DOWNSWING: 속도 급증, 높은 위치에서 내려옴, 회전 시작
            wristSpeed > 0.15f && wristHeight > 0.15f && shoulderRotation > -0.2f && elbowExtension > 140f -> {
                GolfSwingPhase.DOWNSWING
            }

            // 6. IMPACT: 최고 속도, 어깨-엉덩이 높이, 몸 정면
            wristSpeed > 0.3f && abs(wristHeight) < 0.15f && abs(shoulderRotation) < 0.15f -> {
                GolfSwingPhase.IMPACT
            }

            // 7. FOLLOW_THROUGH: 높은 속도, 올라가는 중, 팔로우쓰루 회전
            wristSpeed > 0.15f && wristHeight > 0.1f && shoulderRotation > 0.1f && elbowExtension > 150f -> {
                GolfSwingPhase.FOLLOW_THROUGH
            }

            // 8. FINISH: 속도 감소, 높은 피니시, 최대 회전
            wristSpeed < 0.1f && wristHeight > 0.3f && shoulderRotation > 0.2f -> {
                GolfSwingPhase.FINISH
            }

            else -> GolfSwingPhase.ADDRESS  // 기본값
        }
    }

    private fun calculateConfidenceMediaPipe(
        state: GolfSwingPhase,
        wristSpeed: Float,
        wristHeight: Float,
        shoulderRotation: Float,
        elbowExtension: Float
    ): Float {
        return when (state) {
            GolfSwingPhase.ADDRESS -> {
                val speedMatch = if (wristSpeed < 0.03f) 1.0f else 0.6f
                val positionMatch = if (wristHeight < -0.1f) 1.0f else 0.7f
                (speedMatch + positionMatch) / 2f * 0.9f
            }
            GolfSwingPhase.TAKEAWAY -> {
                val speedMatch = if (wristSpeed in 0.03f..0.08f) 1.0f else 0.7f
                val elbowMatch = if (elbowExtension > 140f) 1.0f else 0.6f
                (speedMatch + elbowMatch) / 2f * 0.85f
            }
            GolfSwingPhase.BACKSWING -> {
                val heightMatch = if (wristHeight > 0.2f) 1.0f else 0.7f
                val rotationMatch = if (shoulderRotation < -0.15f) 1.0f else 0.6f
                (heightMatch + rotationMatch) / 2f * 0.85f
            }
            GolfSwingPhase.BACKSWING_TOP -> {
                val heightMatch = if (wristHeight > 0.4f) 1.0f else 0.8f
                val elbowMatch = if (elbowExtension > 170f) 1.0f else 0.7f
                (heightMatch + elbowMatch) / 2f * 0.9f
            }
            GolfSwingPhase.DOWNSWING -> {
                val speedMatch = if (wristSpeed > 0.15f) 1.0f else 0.7f
                val rotationMatch = if (shoulderRotation > -0.2f) 1.0f else 0.6f
                (speedMatch + rotationMatch) / 2f * 0.85f
            }
            GolfSwingPhase.IMPACT -> {
                val speedMatch = if (wristSpeed > 0.3f) 1.0f else 0.8f
                val positionMatch = if (abs(wristHeight) < 0.15f) 1.0f else 0.7f
                (speedMatch + positionMatch) / 2f * 0.95f
            }
            GolfSwingPhase.FOLLOW_THROUGH -> {
                val speedMatch = if (wristSpeed > 0.15f) 1.0f else 0.7f
                val rotationMatch = if (shoulderRotation > 0.1f) 1.0f else 0.6f
                (speedMatch + rotationMatch) / 2f * 0.85f
            }
            GolfSwingPhase.FINISH -> {
                val heightMatch = if (wristHeight > 0.3f) 1.0f else 0.8f
                val rotationMatch = if (shoulderRotation > 0.2f) 1.0f else 0.7f
                (heightMatch + rotationMatch) / 2f * 0.9f
            }
        }
    }

    private fun Float.pow(n: Int): Float {
        var result = 1f
        repeat(n) { result *= this }
        return result
    }

    data class Point2D(val x: Float, val y: Float)
}