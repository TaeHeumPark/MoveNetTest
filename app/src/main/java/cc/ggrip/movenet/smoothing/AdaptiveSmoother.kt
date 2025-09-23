package cc.ggrip.movenet.smoothing

import kotlin.math.sqrt

data class Vec3(var x: Float, var y: Float, var z: Float)

class AdaptiveSmoother(
    private val lowSpeed: Float = 0.01f,   // 거의 정지
    private val highSpeed: Float = 0.30f,  // 매우 빠름(임팩트 근처)
    private val minAlpha: Float = 0.10f,   // 강한 스무딩(느릴 때)
    private val maxAlpha: Float = 0.80f,   // 약한 스무딩(빠를 때)
    private val jumpThresh: Float = 0.50f  // 말도 안 되는 점프(정규화 좌표 기준)
) {
    // 속도를 [lowSpeed..highSpeed] → [minAlpha..maxAlpha] 로 선형 매핑
    private fun alphaFor(speed: Float): Float {
        val t = ((speed - lowSpeed) / (highSpeed - lowSpeed)).coerceIn(0f, 1f)
        return (minAlpha + t * (maxAlpha - minAlpha)).coerceIn(minAlpha, maxAlpha)
    }

    fun filter(raw: Vec3, prevFiltered: Vec3?, dt: Float, confidence: Float = 1f): Vec3 {
        if (prevFiltered == null) return raw

        // 프레임간 속도(정규화 좌표 기준의 차분 크기 / dt)
        val dx = raw.x - prevFiltered.x
        val dy = raw.y - prevFiltered.y
        val dz = raw.z - prevFiltered.z
        val speed = (sqrt(dx*dx + dy*dy + dz*dz) / (dt.coerceAtLeast(1e-3f)))

        // 신뢰도 낮을수록 더 강하게(= alpha↓)
        val confScale = (0.5f + 0.5f * confidence.coerceIn(0f, 1f)) // 0.5~1.0
        var alpha = alphaFor(speed) * confScale

        // 말도 안 되는 점프는 급격히 줄이기(클리핑/블렌딩)
        val jump = sqrt(dx*dx + dy*dy + dz*dz)
        if (jump > jumpThresh) alpha = minAlpha

        // EMA 적용: filtered = (1-alpha)*prev + alpha*raw
        return Vec3(
            prevFiltered.x + alpha * (raw.x - prevFiltered.x),
            prevFiltered.y + alpha * (raw.y - prevFiltered.y),
            prevFiltered.z + alpha * (raw.z - prevFiltered.z)
        )
    }
}

class PoseSmoother {
    private val jointSmoothers = mutableMapOf<Int, AdaptiveSmoother>()
    private val prevFiltered = mutableMapOf<Int, Vec3>()

    fun smooth(
        jointIndex: Int,
        raw3D: Vec3,
        dt: Float,
        confidence: Float = 1f
    ): Vec3 {
        val smoother = jointSmoothers.getOrPut(jointIndex) { AdaptiveSmoother() }
        val filtered = smoother.filter(raw3D, prevFiltered[jointIndex], dt, confidence)
        prevFiltered[jointIndex] = filtered
        return filtered
    }

    fun reset() {
        prevFiltered.clear()
    }
}