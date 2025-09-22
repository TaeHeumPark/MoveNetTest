package cc.ggrip.movenet.analysis

data class Point2D(val x: Float, val y: Float) {
    fun distanceTo(other: Point2D): Float {
        val dx = x - other.x
        val dy = y - other.y
        return kotlin.math.sqrt(dx * dx + dy * dy)
    }

    fun midPoint(other: Point2D): Point2D {
        return Point2D((x + other.x) / 2f, (y + other.y) / 2f)
    }
}