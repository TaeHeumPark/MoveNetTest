// LatencyMeter.kt
package cc.ggrip.movenet.util

class LatencyMeter(private val capacity: Int = 240) {
    data class Sample(val algoMs: Long, val e2eMs: Long)
    private val buf = ArrayList<Sample>(capacity)

    @Synchronized fun push(algoMs: Long, e2eMs: Long) {
        if (algoMs < 0 || e2eMs < 0) return
        if (buf.size == capacity) buf.removeAt(0)
        buf.add(Sample(algoMs, e2eMs))
    }

    @Synchronized fun snapshot(): Stats {
        if (buf.isEmpty()) return Stats()
        val a = buf.map { it.algoMs.toDouble() }.sorted()
        val e = buf.map { it.e2eMs.toDouble() }.sorted()
        fun p(v: List<Double>, q: Double): Double {
            val i = (q * (v.size - 1)).coerceIn(0.0, (v.size - 1).toDouble())
            val lo = v[i.toInt()]
            val hi = v[kotlin.math.min(i.toInt()+1, v.size-1)]
            val t = i - i.toInt()
            return lo*(1-t) + hi*t
        }
        return Stats(
            algoAvg = a.average(), algoP95 = p(a, 0.95),
            e2eAvg  = e.average(), e2eP95  = p(e, 0.95),
            count   = buf.size
        )
    }

    data class Stats(
        val algoAvg: Double = Double.NaN,
        val algoP95: Double = Double.NaN,
        val e2eAvg:  Double = Double.NaN,
        val e2eP95:  Double = Double.NaN,
        val count:   Int = 0
    )
}
