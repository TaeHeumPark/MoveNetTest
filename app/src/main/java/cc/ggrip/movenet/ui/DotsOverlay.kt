// DotsOverlay.kt
package cc.ggrip.movenet.ui

import android.content.Context
import android.graphics.*
import android.os.SystemClock
import android.view.View
import cc.ggrip.movenet.pose.PoseFrame
import cc.ggrip.movenet.util.LatencyMeter
import cc.ggrip.movenet.analysis.SwingPhaseAnalysis
import cc.ggrip.movenet.analysis.GolfSwingPhase
import kotlin.math.*

class DotsOverlay(
    context: Context,
    private val targetFps: Double,
    private val meter: LatencyMeter
) : View(context) {

    @Volatile private var frame: PoseFrame? = null
    @Volatile private var mirrorX: Boolean = false
    @Volatile private var flipY: Boolean = false

    @Volatile private var accelLabel: String = "CPU"
    fun setAcceleratorLabel(label: String) { accelLabel = label; postInvalidateOnAnimation() }

    @Volatile private var engineLabel: String = "MoveNet"
    fun setEngineLabel(label: String) { engineLabel = label; postInvalidateOnAnimation() }

    @Volatile private var modelLabel: String = "-"
    fun setModelLabel(label: String) { modelLabel = label; postInvalidateOnAnimation() }

    @Volatile private var swingAnalysis: SwingPhaseAnalysis? = null
    fun updateSwingState(analysis: SwingPhaseAnalysis) { swingAnalysis = analysis; postInvalidateOnAnimation() }

    @Volatile private var firstFrameReceivedAtMs: Long = -1L
    @Volatile private var firstUiLatencyMs: Long = -1L

    @Volatile private var srcW: Int = 0
    @Volatile private var srcH: Int = 0

    // 트레일 및 안정성 분석을 위한 히스토리
    private val TRAIL_LENGTH = 30  // 최근 30프레임
    private val STABILITY_WINDOW = 15  // 최근 15프레임 (0.5초 @ 30fps)
    private val jointHistory = Array(33) { ArrayDeque<PointF>(TRAIL_LENGTH) }
    private val jointStability = FloatArray(33)  // 각 관절의 표준편차

    // 스무딩 성능 비교 지표
    private var visibilityRate = 0f  // 관절 가시성 비율
    private var smoothnessScore = 0f  // 움직임 부드러움 점수 (jerk 기반)
    private var temporalCoherence = 0f  // 시간적 일관성
    private var noiseReduction = 0f  // 노이즈 감소율

    // 성능 측정용 버퍼
    private val jerkHistory = ArrayDeque<Float>(30)  // jerk 값 히스토리
    private var prevVelocity = PointF(0f, 0f)  // 이전 속도

    // 성능 로그 기록
    private var frameCount = 0L
    private var lastSwingPhase: GolfSwingPhase? = null
    private val performanceLog = mutableListOf<PerformanceRecord>()
    private var sessionStartTime = System.currentTimeMillis()

    data class PerformanceRecord(
        val timestamp: Long,
        val frameNumber: Long,
        val smoothingMode: String,  // RAW, EMA, FLK
        val swingPhase: GolfSwingPhase,
        val visibilityRate: Float,
        val smoothness: Float,
        val coherence: Float,
        val noiseReduction: Float
    )

    fun setMirrorFlip(mirrorX: Boolean, flipY: Boolean) {
        this.mirrorX = mirrorX
        this.flipY = flipY
        invalidate()
    }

    fun setSourceSize(w: Int, h: Int) {
        if (w != srcW || h != srcH) {
            srcW = w; srcH = h
            postInvalidateOnAnimation()
        }
    }

    fun update(f: PoseFrame) {
        if (firstFrameReceivedAtMs < 0 && f.frameReceivedTsMs > 0) {
            firstFrameReceivedAtMs = f.frameReceivedTsMs
        }
        frame = f
        frameCount++
        updateHistory(f)  // 히스토리 업데이트
        calculatePerformanceMetrics()   // 성능 지표 계산
        logPerformanceIfNeeded()  // 성능 로그 기록
        postInvalidateOnAnimation()
    }

    private fun updateHistory(f: PoseFrame) {
        val screen2d = f.screen2d
        val n = screen2d.size / 2

        for (i in 0 until minOf(n, 33)) {
            val x = screen2d[i * 2]
            val y = screen2d[i * 2 + 1]

            // 화면 좌표로 변환 (미러링 적용)
            val px = if (mirrorX) (1f - x) * width else x * width
            val py = if (flipY) (1f - y) * height else y * height

            val history = jointHistory[i]
            if (history.size >= TRAIL_LENGTH) {
                history.removeFirst()
            }
            history.addLast(PointF(px, py))

            // 안정성 계산 (표준편차)
            if (history.size >= STABILITY_WINDOW) {
                val recent = history.takeLast(STABILITY_WINDOW)
                val meanX = recent.map { it.x }.average().toFloat()
                val meanY = recent.map { it.y }.average().toFloat()

                val variance = recent.map { p ->
                    val dx = p.x - meanX
                    val dy = p.y - meanY
                    dx * dx + dy * dy
                }.average()

                jointStability[i] = sqrt(variance.toFloat())
            }
        }
    }

    private fun calculatePerformanceMetrics() {
        val f = frame ?: return
        val wristIdx = 16  // RIGHT_WRIST

        // 1. 관절 가시성 비율 (좌표 범위와 변화량 기반)
        var visibleCount = 0
        var stableCount = 0
        val totalJoints = f.screen2d.size / 2

        for (i in 0 until totalJoints) {
            val x = f.screen2d[i * 2]
            val y = f.screen2d[i * 2 + 1]

            // 화면 내 위치 체크
            if (x in 0.02f..0.98f && y in 0.02f..0.98f) {
                visibleCount++

                // 안정성 체크 (떨림이 적으면 실제로 감지된 것)
                if (i < jointStability.size && jointStability[i] < 10f) {
                    stableCount++
                }
            }
        }

        // 가시성 = (보이는 관절 + 안정적인 관절) / 2
        visibilityRate = if (totalJoints > 0) {
            (visibleCount.toFloat() / totalJoints * 0.5f +
             stableCount.toFloat() / totalJoints * 0.5f)
        } else 0f

        // 2. 움직임 부드러움 (Jerk 기반 - 가속도의 변화율)
        if (jointHistory[wristIdx].size >= 3) {
            val history = jointHistory[wristIdx].takeLast(5).toList()

            if (history.size >= 3) {
                // 현재 속도 계산
                val currVel = PointF(
                    history.last().x - history[history.size - 2].x,
                    history.last().y - history[history.size - 2].y
                )

                // Jerk = 가속도의 변화율 (속도 변화의 변화)
                val jerk = sqrt(
                    (currVel.x - prevVelocity.x) * (currVel.x - prevVelocity.x) +
                    (currVel.y - prevVelocity.y) * (currVel.y - prevVelocity.y)
                )

                prevVelocity = currVel

                // Jerk 히스토리 업데이트
                if (jerkHistory.size >= 30) jerkHistory.removeFirst()
                jerkHistory.addLast(jerk)

                // 평균 Jerk가 낮을수록 부드러움
                if (jerkHistory.size > 5) {
                    val avgJerk = jerkHistory.average().toFloat()
                    // Jerk를 0~1 범위로 정규화 (낮을수록 좋음)
                    smoothnessScore = 1f / (1f + avgJerk * 0.1f)
                }
            }
        }

        // 3. 시간적 일관성 (프레임 간 위치 예측 정확도)
        if (jointHistory[wristIdx].size >= 10) {
            val history = jointHistory[wristIdx].takeLast(10).toList()
            var coherenceSum = 0f

            // 선형 예측과 실제 위치 비교
            for (i in 2 until history.size) {
                // i-2, i-1로 i를 예측
                val predictedX = history[i-1].x + (history[i-1].x - history[i-2].x)
                val predictedY = history[i-1].y + (history[i-1].y - history[i-2].y)

                // 예측 오차
                val error = sqrt(
                    (predictedX - history[i].x) * (predictedX - history[i].x) +
                    (predictedY - history[i].y) * (predictedY - history[i].y)
                )

                // 오차가 작을수록 일관성 높음
                coherenceSum += 1f / (1f + error * 0.01f)
            }

            temporalCoherence = coherenceSum / (history.size - 2)
        }

        // 4. 노이즈 감소율 (신호 대 잡음비 개선)
        val currentStability = jointStability[wristIdx]

        // 안정성 점수를 노이즈 감소율로 변환
        // 표준편차가 낮을수록 노이즈가 적음
        noiseReduction = when {
            currentStability < 3f -> 0.95f    // 매우 깨끗
            currentStability < 6f -> 0.85f    // 깨끗
            currentStability < 10f -> 0.70f   // 보통
            currentStability < 15f -> 0.50f   // 노이즈 있음
            else -> 0.30f                     // 노이즈 많음
        }
    }

    private fun logPerformanceIfNeeded() {
        val currentPhase = swingAnalysis?.phase ?: return
        val mode = when {
            engineLabel.contains("FLK", true) -> "FLK"
            engineLabel.contains("EMA", true) -> "EMA"
            engineLabel.contains("Raw", true) -> "RAW"
            else -> "UNKNOWN"
        }

        // 스윙 단계가 변경되었거나 10프레임마다 로그
        if (currentPhase != lastSwingPhase || frameCount % 10 == 0L) {
            val record = PerformanceRecord(
                timestamp = System.currentTimeMillis() - sessionStartTime,
                frameNumber = frameCount,
                smoothingMode = mode,
                swingPhase = currentPhase,
                visibilityRate = visibilityRate,
                smoothness = smoothnessScore,
                coherence = temporalCoherence,
                noiseReduction = noiseReduction
            )

            performanceLog.add(record)

            // 콘솔 로그 출력
            android.util.Log.d("PerfMetrics",
                String.format("%s | %s | Vis:%.1f%% | Smooth:%.1f%% | Coh:%.1f%% | Noise:%.1f%%",
                    mode.padEnd(3),
                    currentPhase.name.padEnd(14),
                    visibilityRate * 100,
                    smoothnessScore * 100,
                    temporalCoherence * 100,
                    noiseReduction * 100
                )
            )

            // 스윙 단계 변경 시 요약 출력
            if (currentPhase != lastSwingPhase && lastSwingPhase != null) {
                val phaseRecords = performanceLog.filter {
                    it.swingPhase == lastSwingPhase && it.smoothingMode == mode
                }

                if (phaseRecords.isNotEmpty()) {
                    val avgSmooth = phaseRecords.map { it.smoothness }.average() * 100
                    val minSmooth = phaseRecords.minOf { it.smoothness } * 100
                    val maxSmooth = phaseRecords.maxOf { it.smoothness } * 100

                    android.util.Log.i("PerfSummary",
                        String.format("%s | %s 완료 | 부드러움: 평균 %.1f%% (최소 %.1f%% ~ 최대 %.1f%%)",
                            mode, lastSwingPhase?.name,
                            avgSmooth, minSmooth, maxSmooth
                        )
                    )
                }
            }

            lastSwingPhase = currentPhase
        }
    }

    // CSV 내보내기 메소드 (필요시 호출)
    fun exportPerformanceLog(): String {
        val csv = StringBuilder()
        csv.appendLine("Timestamp,Frame,Mode,Phase,Visibility%,Smoothness%,Coherence%,NoiseReduction%")

        performanceLog.forEach { record ->
            csv.appendLine("${record.timestamp},${record.frameNumber},${record.smoothingMode}," +
                "${record.swingPhase.name},${record.visibilityRate * 100}," +
                "${record.smoothness * 100},${record.coherence * 100},${record.noiseReduction * 100}")
        }

        return csv.toString()
    }
    private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFCC00.toInt()
        style = Paint.Style.FILL
    }
    private val hudPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFFFFF.toInt()
        textSize = 36f
        setShadowLayer(4f, 1f, 1f, 0x80000000.toInt())
    }
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x66000000
        style = Paint.Style.FILL
    }

    // 트레일 페인트
    private val trailPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f
        strokeCap = Paint.Cap.ROUND
    }

    // 안정성 링 페인트
    private val stabilityPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
        color = 0x80FFFFFF.toInt()
    }

    // KPI 텍스트 페인트
    private val kpiPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFFF00.toInt()
        textSize = 28f
        typeface = Typeface.MONOSPACE
    }

    // 스윙 상태 신호등 UI를 위한 페인트
    private val swingStatePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = 20f
        textAlign = Paint.Align.CENTER
    }
    private val swingIndicatorPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val swingIndicatorStrokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f
        color = 0xFFFFFFFF.toInt()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val W = width.toFloat()
        val H = height.toFloat()
        val f = frame

        if (f != null && srcW > 0 && srcH > 0) {
            // 화면에 맞게 원본을 letterbox/columnbox로 배치
            val scale = max(W / srcW, H / srcH)
            val offX = (W - srcW * scale) / 2f
            val offY = (H - srcH * scale) / 2f

            // 입력이 정사각형으로 크롭되어 온 경우를 고려한 매핑
            val cropSize = min(srcW, srcH).toFloat()
            val cropL = (srcW - cropSize) / 2f
            val cropT = (srcH - cropSize) / 2f

            val m = Matrix().apply {
                // 키포인트가 [0,1] 정규화 좌표라고 가정하고 원본 좌표로 투영
                setRectToRect(
                    RectF(0f, 0f, 1f, 1f),
                    RectF(cropL, cropT, cropL + cropSize, cropT + cropSize),
                    Matrix.ScaleToFit.FILL
                )
                // 화면 스케일/이동 적용
                postScale(scale, scale)
                postTranslate(offX, offY)
                // 미러/플립 옵션
                if (mirrorX || flipY) {
                    val px = offX + srcW * scale / 2f
                    val py = offY + srcH * scale / 2f
                    postScale(if (mirrorX) -1f else 1f, if (flipY) -1f else 1f, px, py)
                }
            }

            // 트레일 그리기 (키포인트 그리기 전에)
            drawTrails(canvas)

            // 키포인트 및 안정성 링 그리기
            val p = f.screen2d
            val n = p.size / 2  // 17(MoveNet) 또는 33(MediaPipe) 등
            val tmp = FloatArray(2)

            // 주요 관절 인덱스
            val keyJoints = listOf(15, 16)  // LEFT_WRIST, RIGHT_WRIST

            for (i in 0 until n) {
                tmp[0] = p[i * 2]
                tmp[1] = p[i * 2 + 1]
                m.mapPoints(tmp)

                // 안정성 링 그리기 (손목만)
                if (i in keyJoints && i < jointStability.size) {
                    drawStabilityRing(canvas, tmp[0], tmp[1], jointStability[i])
                }

                // 키포인트 점 그리기
                canvas.drawCircle(tmp[0], tmp[1], 10f, dotPaint)
            }

            // HUD: 지연 통계
            val nowMs = SystemClock.elapsedRealtime()
            val startTs = f.frameReceivedTsMs  // 앱이 프레임을 받은 시각(boottime ms) 사용
            var latestE2eMs = -1L
            if (startTs > 0) {
                val e2e = nowMs - startTs
                latestE2eMs = e2e
                val algo = if (f.algoStartTsMs > 0 && f.algoDoneTsMs > 0) {
                    f.algoDoneTsMs - f.algoStartTsMs
                } else -1L
                if (firstUiLatencyMs < 0 && firstFrameReceivedAtMs >= 0) {
                    firstUiLatencyMs = nowMs - firstFrameReceivedAtMs
                }
                meter.push(algo, e2e)
            }

            val stats = meter.snapshot()
            val frameInterval = 1000.0 / targetFps
            val eAvgF = if (!stats.e2eAvg.isNaN()) stats.e2eAvg / frameInterval else Double.NaN
            val eP95F = if (!stats.e2eP95.isNaN()) stats.e2eP95 / frameInterval else Double.NaN

            fun fmtMs(d: Double) = if (d.isNaN()) "-" else "%.1f".format(d)
            fun fmtFr(d: Double) = if (d.isNaN()) "-" else "%.2f".format(d)
            fun fmtMsLong(value: Long) = if (value < 0) "-" else fmtMs(value.toDouble())

            // 스윙 상태 신호등 그리기
            drawSwingStateIndicator(canvas, W, swingAnalysis)

            // 성능 지표 패널 그리기
            drawPerformanceMetrics(canvas)

            val lines = listOf(
                "$engineLabel • $modelLabel • 목표 ${"%.0f".format(targetFps)} FPS",
                "가속기: $accelLabel",
                "알고리즘 지연 평균/95퍼: ${fmtMs(stats.algoAvg)} / ${fmtMs(stats.algoP95)} ms",
                "E2E 평균/95퍼: ${fmtMs(stats.e2eAvg)} / ${fmtMs(stats.e2eP95)} ms",
                "카메라->UI(최근): ${fmtMsLong(latestE2eMs)} ms",
                "첫 프레임 지연: ${fmtMsLong(firstUiLatencyMs)} ms",
                "프레임 지연: ${fmtFr(eAvgF)}프 (평균) | ${fmtFr(eP95F)}프 (95p)"
            )

            val pad = 12f
            val boxW = lines.maxOf { hudPaint.measureText(it) } + pad * 2
            val boxH = hudPaint.textSize * lines.size + pad * 2
            canvas.drawRoundRect(16f, 16f, 16f + boxW, 16f + boxH, 18f, 18f, boxPaint)

            var yText = 16f + pad + hudPaint.textSize
            for (ln in lines) {
                canvas.drawText(ln, 16f + pad, yText, hudPaint)
                yText += hudPaint.textSize
            }
        } else {
            // 대기 메시지
            val msg = listOf(
                if (srcW == 0 || srcH == 0) "소스 크기 대기…" else "키포인트 대기…"
            )
            val pad = 12f
            val boxW = msg.maxOf { hudPaint.measureText(it) } + pad * 2
            val boxH = hudPaint.textSize * msg.size + pad * 2
            canvas.drawRoundRect(16f, 16f, 16f + boxW, 16f + boxH, 18f, 18f, boxPaint)

            var yText = 16f + pad + hudPaint.textSize
            for (ln in msg) {
                canvas.drawText(ln, 16f + pad, yText, hudPaint)
                yText += hudPaint.textSize
            }

        }
    }

    private fun drawTrails(canvas: Canvas) {
        // 손목 관절만 트레일 표시 (15: LEFT_WRIST, 16: RIGHT_WRIST)
        val wristIndices = listOf(15, 16)

        for (idx in wristIndices) {
            if (idx >= jointHistory.size) continue
            val history = jointHistory[idx]
            if (history.size < 2) continue

            val path = Path()
            var first = true

            // 트레일 그라데이션 효과
            history.forEachIndexed { i, point ->
                val alpha = (i.toFloat() / history.size * 255).toInt()
                trailPaint.alpha = alpha

                // 스무딩 정도에 따라 색상 변경
                val stability = if (idx < jointStability.size) jointStability[idx] else 0f
                trailPaint.color = when {
                    stability < 5f -> 0xFF00FF00.toInt()   // 안정적: 녹색
                    stability < 15f -> 0xFFFFFF00.toInt()  // 보통: 노란색
                    else -> 0xFFFF4444.toInt()             // 불안정: 빨간색
                }

                // 안정성에 따라 선 두께 조절
                trailPaint.strokeWidth = when {
                    stability < 5f -> 1.5f   // 얇은 선
                    stability < 15f -> 3f    // 중간
                    else -> 5f               // 두꺼운 선
                }

                if (first) {
                    path.moveTo(point.x, point.y)
                    first = false
                } else {
                    path.lineTo(point.x, point.y)
                }

                if (i > 0) {
                    val prevPoint = history.elementAt(i - 1)
                    canvas.drawLine(prevPoint.x, prevPoint.y, point.x, point.y, trailPaint)
                }
            }
        }
    }

    private fun drawStabilityRing(canvas: Canvas, x: Float, y: Float, stability: Float) {
        // 안정성 수치에 따른 링 색상
        stabilityPaint.color = when {
            stability < 5f -> 0x8000FF00.toInt()   // 안정: 녹색
            stability < 15f -> 0x80FFFF00.toInt()  // 보통: 노란색
            else -> 0x80FF4444.toInt()             // 불안정: 빨간색
        }

        // 링 크기는 안정성에 반비례
        val radius = when {
            stability < 5f -> 20f    // 작은 링 = 안정적
            stability < 15f -> 30f   // 중간 링
            else -> 40f              // 큰 링 = 불안정
        }

        // 링 두께도 안정성에 따라
        stabilityPaint.strokeWidth = when {
            stability < 5f -> 2f
            stability < 15f -> 3f
            else -> 4f
        }

        canvas.drawCircle(x, y, radius, stabilityPaint)

        // 안정성 수치 텍스트 (디버그용, 선택적)
        if (stability > 15f) {  // 불안정할 때만 수치 표시
            val textPaint = Paint(kpiPaint).apply {
                textSize = 16f
                color = 0xFFFF4444.toInt()
            }
            canvas.drawText("%.1f".format(stability), x + radius + 5, y, textPaint)
        }
    }

    private fun drawPerformanceMetrics(canvas: Canvas) {
        // 성능 지표를 화면 우측 상단에 표시
        val x = width - 320f
        val y = 30f
        val lineHeight = 35f

        // 배경 박스
        val bgPaint = Paint().apply {
            color = 0xBB000000.toInt()
            style = Paint.Style.FILL
        }
        canvas.drawRoundRect(x - 10, y - 10, width - 16f, y + lineHeight * 5 + 10, 14f, 14f, bgPaint)

        // 제목
        val titlePaint = Paint(kpiPaint).apply {
            textSize = 24f
            color = 0xFFFFCC00.toInt()
            isFakeBoldText = true
        }
        canvas.drawText("스무딩 효과 (실험적)", x, y + 15, titlePaint)

        // 지표 텍스트
        val metrics = listOf(
            "관절 감지율: %.1f%%".format(visibilityRate * 100),
            "움직임 부드러움: %.1f%%".format(smoothnessScore * 100),
            "시간적 일관성: %.1f%%".format(temporalCoherence * 100),
            "노이즈 감소: %.1f%%".format(noiseReduction * 100)
        )

        val metricPaint = Paint(kpiPaint).apply {
            textSize = 22f
        }

        metrics.forEachIndexed { i, text ->
            // 각 지표에 따른 색상 설정
            val value = when(i) {
                0 -> visibilityRate
                1 -> smoothnessScore
                2 -> temporalCoherence
                3 -> noiseReduction
                else -> 0f
            }

            metricPaint.color = when {
                value > 0.8f -> 0xFF00FF00.toInt()  // 녹색: 우수
                value > 0.5f -> 0xFFFFFF00.toInt()  // 노란색: 보통
                else -> 0xFFFF6666.toInt()          // 빨간색: 개선 필요
            }

            canvas.drawText(text, x, y + 50 + lineHeight * i, metricPaint)
        }
    }

    private fun drawSwingStateIndicator(canvas: Canvas, screenWidth: Float, analysis: SwingPhaseAnalysis?) {
        // 스윙 단계들
        val phases = listOf(
            GolfSwingPhase.ADDRESS to "어드레스",
            GolfSwingPhase.TAKEAWAY to "테이크",
            GolfSwingPhase.BACKSWING to "백스윙",
            GolfSwingPhase.BACKSWING_TOP to "탑",
            GolfSwingPhase.DOWNSWING to "다운",
            GolfSwingPhase.IMPACT to "임팩트",
            GolfSwingPhase.FOLLOW_THROUGH to "팔로우",
            GolfSwingPhase.FINISH to "피니시"
        )

        val indicatorHeight = 60f
        val indicatorY = height - indicatorHeight - 40f  // 화면 하단에서 40px 위
        val padding = 16f
        val totalWidth = screenWidth - (padding * 2)
        val itemWidth = totalWidth / phases.size
        val cornerRadius = 12f

        // 배경 박스
        val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0x88000000.toInt()
            style = Paint.Style.FILL
        }
        canvas.drawRoundRect(
            padding, indicatorY - 10f,
            screenWidth - padding, indicatorY + indicatorHeight + 10f,
            cornerRadius, cornerRadius, bgPaint
        )

        phases.forEachIndexed { index, (phase, label) ->
            val x = padding + (index * itemWidth)
            val centerX = x + (itemWidth / 2)
            val centerY = indicatorY + (indicatorHeight / 2)

            // 현재 상태 확인
            val isCurrentPhase = analysis?.phase == phase
            val isCompletedPhase = analysis?.let { currentAnalysis ->
                phases.indexOfFirst { it.first == currentAnalysis.phase } >= index
            } ?: false

            // 원 그리기 (신호등)
            val circleRadius = 18f

            swingIndicatorPaint.color = when {
                isCurrentPhase -> {
                    // 현재 상태: 밝은 녹색 + 깜빡임 효과
                    val alpha = (((System.currentTimeMillis() / 300) % 2) * 127 + 128).toInt()
                    (alpha shl 24) or 0x00FF00
                }
                isCompletedPhase -> 0xFF4CAF50.toInt()  // 완료된 상태: 연한 녹색
                else -> 0xFF424242.toInt()  // 미도달 상태: 회색
            }

            canvas.drawCircle(centerX, centerY - 10f, circleRadius, swingIndicatorPaint)

            // 테두리
            if (isCurrentPhase) {
                swingIndicatorStrokePaint.strokeWidth = 3f
                canvas.drawCircle(centerX, centerY - 10f, circleRadius, swingIndicatorStrokePaint)
            }

            // 레이블
            swingStatePaint.color = if (isCurrentPhase || isCompletedPhase) {
                0xFFFFFFFF.toInt()
            } else {
                0xFF888888.toInt()
            }
            swingStatePaint.textSize = if (isCurrentPhase) 18f else 16f
            swingStatePaint.isFakeBoldText = isCurrentPhase

            canvas.drawText(label, centerX, centerY + 20f, swingStatePaint)

            // 신뢰도 표시 (현재 상태일 때만)
            if (isCurrentPhase && analysis != null) {
                val confidenceText = "${(analysis.confidence * 100).toInt()}%"
                swingStatePaint.textSize = 14f
                swingStatePaint.color = 0xFFFFCC00.toInt()
                canvas.drawText(confidenceText, centerX, centerY + 38f, swingStatePaint)
            }
        }
    }
}
