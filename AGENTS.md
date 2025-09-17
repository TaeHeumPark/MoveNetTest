# Repository Guidelines

## Project Structure & Module Organization
- `app/` hosts the Android application module; Kotlin sources live under `app/src/main/java/cc/ggrip/movenet` and are grouped by feature folders such as `ui`, `tflite`, `pose`, `util`, and `bench`.
- `app/src/main/res` stores XML layouts, drawables, and strings; keep resources close to the fragment or activity that consumes them.
- `app/src/main/assets/models` contains TensorFlow Lite and MediaPipe pose bundles; remove unused variants when shipping to reduce APK size.
- JVM unit tests reside in `app/src/test/java`; instrumentation suites live in `app/src/androidTest/java`.

## Build, Test, and Development Commands
- `./gradlew assembleDebug` compiles a debug APK with bundled native delegates.
- `./gradlew installDebug` pushes the debug build to a connected device or emulator.
- `./gradlew testDebugUnitTest` runs JVM tests; use it for fast regression checks.
- `./gradlew connectedDebugAndroidTest` launches instrumented tests on hardware.
- `./gradlew lintDebug` runs Android Lint; treat new warnings as blockers.

## Coding Style & Naming Conventions
- Adopt Kotlin style defaults: 4-space indentation, trailing commas where helpful, camelCase members, PascalCase types, and UPPER_SNAKE_CASE constants.
- Name UI components with their role suffix (`RealtimeDotsFragment`, `DotsOverlay`); processors end in `Processor`.
- Keep asynchronous work in coroutines; dispatch inference to `Dispatchers.Default` and UI updates to `Dispatchers.Main`.
- Place reusable helpers in `util/`; keep `MainActivity` focused on navigation wiring.

## Testing Guidelines
- Target `org.junit` JVM tests for pure logic; name files <ClassName>Test.kt and match the package path of the source.
- Mirror Android components in `androidTest` packages, using Espresso or Instrumentation APIs for camera flows.
- Add deterministic latency/FPS checks when editing `util/LatencyMeter` or `util/FpsGovernor`.
- Capture manual validation notes in the PR when camera hardware or GPU delegates are involved.

## Commit & Pull Request Guidelines
- Follow <Type> : <summary> formatting as in Feat : Enable GPU delegate; keep the summary imperative and scope a single concern per commit.
- Reference model moves or packaging changes inside the commit body when applicable.
- PRs should include a concise description, a checklist of executed commands, linked issues, and UI evidence (screenshots or screen recordings) for visual tweaks.
- Request peer review and wait for CI (unit, lint, instrumentation) to finish before merging.

## Model Assets & Configuration Notes
- Store approved .tflite or .task files inside app/src/main/assets/models; document provenance and version in the PR.
- Update android.packagingOptions.pickFirst when introducing new native libraries to avoid shared-object conflicts.
- Record threshold or delegate changes in bench/BenchmarkConfig comments so benchmark reruns stay reproducible.
