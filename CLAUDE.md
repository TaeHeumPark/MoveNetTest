# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is an Android pose estimation application using TensorFlow Lite MoveNet and MediaPipe models for real-time motion tracking and video analysis. The app specializes in swing motion detection and analysis with significant performance optimizations.

## Build Commands
- `./gradlew assembleDebug` - Compile debug APK with bundled native delegates
- `./gradlew installDebug` - Install debug build to connected device/emulator
- `./gradlew testDebugUnitTest` - Run JVM unit tests for fast regression checks
- `./gradlew connectedDebugAndroidTest` - Run instrumented tests on hardware
- `./gradlew lintDebug` - Run Android Lint (treat new warnings as blockers)

## Architecture Overview
- **Engine Layer**: Supports two pose estimation engines - MoveNet (TensorFlow Lite) and MediaPipe
- **Performance Tier System**: Light/Mid/Heavy tiers for different model complexities
- **Dual Mode Operation**: Real-time camera processing and video file analysis
- **Optimized Processing Pipeline**: YUV→RGB conversion with bitmap reuse, RenderScript allocation caching

### Key Components
- `MainActivity.kt` - Navigation hub for model selection and mode switching
- `tflite/` - Pose processing engines (MoveNet, MediaPipe) with performance optimizations
- `analysis/` - Video analysis pipeline with swing detection algorithms
- `ui/` - Fragment-based UI for real-time display and video analysis
- `util/` - Performance monitoring (LatencyMeter, FpsGovernor)
- `bench/` - Benchmark configuration and model asset management

### Critical Performance Features
- **Bitmap Reuse**: Recycled bitmap allocation for video analysis (reduced processing time from 4m50s to 26s)
- **YUV Conversion Caching**: Reusable byte buffers and RenderScript allocations
- **Frame-level Latency Tracking**: Real-time performance monitoring (~60ms per frame for MediaPipe full)
- **GPU Delegate Support**: Automatic fallback from GPU to CPU on errors

## Language Requirements
All comments and responses should be in Korean (한글).

## Code Style
- Kotlin style: 4-space indentation, camelCase members, PascalCase types
- UI components: suffix with role (`RealtimeDotsFragment`, `DotsOverlay`)
- Processors: end with `Processor`
- Coroutines: use `Dispatchers.Default` for inference, `Dispatchers.Main` for UI

## Testing Strategy
- JVM tests in `app/src/test/java` for pure logic
- Instrumented tests in `app/src/androidTest/java` for camera/GPU functionality
- Include performance validation for latency/FPS changes
- Manual validation required for camera hardware and GPU delegate changes

## Model Assets
- TensorFlow Lite models: `app/src/main/assets/models/*.tflite`
- MediaPipe task bundles: `app/src/main/assets/models/*.task`
- Update `android.packagingOptions.pickFirst` for new native libraries
- Document model provenance and version in PRs

## Performance Considerations
- Monitor `util/LatencyMeter` and `util/FpsGovernor` when making changes
- Test both GPU and CPU delegates for MediaPipe
- Validate swing detection accuracy after algorithm changes
- Include benchmark results for significant performance modifications