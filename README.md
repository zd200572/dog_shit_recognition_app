# 💩 Dog Shit Detector

AI-powered real-time dog shit detection mobile app built with Flutter + YOLOv8.

> Never step on dog shit again. Your phone watches the ground so you don't have to.

---

## Demo

```
[Camera View] ───auto-capture every 4s──▶ [ONNX Inference] ──▶ [Bounding Boxes + Vibration + TTS]
```

---

## Architecture

```
dog_shit_app_flutter/
├── lib/
│   └── main.dart              # All app logic (~500 lines)
├── assets/
│   └── models/
│       └── best.onnx          # YOLOv8n ONNX model (11.5 MB)
├── android/                   # Android build config
├── pubspec.yaml               # Dependencies
└── README.md
```

---

## Model — How It Was Trained

### Dataset
- **Source**: Custom collected images of dog shit in various environments
- **Total**: 1,233 images
- **Split**: 934 training / 299 validation
- **Annotations**: Bounding boxes (YOLO format)

### Training
- **Framework**: Ultralytics YOLOv8n (nano, fastest)
- **Epochs**: 100 (best @ epoch 87)
- **Results** (validation set):
  | Metric | Value |
  |--------|-------|
  | mAP50  | **0.914** |
  | Precision | **0.935** |
  | Recall | **0.814** |

- **Comparison**:
  - exp1 (96 images): mAP50=0.852
  - exp3_v4 (1,233 images): mAP50=0.914 ← **this model**

### Export to ONNX
```python
from ultralytics import YOLO
model = YOLO('dog_shit_runs/exp3_v4/weights/best.pt')
model.export(format='onnx', opset=12, dynamic=False)
# Output: best.onnx (11.5 MB)
# Input: [1, 3, 640, 640]  (NCHW, 0-1 normalized)
# Output: [1, 5, 8400]       (cx, cy, w, h, conf)
```

The ONNX file is bundled in `assets/models/best.onnx`.

---

## App — How It Works

### Tech Stack
| Layer | Package | Purpose |
|-------|----------|---------|
| UI Framework | `flutter` | Cross-platform UI |
| Camera | `camera: ^0.12.0` | Live preview + capture |
| Inference | `onnxruntime: ^1.1.0` | ONNX model execution |
| Alerts | `flutter_vibrate: ^1.3.0` | Haptic feedback |
| Alerts | `flutter_tts: ^4.2.1` | Voice warning |
| Image Picker | `image_picker: ^1.1.2` | Gallery import |

### Inference Pipeline

```
Image (camera or gallery)
  │
  ▼
[Preprocess]  Resize to 640×640, letterbox padding,
  │            RGB→normalized float [0,1], NCHW layout
  ▼
[ONNX Runtime]  OrtSession.run()
  │             Output: [1, 5, 8400]
  ▼
[Postprocess]   Filter by confidence threshold (_confidence)
  │             Non-Maximum Suppression (NMS) by IoU (_iouThreshold)
  ▼
[Display]      Draw bounding boxes on image
[Alert]        Vibrate ×3 + TTS: "Warning! N dog shit(s) detected ahead!"
```

### Key Parameters
| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| Confidence | 0.5 | 0.1 – 0.9 | Lower = more detections, higher false positives |
| IoU Threshold | 0.7 | 0.3 – 0.9 | NMS overlap threshold |
| Auto-capture Interval | 4s | fixed | Only when auto-detection is ON |

### App Features
- ✅ **Auto-detection mode**: Camera captures + runs inference every 4 seconds
- ✅ **Manual mode**: Tap to capture from camera, or pick from gallery
- ✅ **Live preview**: Camera stream when auto-detection is enabled
- ✅ **Bounding box visualization**: Red boxes with confidence labels
- ✅ **Vibration alert**: 3 quick pulses when dog shit is detected
- ✅ **Voice alert**: English TTS warning message
- ✅ **Settings sheet**: Toggle auto/vibration/voice, adjust sliders
- ✅ **Dark mode support**: Follows system theme

---

## Installation

### Pre-built APK
See [Releases](https://github.com/your-repo/dog-shit-detector/releases) (coming soon).

### Build from Source

**Requirements**:
- Flutter SDK 3.44.0+
- Android SDK (API 34+)
- Android NDK r28b (or compatible)

```bash
# 1. Clone
git clone https://github.com/your-repo/dog-shit-detector.git
cd dog-shit-detector/dog_shit_app_flutter

# 2. Install dependencies
flutter pub get

# 3. Connect Android device or start emulator
flutter devices

# 4. Run (debug)
flutter run

# 5. Build release APK
flutter build apk --release
# Output: build/app/outputs/flutter-apk/app-release.apk
```

> **Windows users**: If you hit Gradle/TLS issues, build on Linux/macOS or use the pre-built APK.

---

## Usage

1. **Grant camera permission** when prompted
2. **Auto-detection is ON by default** — point camera at the ground
3. App captures every 4 seconds and runs detection
4. If dog shit is detected:
   - 📳 Phone vibrates 3 times
   - 🔊 TTS says: *"Warning! N dog shit(s) detected ahead!"*
   - 🔴 Bounding box appears on screen
5. **Adjust sensitivity**: Tap ⚙️ settings icon
   - Lower confidence = more detections (may include false positives)
   - Higher confidence = fewer but more reliable detections
6. **Turn off auto-detection** to use manual capture/gallery mode

---

## Model Performance Notes

| Confidence Threshold | Speed | Detections | False Positives | Use Case |
|---------------------|-------|-------------|-----------------|----------|
| 0.3 | ~17ms/img (58 FPS) | Highest | More likely | Data collection |
| **0.5 (default)** | **~8ms/img (127 FPS)** | **Balanced** | **Minimal** | **Daily use** ✅ |
| 0.7 | ~7ms/img (137 FPS) | Lowest | Near zero | Automated cleanup |

---

## Known Issues

- [ ] ONNX output parsing assumes single-class `[1, 5, 8400]` format — update if you retrain with more classes
- [ ] Camera auto-capture may drain battery quickly
- [ ] TTS language is hardcoded to `en-US`
- [ ] No iOS test device — iOS build untested

---

## Retrain the Model

```bash
pip install ultralytics

# Train
yolo detect train model=yolov8n.pt data=dog_shit_dataset.yaml epochs=100 imgsz=640

# Export to ONNX
yolo export model=path/to/best.pt format=onnx opset=12

# Replace assets/models/best.onnx and rebuild app
```

Dataset structure:
```
dog_shit_dataset/
├── images/
│   ├── train/  (934 images)
│   └── val/    (299 images)
├── labels/
│   ├── train/  (934 .txt files)
│   └── val/    (299 .txt files)
└── data.yaml
```

---

## License

MIT License — free to use, modify, and distribute.

---

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — SOTA object detection
- [ONNX Runtime](https://onnxruntime.ai/) — Cross-platform inference
- Flutter team — Cross-platform mobile framework

---

*Made with 💩 and Flutter. Step safely.*
