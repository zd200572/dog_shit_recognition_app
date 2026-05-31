import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:camera/camera.dart' as camera;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_vibrate/flutter_vibrate.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:onnxruntime/onnxruntime.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const DogShitDetectorApp());
}

class DogShitDetectorApp extends StatelessWidget {
  const DogShitDetectorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Dog Shit Detector',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: const Color(0xFF8B4513),
        useMaterial3: true,
        brightness: Brightness.light,
      ),
      darkTheme: ThemeData(
        colorSchemeSeed: const Color(0xFFD2691E),
        useMaterial3: true,
        brightness: Brightness.dark,
      ),
      home: const DetectorHomePage(),
    );
  }
}

class DetectorHomePage extends StatefulWidget {
  const DetectorHomePage({super.key});

  @override
  State<DetectorHomePage> createState() => _DetectorHomePageState();
}

class _DetectorHomePageState extends State<DetectorHomePage>
    with WidgetsBindingObserver {
  OrtSession? _session;
  camera.CameraController? _cameraController;
  Timer? _autoCaptureTimer;
  bool _modelLoaded = false;
  bool _isLoading = false;
  bool _cameraInitializing = false;
  bool _cameraReady = false;
  bool _autoCaptureEnabled = true;
  bool _autoCaptureBusy = false;
  String _status = "Starting automatic detection...";

  File? _imageFile;
  Uint8List? _imageBytes;
  List<DetectionResult> _detections = [];

  double _confidence = 0.5;
  double _iouThreshold = 0.7;
  bool _vibrationEnabled = true;
  bool _voiceEnabled = true;

  late FlutterTts _tts;
  bool _canVibrate = false;
  int _detectedImageWidth = 640;
  int _detectedImageHeight = 640;

  static const int inputSize = 640;
  static const Duration autoCaptureInterval = Duration(seconds: 4);

  bool get _busy => _isLoading || _cameraInitializing || _autoCaptureBusy;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initServices();
    _initCamera();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _stopAutoCapture();
    _cameraController?.dispose();
    _tts.stop();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      _disposeCamera();
    } else if (state == AppLifecycleState.resumed && _autoCaptureEnabled) {
      _initCamera();
    }
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_modelLoaded && !_isLoading) {
      _loadModel();
    }
  }

  Future<void> _initServices() async {
    _tts = FlutterTts();
    await _tts.setLanguage("en-US");
    await _tts.setSpeechRate(0.5);
    await _tts.setVolume(1.0);
    await _tts.setPitch(1.0);

    _canVibrate = await Vibrate.canVibrate;
  }

  Future<void> _triggerAlerts(int count) async {
    if (count == 0) return;

    if (_vibrationEnabled && _canVibrate) {
      Vibrate.vibrate();
      await Future.delayed(const Duration(milliseconds: 150));
      Vibrate.vibrate();
      await Future.delayed(const Duration(milliseconds: 150));
      Vibrate.vibrate();
    }

    if (_voiceEnabled) {
      final msg = count == 1
          ? "Warning! One dog shit detected ahead!"
          : "Warning! $count dog shits detected ahead!";
      await _tts.speak(msg);
    }
  }

  Future<void> _loadModel() async {
    setState(() {
      _isLoading = true;
      _status = "Loading model...";
    });

    try {
      final rawAsset =
          await DefaultAssetBundle.of(context).load('assets/models/best.onnx');

      _session = OrtSession.fromBuffer(
        rawAsset.buffer.asUint8List(),
        OrtSessionOptions()..setIntraOpNumThreads(2),
      );

      setState(() {
        _modelLoaded = true;
        _isLoading = false;
        _status = _autoCaptureEnabled
            ? "Model loaded. Automatic detection is starting..."
            : "Model loaded! Ready to detect";
      });
      _syncAutoCapture();
    } catch (e) {
      setState(() {
        _isLoading = false;
        _status = "Failed to load model: $e";
      });
    }
  }

  Future<void> _initCamera() async {
    if (_cameraController != null || _cameraInitializing) return;

    setState(() {
      _cameraInitializing = true;
      if (_autoCaptureEnabled) {
        _status = "Starting camera...";
      }
    });

    try {
      final cameras = await camera.availableCameras();
      if (cameras.isEmpty) {
        throw Exception("No camera found");
      }

      final selectedCamera = cameras.firstWhere(
        (cam) => cam.lensDirection == camera.CameraLensDirection.back,
        orElse: () => cameras.first,
      );

      final controller = camera.CameraController(
        selectedCamera,
        camera.ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: camera.ImageFormatGroup.jpeg,
      );

      await controller.initialize();

      if (!mounted) {
        await controller.dispose();
        return;
      }

      setState(() {
        _cameraController = controller;
        _cameraReady = true;
        _cameraInitializing = false;
        if (_autoCaptureEnabled) {
          _status = _modelLoaded
              ? "Automatic detection is running"
              : "Camera ready. Loading model...";
        }
      });

      _syncAutoCapture();
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _cameraReady = false;
        _cameraInitializing = false;
        _status = "Camera unavailable: $e";
      });
    }
  }

  void _syncAutoCapture() {
    if (!_autoCaptureEnabled || !_modelLoaded || !_cameraReady) return;
    _autoCaptureTimer ??= Timer.periodic(
      autoCaptureInterval,
      (_) => _captureAndDetect(),
    );
    _captureAndDetect();
  }

  void _stopAutoCapture() {
    _autoCaptureTimer?.cancel();
    _autoCaptureTimer = null;
  }

  Future<void> _disposeCamera() async {
    _stopAutoCapture();
    final controller = _cameraController;
    if (controller == null) return;

    setState(() {
      _cameraController = null;
      _cameraReady = false;
    });

    await controller.dispose();
  }

  Future<void> _setAutoCaptureEnabled(bool enabled) async {
    setState(() {
      _autoCaptureEnabled = enabled;
      _status = enabled
          ? "Automatic detection is starting..."
          : "Automatic detection paused. Use Camera or Gallery.";
    });

    if (enabled) {
      if (!_cameraReady) {
        await _initCamera();
      }
      _syncAutoCapture();
    } else {
      await _disposeCamera();
    }
  }

  Future<void> _captureAndDetect() async {
    final controller = _cameraController;
    if (!_autoCaptureEnabled ||
        !_modelLoaded ||
        controller == null ||
        !controller.value.isInitialized ||
        controller.value.isTakingPicture ||
        _autoCaptureBusy) {
      return;
    }

    setState(() {
      _autoCaptureBusy = true;
      _status = "Capturing and detecting...";
    });

    try {
      final captured = await controller.takePicture();
      await _processImageFile(File(captured.path));
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _status = "Automatic capture failed: $e";
      });
    } finally {
      if (mounted) {
        setState(() {
          _autoCaptureBusy = false;
        });
      }
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    final wasAutoCaptureEnabled = _autoCaptureEnabled;
    _stopAutoCapture();

    final picker = ImagePicker();
    final picked = await picker.pickImage(
      source: source,
      imageQuality: 85,
      maxWidth: 1920,
      maxHeight: 1920,
    );

    if (picked == null) {
      if (wasAutoCaptureEnabled) _syncAutoCapture();
      return;
    }

    await _processImageFile(File(picked.path));
    if (wasAutoCaptureEnabled) _syncAutoCapture();
  }

  Future<void> _processImageFile(File imageFile) async {
    setState(() {
      _imageFile = imageFile;
      _detections = [];
      _status = "Processing image...";
      _isLoading = true;
    });

    _imageBytes = await imageFile.readAsBytes();

    final codec = await ui.instantiateImageCodec(_imageBytes!);
    final frame = await codec.getNextFrame();
    _detectedImageWidth = frame.image.width;
    _detectedImageHeight = frame.image.height;

    await _runInference();
  }

  Future<void> _runInference() async {
    if (_session == null || _imageBytes == null) return;

    try {
      final inputTensor = await _preprocessImage(_imageBytes!);

      final inputs = {_session!.inputNames.first: inputTensor};
      final outputs = _session!.run(OrtRunOptions(), inputs);

      final rawOutput = outputs[0];
      _detections = _postprocess(rawOutput!);

      setState(() {
        _isLoading = false;
        if (_detections.isEmpty) {
          _status = "No dog shit detected";
        } else {
          _status = "Detected ${_detections.length} target(s)!";
        }
      });

      await _triggerAlerts(_detections.length);
    } catch (e) {
      setState(() {
        _isLoading = false;
        _status = "Detection error: $e";
      });
    }
  }

  Future<OrtValueTensor> _preprocessImage(Uint8List bytes) async {
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    final image = frame.image;

    final imgWidth = image.width;
    final imgHeight = image.height;

    final scale = min(inputSize / imgWidth, inputSize / imgHeight);
    final newW = (imgWidth * scale).round();
    final newH = (imgHeight * scale).round();
    final dw = (inputSize - newW) ~/ 2;
    final dh = (inputSize - newH) ~/ 2;

    final byteData = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
    if (byteData == null) throw Exception("Cannot read image pixels");

    final pixels = byteData.buffer.asUint8List();
    final input = List.filled(inputSize * inputSize * 3, 0.0);

    for (int y = 0; y < newH; y++) {
      for (int x = 0; x < newW; x++) {
        final srcX = (x / newW * imgWidth).round().clamp(0, imgWidth - 1);
        final srcY = (y / newH * imgHeight).round().clamp(0, imgHeight - 1);
        final pixelOffset = (srcY * imgWidth + srcX) * 4;

        final r = pixels[pixelOffset] / 255.0;
        final g = pixels[pixelOffset + 1] / 255.0;
        final b = pixels[pixelOffset + 2] / 255.0;

        final dx = x + dw;
        final dy = y + dh;

        // NCHW layout: channels are separated into three planes of size inputSize * inputSize
        final rIdx = 0 * inputSize * inputSize + dy * inputSize + dx;
        final gIdx = 1 * inputSize * inputSize + dy * inputSize + dx;
        final bIdx = 2 * inputSize * inputSize + dy * inputSize + dx;

        input[rIdx] = r;
        input[gIdx] = g;
        input[bIdx] = b;
      }
    }

    return OrtValueTensor.createTensorWithDataList(
      Float32List.fromList(input),
      [1, 3, inputSize, inputSize],
    );
  }

  List<DetectionResult> _postprocess(OrtValue output) {
    // YOLOv8 output shape: [1, 5, 8400] -> 5 = cx, cy, w, h, conf
    final rawData = output.value as List;
    if (rawData.isEmpty) return [];

    // rawData is [1][5][8400]
    final batch = rawData[0] as List;
    final numAnchors = (batch[0] as List).length;
    final detections = <DetectionResult>[];

    for (int i = 0; i < numAnchors; i++) {
      final cx = (batch[0] as List)[i] as double;
      final cy = (batch[1] as List)[i] as double;
      final w = (batch[2] as List)[i] as double;
      final h = (batch[3] as List)[i] as double;
      final conf = (batch[4] as List)[i] as double;

      if (conf < _confidence) continue;

      detections.add(DetectionResult(
        x1: cx - w / 2,
        y1: cy - h / 2,
        x2: cx + w / 2,
        y2: cy + h / 2,
        confidence: conf,
        classId: 0,
        className: 'dog shit',
      ));
    }

    return _nms(detections, _iouThreshold);
  }

  List<DetectionResult> _nms(List<DetectionResult> dets, double iouThresh) {
    if (dets.isEmpty) return dets;
    dets.sort((a, b) => b.confidence.compareTo(a.confidence));

    final keep = <DetectionResult>[];
    final suppressed = List<bool>.filled(dets.length, false);

    for (int i = 0; i < dets.length; i++) {
      if (suppressed[i]) continue;
      keep.add(dets[i]);
      for (int j = i + 1; j < dets.length; j++) {
        if (!suppressed[j] && _iou(dets[i], dets[j]) > iouThresh) {
          suppressed[j] = true;
        }
      }
    }
    return keep;
  }

  double _iou(DetectionResult a, DetectionResult b) {
    final ix1 = max(a.x1, b.x1);
    final iy1 = max(a.y1, b.y1);
    final ix2 = min(a.x2, b.x2);
    final iy2 = min(a.y2, b.y2);
    final interArea = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1);
    final areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    final areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    final unionArea = areaA + areaB - interArea;
    return unionArea <= 0 ? 0 : interArea / unionArea;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('💩 Dog Shit Detector'),
        elevation: 2,
        actions: [
          IconButton(
            icon: const Icon(Icons.settings_outlined),
            onPressed: () => _showSettings(),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _buildStatusCard(),
            const SizedBox(height: 16),
            _buildAutoModeCard(),
            const SizedBox(height: 16),
            _buildCaptureArea(),
            const SizedBox(height: 16),
            if (_detections.isNotEmpty) ...[
              _buildResultsCard(),
              const SizedBox(height: 16),
            ],
            _buildActionButtons(),
            const SizedBox(height: 32),
          ],
        ),
      ),
    );
  }

  Widget _buildStatusCard() {
    Color statusColor;
    IconData statusIcon;

    if (_busy) {
      statusColor = Colors.orange;
      statusIcon = Icons.hourglass_top;
    } else if (!_modelLoaded) {
      statusColor = Colors.red;
      statusIcon = Icons.error_outline;
    } else if (_detections.isNotEmpty) {
      statusColor = Colors.red.shade700;
      statusIcon = Icons.warning_amber_rounded;
    } else if (_imageFile != null) {
      statusColor = Colors.green;
      statusIcon = Icons.check_circle_outline;
    } else {
      statusColor = Colors.blue;
      statusIcon = Icons.info_outline;
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: statusColor.withAlpha(25),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: statusColor.withAlpha(77)),
      ),
      child: Row(
        children: [
          Icon(statusIcon, color: statusColor, size: 24),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              _status,
              style: TextStyle(
                color: statusColor,
                fontWeight: FontWeight.w500,
                fontSize: 14,
              ),
            ),
          ),
          if (_busy)
            const SizedBox(
              width: 20,
              height: 20,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
        ],
      ),
    );
  }

  Widget _buildAutoModeCard() {
    final theme = Theme.of(context);
    final subtitle = _autoCaptureEnabled
        ? _cameraReady
            ? "Captures every ${autoCaptureInterval.inSeconds}s"
            : "Preparing camera"
        : "Manual capture";

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerHighest.withAlpha(128),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: theme.dividerColor.withAlpha(64)),
      ),
      child: Row(
        children: [
          Icon(
            _autoCaptureEnabled
                ? Icons.motion_photos_auto_outlined
                : Icons.touch_app_outlined,
            color: theme.colorScheme.primary,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Automatic detection',
                  style: TextStyle(fontWeight: FontWeight.w600),
                ),
                const SizedBox(height: 2),
                Text(
                  subtitle,
                  style: TextStyle(
                    color: theme.colorScheme.onSurfaceVariant,
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
          Switch(
            value: _autoCaptureEnabled,
            onChanged: _busy ? null : (value) => _setAutoCaptureEnabled(value),
          ),
        ],
      ),
    );
  }

  Widget _buildCaptureArea() {
    if (_imageFile != null) {
      return _buildImageWithDetections();
    }

    if (_autoCaptureEnabled) {
      return _buildCameraPreview();
    }

    return _buildPlaceholder();
  }

  Widget _buildPlaceholder() {
    return Container(
      height: 300,
      decoration: BoxDecoration(
        color: Colors.grey.shade100,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.grey.shade300),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.photo_camera_back_outlined,
              size: 64, color: Colors.grey.shade400),
          const SizedBox(height: 12),
          Text(
            'Take a photo or pick from gallery',
            style: TextStyle(color: Colors.grey.shade600, fontSize: 16),
          ),
          const SizedBox(height: 8),
          Text(
            'YOLOv8 • mAP50=0.914 • 1233 images trained',
            style: TextStyle(color: Colors.grey.shade500, fontSize: 12),
          ),
          const SizedBox(height: 4),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.vibration, size: 14, color: Colors.grey.shade500),
              const SizedBox(width: 4),
              Text(
                'Vibration + Voice alerts on detection',
                style: TextStyle(color: Colors.grey.shade500, fontSize: 12),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildCameraPreview() {
    final controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) {
      return _buildPlaceholder();
    }

    return ClipRRect(
      borderRadius: BorderRadius.circular(16),
      child: AspectRatio(
        aspectRatio: controller.value.aspectRatio,
        child: camera.CameraPreview(controller),
      ),
    );
  }

  Widget _buildImageWithDetections() {
    return ClipRRect(
      borderRadius: BorderRadius.circular(16),
      child: AspectRatio(
        aspectRatio: _detectedImageWidth / _detectedImageHeight,
        child: LayoutBuilder(
          builder: (context, constraints) {
            return Stack(
              fit: StackFit.expand,
              children: [
                Image.file(
                  _imageFile!,
                  fit: BoxFit.contain,
                ),
                if (_detections.isNotEmpty)
                  CustomPaint(
                    painter: DetectionBoxPainter(
                      detections: _detections,
                      imageSize: Size(
                        _detectedImageWidth.toDouble(),
                        _detectedImageHeight.toDouble(),
                      ),
                      displaySize: constraints.biggest,
                    ),
                  ),
              ],
            );
          },
        ),
      ),
    );
  }

  Widget _buildResultsCard() {
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(color: Colors.red.shade200),
      ),
      color: Colors.red.shade50,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.warning_amber, color: Colors.red.shade700, size: 20),
                const SizedBox(width: 8),
                Text(
                  '${_detections.length} Target(s) Detected',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    color: Colors.red.shade700,
                    fontSize: 16,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 4),
            Row(
              children: [
                Icon(Icons.vibration, size: 14, color: Colors.red.shade400),
                const SizedBox(width: 4),
                Icon(Icons.record_voice_over,
                    size: 14, color: Colors.red.shade400),
                const SizedBox(width: 6),
                Text(
                  'Alerts sent!',
                  style: TextStyle(fontSize: 12, color: Colors.red.shade400),
                ),
              ],
            ),
            const SizedBox(height: 12),
            ..._detections.asMap().entries.map((entry) {
              final det = entry.value;
              final idx = entry.key + 1;
              return Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Row(
                  children: [
                    Container(
                      width: 8,
                      height: 8,
                      decoration: const BoxDecoration(
                        color: Colors.red,
                        shape: BoxShape.circle,
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        'Target #$idx — ${(det.confidence * 100).toStringAsFixed(1)}%',
                        style:
                            TextStyle(fontSize: 13, color: Colors.red.shade800),
                      ),
                    ),
                    SizedBox(
                      width: 100,
                      child: LinearProgressIndicator(
                        value: det.confidence,
                        backgroundColor: Colors.red.shade100,
                        valueColor:
                            AlwaysStoppedAnimation<Color>(Colors.red.shade700),
                        borderRadius: BorderRadius.circular(4),
                      ),
                    ),
                  ],
                ),
              );
            }),
          ],
        ),
      ),
    );
  }

  Widget _buildActionButtons() {
    if (_autoCaptureEnabled) {
      return Row(
        children: [
          Expanded(
            child: ElevatedButton.icon(
              onPressed: _busy || !_cameraReady ? null : _captureAndDetect,
              icon: const Icon(Icons.camera),
              label: const Padding(
                padding: EdgeInsets.symmetric(vertical: 12),
                child: Text('Capture Now', style: TextStyle(fontSize: 16)),
              ),
              style: ElevatedButton.styleFrom(
                foregroundColor: Colors.white,
                backgroundColor: Theme.of(context).colorScheme.primary,
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12)),
              ),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: OutlinedButton.icon(
              onPressed: _busy ? null : () => _pickImage(ImageSource.gallery),
              icon: const Icon(Icons.photo_library_outlined),
              label: const Padding(
                padding: EdgeInsets.symmetric(vertical: 12),
                child: Text('Gallery', style: TextStyle(fontSize: 16)),
              ),
              style: OutlinedButton.styleFrom(
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12)),
              ),
            ),
          ),
        ],
      );
    }

    return Row(
      children: [
        Expanded(
          child: ElevatedButton.icon(
            onPressed: _busy ? null : () => _pickImage(ImageSource.camera),
            icon: const Icon(Icons.photo_camera),
            label: const Padding(
              padding: EdgeInsets.symmetric(vertical: 12),
              child: Text('Camera', style: TextStyle(fontSize: 16)),
            ),
            style: ElevatedButton.styleFrom(
              foregroundColor: Colors.white,
              backgroundColor: Theme.of(context).colorScheme.primary,
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12)),
            ),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: OutlinedButton.icon(
            onPressed: _busy ? null : () => _pickImage(ImageSource.gallery),
            icon: const Icon(Icons.photo_library_outlined),
            label: const Padding(
              padding: EdgeInsets.symmetric(vertical: 12),
              child: Text('Gallery', style: TextStyle(fontSize: 16)),
            ),
            style: OutlinedButton.styleFrom(
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12)),
            ),
          ),
        ),
      ],
    );
  }

  void _showSettings() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setSheetState) => Padding(
          padding: EdgeInsets.only(
            bottom: MediaQuery.of(ctx).viewInsets.bottom,
          ),
          child: Container(
            padding: const EdgeInsets.all(24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Center(
                  child: Container(
                    width: 40,
                    height: 4,
                    decoration: BoxDecoration(
                      color: Colors.grey.shade300,
                      borderRadius: BorderRadius.circular(2),
                    ),
                  ),
                ),
                const SizedBox(height: 20),
                const Text(
                  '⚙️ Settings',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 20),
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.grey.shade100,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Column(
                    children: [
                      _buildSwitch(
                        'Automatic Detection',
                        'Capture and detect every ${autoCaptureInterval.inSeconds}s',
                        _autoCaptureEnabled,
                        (val) {
                          setSheetState(() => _autoCaptureEnabled = val);
                          _setAutoCaptureEnabled(val);
                        },
                      ),
                      const Divider(height: 1),
                      _buildSwitch(
                        '📳 Vibration Alert',
                        'Phone vibrates on detection',
                        _vibrationEnabled,
                        (val) {
                          setSheetState(() => _vibrationEnabled = val);
                          setState(() => _vibrationEnabled = val);
                        },
                      ),
                      const Divider(height: 1),
                      _buildSwitch(
                        '🔊 Voice Alert',
                        'Speaks warning message',
                        _voiceEnabled,
                        (val) {
                          setSheetState(() => _voiceEnabled = val);
                          setState(() => _voiceEnabled = val);
                        },
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 20),
                Text(
                  'Confidence: ${_confidence.toStringAsFixed(2)}',
                  style: const TextStyle(fontWeight: FontWeight.w600),
                ),
                Slider(
                  value: _confidence,
                  min: 0.1,
                  max: 0.9,
                  divisions: 16,
                  onChanged: (val) {
                    setSheetState(() => _confidence = val);
                    setState(() => _confidence = val);
                  },
                ),
                Text(
                  'Low = more detections | High = fewer but more confident',
                  style: TextStyle(fontSize: 11, color: Colors.grey.shade600),
                ),
                const SizedBox(height: 16),
                Text(
                  'IoU Threshold: ${_iouThreshold.toStringAsFixed(2)}',
                  style: const TextStyle(fontWeight: FontWeight.w600),
                ),
                Slider(
                  value: _iouThreshold,
                  min: 0.3,
                  max: 0.9,
                  divisions: 12,
                  onChanged: (val) {
                    setSheetState(() => _iouThreshold = val);
                    setState(() => _iouThreshold = val);
                  },
                ),
                const SizedBox(height: 24),
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.grey.shade100,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('📊 Model',
                          style: TextStyle(fontWeight: FontWeight.w600)),
                      const SizedBox(height: 8),
                      _infoRow('Architecture', 'YOLOv8n'),
                      _infoRow('mAP50', '0.914'),
                      _infoRow('Dataset', '1,233 images'),
                      _infoRow('Input', '640×640'),
                      _infoRow('Format', 'ONNX'),
                    ],
                  ),
                ),
                const SizedBox(height: 24),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildSwitch(
      String title, String subtitle, bool value, ValueChanged<bool> onChanged) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title,
                    style: const TextStyle(fontWeight: FontWeight.w500)),
                Text(subtitle,
                    style:
                        TextStyle(fontSize: 12, color: Colors.grey.shade600)),
              ],
            ),
          ),
          Switch(value: value, onChanged: onChanged),
        ],
      ),
    );
  }

  Widget _infoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label,
              style: TextStyle(color: Colors.grey.shade700, fontSize: 13)),
          Text(value,
              style:
                  const TextStyle(fontWeight: FontWeight.w500, fontSize: 13)),
        ],
      ),
    );
  }
}

class DetectionResult {
  final double x1, y1, x2, y2;
  final double confidence;
  final int classId;
  final String className;

  DetectionResult({
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
    required this.confidence,
    required this.classId,
    required this.className,
  });
}

class DetectionBoxPainter extends CustomPainter {
  final List<DetectionResult> detections;
  final Size imageSize;
  final Size displaySize;

  DetectionBoxPainter({
    required this.detections,
    required this.imageSize,
    required this.displaySize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    const modelInputSize = 640.0;
    final resizeScale = min(
      modelInputSize / imageSize.width,
      modelInputSize / imageSize.height,
    );
    final resizedWidth = imageSize.width * resizeScale;
    final resizedHeight = imageSize.height * resizeScale;
    final padX = (modelInputSize - resizedWidth) / 2;
    final padY = (modelInputSize - resizedHeight) / 2;
    final scaleX = size.width / imageSize.width;
    final scaleY = size.height / imageSize.height;

    final linePaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..color = Colors.red;

    final bgPaint = Paint()..color = Colors.red.withAlpha(38);

    for (final det in detections) {
      final imageX1 =
          ((det.x1 - padX) / resizeScale).clamp(0.0, imageSize.width);
      final imageY1 =
          ((det.y1 - padY) / resizeScale).clamp(0.0, imageSize.height);
      final imageX2 =
          ((det.x2 - padX) / resizeScale).clamp(0.0, imageSize.width);
      final imageY2 =
          ((det.y2 - padY) / resizeScale).clamp(0.0, imageSize.height);
      final rect = Rect.fromLTRB(
        imageX1 * scaleX,
        imageY1 * scaleY,
        imageX2 * scaleX,
        imageY2 * scaleY,
      );

      canvas.drawRect(rect, bgPaint);
      canvas.drawRect(rect, linePaint);

      final label =
          '${det.className} ${(det.confidence * 100).toStringAsFixed(0)}%';
      final textPainter = TextPainter(
        text: TextSpan(
          text: label,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 12,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      final labelRect = Rect.fromLTWH(
        rect.left,
        max(0.0, rect.top - textPainter.height - 4),
        textPainter.width + 8,
        textPainter.height + 4,
      );

      canvas.drawRect(labelRect, linePaint);
      textPainter.paint(canvas, Offset(labelRect.left + 4, labelRect.top + 2));
    }
  }

  @override
  bool shouldRepaint(covariant DetectionBoxPainter old) =>
      old.detections != detections;
}
