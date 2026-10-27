# ONNX Vision Service Module

A Viam vision service that runs object detection using an ONNX model exported by this training pipeline. Uses only `onnxruntime` — no PyTorch needed at inference time.

## Build

```bash
bash src/onnx_vision_service/build.sh
```

This creates a standalone executable at `dist/onnx-vision-service`.

You can also install just the vision-service dependencies into an existing env:

```bash
pip install ".[vision-service]"
```

## Viam Config

Add the module and service to your machine config:

```json
{
  "modules": [
    {
      "executable_path": "/path/to/dist/onnx-vision-service",
      "name": "onnx-module",
      "type": "local"
    }
  ],
  "services": [
    {
      "name": "my-detector",
      "type": "vision",
      "namespace": "rdk",
      "model": "viam:vision:onnx-detector",
      "attributes": {
        "model_path": "/path/to/model.onnx",
        "camera_name": "cam",
        "labels_path": "/path/to/labels.txt",
        "min_confidence": 0.3
      }
    }
  ]
}
```

### Attributes

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `model_path` | string | yes | Path to the ONNX model file |
| `camera_name` | string | yes | Name of the camera component |
| `labels_path` | string | yes | Path to `labels.txt` (one class name per line, line number = class index) |
| `min_confidence` | float | no | Minimum confidence threshold (default: 0.0) |
| `execution_provider` | string | no | onnxruntime execution provider: `cpu` (default) or `openvino` (Intel OpenVINO via the `onnxruntime-openvino` package; falls back to CPU if the provider is not available, check the startup log for the active providers) |
| `openvino_device` | string | no | OpenVINO device when `execution_provider` is `openvino`: `CPU` (default), `GPU`, `NPU`, `AUTO`, `HETERO:GPU,CPU`. The device must be able to compile the model; a failed compile fails the service with the OpenVINO error |

#### OpenVINO execution provider

With `"execution_provider": "openvino"` the same ONNX file runs through Intel's OpenVINO runtime
instead of onnxruntime's own CPU kernels; pre- and post-processing are unchanged. On the Kongsberg
vessel PC (Intel CPU, Windows) this cut the omni-detector round trip from 1237 ms to 840 ms p50 and
let fish-predictor hold its 1 Hz loop. The Linux and Windows builds install `onnxruntime-openvino`
(Windows additionally needs the `openvino` package, both in the `vision-service` extra); macOS has no
OpenVINO build and keeps plain `onnxruntime`, where requesting `openvino` logs a warning and runs on CPU.

### ONNX Model Contract

The module expects models exported by `convert_to_onnx.py` with:

- **Input**: `image` — uint8 `[1, C, H, W]`, range `[0, 255]`
- **Outputs**: `location` `[N, 4]`, `score` `[N]`, `category` `[N]` (all float32)

Input size is auto-detected from the ONNX model metadata. Images are resized with PIL before inference.

## API

The service implements `GetDetections`, `GetDetectionsFromCamera`, and `CaptureAllFromCamera`.
