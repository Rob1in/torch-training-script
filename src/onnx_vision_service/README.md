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

### ONNX Model Contract

The module expects models exported by `convert_to_onnx.py` with:

- **Input**: `image` — uint8 `[1, C, H, W]`, range `[0, 255]`
- **Outputs**: `location` `[N, 4]`, `score` `[N]`, `category` `[N]` (all float32)

Input size is auto-detected from the ONNX model metadata. Images are resized with PIL before inference.

## API

The service implements `GetDetections`, `GetDetectionsFromCamera`, and `CaptureAllFromCamera`.
