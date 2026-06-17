"""Dependency-free ONNX inference core for the omni-detector.

This package contains the pure inference logic extracted from the Viam
``OnnxVisionService``. It has no dependency on viam or torch — only
onnxruntime, numpy, pillow, and opencv (cv2).

Public API:
    OmniOnnxInference — loads an ONNX detection model and runs inference.
    RawDetection      — a single detection with normalized [0,1] coordinates.
"""

from omni_inference.core import OmniOnnxInference, RawDetection

__all__ = ["OmniOnnxInference", "RawDetection"]
