"""Export PyTorch segmentation model to ONNX format.

Exports three precision variants:
  - model_fp32.onnx: Original FP32 export (unchanged)
  - model_fp16.onnx: FP16 mixed-precision via modelopt AutoCast
  - model_int8.onnx: INT8 with Q/DQ nodes via modelopt PTQ (produced by calibrate.py)

Each variant is verified with onnxruntime before proceeding.

TensorRT 11.x notes:
  Precision is now driven by the ONNX graph itself, not by builder flags.
  FP16 and INT8 variants must be separate ONNX files with the appropriate
  precision annotations baked in.
"""

from __future__ import annotations

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple

import numpy as np
import torch
import yaml

# Ensure project root is on path
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.build import build_model_for_export


def get_git_sha() -> str:
    """Get short git SHA of current commit."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_gpu_info() -> dict:
    """Get GPU name and driver version via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        parts = result.stdout.strip().split(", ")
        return {"gpu_name": parts[0], "driver_version": parts[1]}
    except Exception:
        return {"gpu_name": "unknown", "driver_version": "unknown"}


def load_config(config_path: str = "configs/export.yaml") -> dict:
    """Load export configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_size: Tuple[int, int] = (256, 256),
    batch_size: int = 4,
    opset_version: int = 17,
    input_name: str = "input",
    output_name: str = "output",
) -> str:
    """Export model to ONNX format.

    Args:
        model: PyTorch model in eval mode.
        output_path: Path to save the ONNX model.
        input_size: (H, W) input dimensions.
        batch_size: Fixed batch size.
        opset_version: ONNX opset version.
        input_name: Name for the input tensor.
        output_name: Name for the output tensor.

    Returns:
        Path to the saved ONNX model.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    device = next(model.parameters()).device
    dummy_input = torch.randn(batch_size, 3, *input_size, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes=None,  # fixed shape
    )
    print(f"ONNX model saved to {output_path}")
    return output_path


def convert_to_fp16(
    onnx_path: str,
    output_path: str,
    keep_io_types: bool = True,
) -> str:
    """Convert FP32 ONNX model to FP16 mixed-precision using modelopt AutoCast.

    Uses modelopt.onnx.autocast.convert_to_mixed_precision() to intelligently
    select which nodes to keep in FP32 for accuracy, while converting the
    rest to FP16.

    Args:
        onnx_path: Path to the FP32 ONNX model.
        output_path: Path to save the FP16 model.
        keep_io_types: If True, preserve input/output tensor types as FP32.

    Returns:
        Path to the saved FP16 ONNX model.
    """
    from modelopt.onnx.autocast import convert_to_mixed_precision

    print(f"Converting to FP16 mixed-precision: {onnx_path} -> {output_path}")

    converted_model = convert_to_mixed_precision(
        onnx_path=onnx_path,
        low_precision_type="fp16",
        keep_io_types=keep_io_types,
        providers=["cpu"],
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    import onnx
    onnx.save(converted_model, output_path)
    print(f"FP16 ONNX model saved to {output_path}")
    return output_path


def verify_onnx(
    onnx_path: str,
    pytorch_model: torch.nn.Module,
    input_size: Tuple[int, int] = (256, 256),
    batch_size: int = 4,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """Verify ONNX model output matches PyTorch reference.

    Uses onnxruntime for inference and compares against PyTorch output.

    Args:
        onnx_path: Path to the ONNX model.
        pytorch_model: Original PyTorch model for comparison.
        input_size: (H, W) input dimensions.
        batch_size: Batch size for verification.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.

    Returns:
        True if outputs match within tolerance.
    """
    import onnxruntime as ort

    # PyTorch reference output
    device = next(pytorch_model.parameters()).device
    dummy_input = torch.randn(batch_size, 3, *input_size, device=device)
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input)["out"].cpu().numpy()

    # ONNX Runtime output
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]

    # Compare
    max_diff = np.max(np.abs(pytorch_output - ort_output))
    mean_diff = np.mean(np.abs(pytorch_output - ort_output))
    close = np.allclose(pytorch_output, ort_output, rtol=rtol, atol=atol)

    print(f"ONNX verification: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
          f"allclose={close}")

    if not close:
        print("WARNING: ONNX output differs from PyTorch reference!")
        return False

    print("ONNX verification PASSED")
    return True


def verify_onnx_loadable(onnx_path: str) -> bool:
    """Verify that an ONNX model loads correctly with onnxruntime.

    Args:
        onnx_path: Path to the ONNX model.

    Returns:
        True if model loads successfully.
    """
    import onnxruntime as ort

    try:
        session = ort.InferenceSession(onnx_path)
        input_info = session.get_inputs()[0]
        output_info = session.outputs[0]
        print(f"  Loaded {onnx_path}: input={input_info.name}{input_info.shape}, "
              f"output={output_info.name}{output_info.shape}")
        return True
    except Exception as e:
        print(f"  FAILED to load {onnx_path}: {e}")
        return False


def main() -> None:
    """Export model to ONNX variants and verify."""
    config = load_config()

    model_config = config["model"]
    onnx_config = config["onnx"]

    num_classes = model_config["num_classes"]
    input_size = tuple(model_config["input_size"])
    batch_size = model_config["batch_size"]
    opset_version = onnx_config["opset_version"]

    # Paths for each precision variant
    fp32_path = onnx_config["fp32_path"]
    fp16_path = onnx_config["fp16_path"]

    print(f"Building model with {num_classes} classes...")
    model = build_model_for_export(num_classes=num_classes)

    # === Step 1: Export FP32 ONNX ===
    print(f"\n=== Exporting FP32 ONNX (opset {opset_version}) ===")
    export_to_onnx(
        model=model,
        output_path=fp32_path,
        input_size=input_size,
        batch_size=batch_size,
        opset_version=opset_version,
        input_name=onnx_config["input_name"],
        output_name=onnx_config["output_name"],
    )

    print("\nVerifying FP32 ONNX against PyTorch reference...")
    fp32_ok = verify_onnx(
        onnx_path=fp32_path,
        pytorch_model=model,
        input_size=input_size,
        batch_size=batch_size,
    )

    # === Step 2: Convert to FP16 ===
    print(f"\n=== Converting to FP16 mixed-precision ===")
    convert_to_fp16(
        onnx_path=fp32_path,
        output_path=fp16_path,
        keep_io_types=True,
    )

    print("\nVerifying FP16 ONNX loads correctly...")
    fp16_loadable = verify_onnx_loadable(fp16_path)

    # Verify FP16 produces reasonable output (compare against FP32)
    print("\nVerifying FP16 ONNX output against PyTorch reference...")
    fp16_ok = verify_onnx(
        onnx_path=fp16_path,
        pytorch_model=model,
        input_size=input_size,
        batch_size=batch_size,
        rtol=1e-2,  # FP16 has lower precision
        atol=1e-3,
    )

    # === Summary ===
    print("\n=== Export Summary ===")
    print(f"  FP32: {fp32_path} (verified: {fp32_ok})")
    print(f"  FP16: {fp16_path} (loadable: {fp16_loadable}, verified: {fp16_ok})")
    print(f"  INT8: Will be produced by calibrate.py using modelopt PTQ")

    if not (fp32_ok and fp16_ok):
        print("\nWARNING: Some verifications failed!")
        sys.exit(1)

    # Write result JSON
    os.makedirs("results", exist_ok=True)
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "seed": 42,
        **get_gpu_info(),
        "onnx_paths": {
            "fp32": fp32_path,
            "fp16": fp16_path,
        },
        "opset_version": opset_version,
        "input_size": list(input_size),
        "batch_size": batch_size,
        "num_classes": num_classes,
        "verification": {
            "fp32_passed": fp32_ok,
            "fp16_loadable": fp16_loadable,
            "fp16_passed": fp16_ok,
        },
    }

    result_path = f"results/onnx_export_{get_git_sha()}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()
