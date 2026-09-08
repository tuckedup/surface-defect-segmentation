"""Build TensorRT engines from ONNX models.

TensorRT 11.x changes:
  - BuilderFlag.FP16 and BuilderFlag.INT8 no longer exist.
  - IInt8Calibrator interface has been removed.
  - Precision is now driven by the ONNX graph itself, not by builder flags.
  - Each precision variant must be a SEPARATE ONNX file:
      - model_fp32.onnx: Original FP32
      - model_fp16.onnx: FP16 mixed-precision (via modelopt AutoCast)
      - model_int8.onnx: INT8 with Q/DQ nodes (via modelopt PTQ)
  - TRT 11.x reads the ONNX graph's tensor types to determine precision.
  - builder.create_network() with no flags uses strong typing by default.
"""

from __future__ import annotations

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime, timezone

import yaml

# Ensure project root is on path
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("WARNING: TensorRT not available. Engine building will be skipped.")


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


def build_engine(
    onnx_path: str,
    engine_path: str,
    workspace_size_gb: int = 4,
    label: str = "engine",
) -> str:
    """Build a TensorRT engine from an ONNX model.

    In TRT 11.x, the ONNX graph's own tensor types drive precision.
    No builder flags are needed - strong typing is the default.
    The builder reads Q/DQ nodes (for INT8) or cast nodes (for FP16)
    directly from the ONNX graph.

    Args:
        onnx_path: Path to the ONNX model (must already have precision baked in).
        engine_path: Path to save the engine.
        workspace_size_gb: Maximum workspace size in GB.
        label: Human-readable label for log messages.

    Returns:
        Path to the saved engine.
    """
    if not TRT_AVAILABLE:
        raise RuntimeError("TensorRT is not installed.")

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # create_network() with no flags uses strong typing (TRT 11.x default).
    # The ONNX parser reads the graph's tensor types to determine precision.
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"ONNX Parse Error: {parser.get_error(error)}")
            raise RuntimeError(f"Failed to parse ONNX model: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_size_gb * (1 << 30)
    )

    print(f"Building {label} from {onnx_path}...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(f"TensorRT failed to build {label}")

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    print(f"{label} saved to {engine_path}")
    return engine_path


def load_engine(engine_path: str) -> "trt.ICudaEngine":
    """Load a serialized TensorRT engine."""
    if not TRT_AVAILABLE:
        raise RuntimeError("TensorRT is not installed.")

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    print(f"Engine loaded from {engine_path}")
    return engine


def main() -> None:
    """Build all TensorRT engine variants from their respective ONNX files."""
    config_path = os.path.join(PROJECT_ROOT, "configs", "export.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    onnx_config = config["onnx"]
    trt_config = config["tensorrt"]
    model_config = config["model"]
    calib_config = config["calibration"]

    # Each precision has its OWN ONNX source file
    fp32_onnx = onnx_config["fp32_path"]
    fp16_onnx = onnx_config["fp16_path"]
    int8_onnx = onnx_config["int8_path"]

    # Verify all required ONNX files exist
    missing = []
    for name, path in [("FP32", fp32_onnx), ("FP16", fp16_onnx), ("INT8", int8_onnx)]:
        if not os.path.exists(path):
            missing.append(f"  {name}: {path}")

    if missing:
        print("ERROR: Required ONNX models not found:")
        for m in missing:
            print(m)
        print("\nRun src/export/to_onnx.py and src/export/calibrate.py first.")
        sys.exit(1)

    # Build FP32 engine
    print("\n=== Building FP32 Engine ===")
    build_engine(
        onnx_path=fp32_onnx,
        engine_path=trt_config["fp32_path"],
        workspace_size_gb=trt_config["workspace_size_gb"],
        label="FP32 engine",
    )

    # Build FP16 engine (from FP16 ONNX with cast nodes)
    print("\n=== Building FP16 Engine ===")
    build_engine(
        onnx_path=fp16_onnx,
        engine_path=trt_config["fp16_path"],
        workspace_size_gb=trt_config["workspace_size_gb"],
        label="FP16 engine",
    )

    # Build INT8 engine (from INT8 ONNX with Q/DQ nodes)
    print("\n=== Building INT8 Engine ===")
    build_engine(
        onnx_path=int8_onnx,
        engine_path=trt_config["int8_path"],
        workspace_size_gb=trt_config["workspace_size_gb"],
        label="INT8 engine",
    )

    # Write result JSON
    os.makedirs("results", exist_ok=True)
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "seed": calib_config["seed"],
        **get_gpu_info(),
        "engines": {
            "fp32": trt_config["fp32_path"],
            "fp16": trt_config["fp16_path"],
            "int8": trt_config["int8_path"],
        },
        "onnx_sources": {
            "fp32": fp32_onnx,
            "fp16": fp16_onnx,
            "int8": int8_onnx,
        },
        "input_size": model_config["input_size"],
        "batch_size": model_config["batch_size"],
        "note": "Each engine built from its own ONNX file with precision baked in (TRT 11.x)",
    }

    result_path = f"results/trt_build_{get_git_sha()}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()
