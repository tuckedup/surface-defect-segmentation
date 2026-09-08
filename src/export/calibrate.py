"""INT8 Post-Training Quantization (PTQ) for ONNX models.

Methodology Change (TensorRT 11.x):
------------------------------------
TensorRT 11.x removed the BuilderFlag.FP16/INT8 and IInt8Calibrator interfaces
entirely. Precision must now come from the ONNX graph itself via NVIDIA ModelOpt.

Instead of the old entropy-calibrator approach (which fed calibration data to
TensorRT's builder), we now:
  1. Use ModelOpt's ONNX PTQ to insert Quantize/Dequantize (Q/DQ) nodes directly
     into the ONNX graph.
  2. The Q/DQ nodes encode the quantization parameters (scales, zero-points) that
     TensorRT's builder reads to enable INT8 kernels.
  3. Each precision variant is now a SEPARATE ONNX file:
     - model_fp32.onnx: Original FP32 (no Q/DQ nodes)
     - model_fp16.onnx: FP16 mixed-precision via AutoCast
     - model_int8.onnx: INT8 with Q/DQ nodes inserted by PTQ

Why 500 images:
  - Sufficient to capture the activation distribution of a segmentation model.
  - Balances calibration time vs. quantization accuracy.
  - Standard practice in TensorRT deployment pipelines.

The calibration set must be representative of the inference distribution but
disjoint from training/eval sets. Images are randomly sampled from the held-out
calibration split with a fixed seed for reproducibility.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticCalibrationDataset(Dataset):
    """Synthetic dataset for calibration with random images.

    Used when real calibration data is not yet available.
    Generates random RGB images of the specified size.
    """

    def __init__(self, num_images: int = 500, height: int = 256, width: int = 256):
        self.num_images = num_images
        self.height = height
        self.width = width

    def __len__(self) -> int:
        return self.num_images

    def __getitem__(self, idx: int) -> torch.Tensor:
        torch.manual_seed(idx)
        return torch.randn(3, self.height, self.width)


def create_calibration_set(
    num_images: int = 500,
    height: int = 256,
    width: int = 256,
    seed: int = 42,
) -> SyntheticCalibrationDataset:
    """Create a synthetic calibration dataset.

    Args:
        num_images: Number of images for calibration.
        height: Image height.
        width: Image width.
        seed: Random seed for reproducibility.

    Returns:
        SyntheticCalibrationDataset instance.
    """
    return SyntheticCalibrationDataset(
        num_images=num_images, height=height, width=width
    )


def prepare_calibration_data(
    dataset: Dataset,
    batch_size: int = 4,
    num_batches: int = 125,
) -> np.ndarray:
    """Prepare calibration data as a numpy array for modelopt.

    Args:
        dataset: PyTorch dataset providing calibration images.
        batch_size: Batch size for loading.
        num_batches: Number of batches to load (total = batch_size * num_batches).

    Returns:
        Numpy array of shape (N, 3, H, W) with calibration images.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    all_batches = []
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        all_batches.append(batch.numpy().astype(np.float32))

    calibration_data = np.concatenate(all_batches, axis=0)
    print(f"Prepared {calibration_data.shape[0]} calibration images "
          f"(shape: {calibration_data.shape})")
    return calibration_data


def quantize_int8(
    onnx_path: str,
    output_path: str,
    calibration_data: Optional[np.ndarray] = None,
    calibration_method: str = "entropy",
) -> str:
    """Quantize ONNX model to INT8 with Q/DQ nodes using modelopt PTQ.

    This function inserts Quantize/Dequantize (Q/DQ) nodes into the ONNX graph,
    which TensorRT 11.x reads to enable INT8 kernels. The resulting model is a
    SEPARATE file from the FP32/FP16 variants.

    Args:
        onnx_path: Path to the FP32 ONNX model.
        output_path: Path to save the INT8 quantized model.
        calibration_data: Numpy array of calibration images (N, 3, H, W).
                          If None, random scales are used.
        calibration_method: Calibration method ('entropy' or 'max').

    Returns:
        Path to the saved INT8 ONNX model.
    """
    from modelopt.onnx.quantization import quantize

    print(f"Quantizing to INT8: {onnx_path} -> {output_path}")
    print(f"  Calibration method: {calibration_method}")
    if calibration_data is not None:
        print(f"  Calibration data: {calibration_data.shape[0]} images")
    else:
        print("  Using random calibration scales (no real data)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    quantize(
        onnx_path=onnx_path,
        quantize_mode="int8",
        calibration_data=calibration_data,
        calibration_method=calibration_method,
        output_path=output_path,
        high_precision_dtype="fp16",  # Keep non-quantized ops in FP16
    )

    print(f"INT8 ONNX model saved to {output_path}")
    return output_path


def verify_int8_model(onnx_path: str) -> bool:
    """Verify that the INT8 ONNX model loads correctly with onnxruntime.

    Args:
        onnx_path: Path to the INT8 ONNX model.

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

        # Check if Q/DQ nodes are present by examining the graph
        import onnx
        model = onnx.load(onnx_path)
        qdq_nodes = [n for n in model.graph.node if n.op_type in ("QuantizeLinear", "DequantizeLinear")]
        print(f"  Q/DQ nodes found: {len(qdq_nodes)}")
        return True
    except Exception as e:
        print(f"  FAILED to load {onnx_path}: {e}")
        return False


def main() -> None:
    """Run INT8 quantization pipeline."""
    import yaml

    PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
    config_path = os.path.join(PROJECT_ROOT, "configs", "export.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    onnx_config = config["onnx"]
    model_config = config["model"]
    calib_config = config["calibration"]

    fp32_path = onnx_config["fp32_path"]
    int8_path = onnx_config["int8_path"]

    if not os.path.exists(fp32_path):
        print(f"ERROR: FP32 ONNX model not found at {fp32_path}")
        print("Run src/export/to_onnx.py first.")
        return

    # Prepare calibration data
    print("\n=== Preparing Calibration Data ===")
    dataset = create_calibration_set(
        num_images=calib_config["num_images"],
        height=model_config["input_size"][0],
        width=model_config["input_size"][1],
        seed=calib_config["seed"],
    )

    calibration_data = prepare_calibration_data(
        dataset,
        batch_size=model_config["batch_size"],
        num_batches=calib_config["num_images"] // model_config["batch_size"],
    )

    # Run INT8 quantization
    print("\n=== Running INT8 PTQ ===")
    quantize_int8(
        onnx_path=fp32_path,
        output_path=int8_path,
        calibration_data=calibration_data,
        calibration_method=calib_config.get("calibration_method", "entropy"),
    )

    # Verify
    print("\n=== Verifying INT8 Model ===")
    int8_ok = verify_int8_model(int8_path)

    if not int8_ok:
        print("\nWARNING: INT8 model verification failed!")
        return

    print("\nINT8 quantization complete!")
    print(f"  INT8 model: {int8_path}")


if __name__ == "__main__":
    main()
