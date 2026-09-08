"""Benchmark inference latency for PyTorch and TensorRT models.

Measures mean FPS and p95 latency under frozen conditions:
- Input: 256x256, batch 4
- 50 warmup + 500 timed iterations
- GPU name and driver version recorded
- Reports mean FPS and p95 latency in ms

Runs against a randomly-initialized model (no trained weights needed).
"""

from __future__ import annotations

import os
import sys
import time
import subprocess
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
import torch
import yaml

# Ensure project root is on path
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import tensorrt as trt
    import tensorrt_bindings
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

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
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def benchmark_pytorch(
    model: torch.nn.Module,
    batch_size: int = 4,
    input_size: Tuple[int, int] = (256, 256),
    warmup_iters: int = 50,
    timed_iters: int = 500,
    use_amp: bool = True,
) -> dict:
    """Benchmark PyTorch model inference latency.

    Args:
        model: PyTorch model in eval mode.
        batch_size: Batch size for inference.
        input_size: (H, W) input dimensions.
        warmup_iters: Number of warmup iterations.
        timed_iters: Number of timed iterations.
        use_amp: Whether to use automatic mixed precision.

    Returns:
        Dictionary with benchmark results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, *input_size, device=device)

    # Warmup
    print(f"  Warmup: {warmup_iters} iterations...")
    with torch.no_grad():
        for _ in range(warmup_iters):
            with torch.amp.autocast("cuda", enabled=use_amp):
                _ = model(dummy_input)
    torch.cuda.synchronize()

    # Timed iterations
    print(f"  Timed: {timed_iters} iterations...")
    latencies = []
    with torch.no_grad():
        for _ in range(timed_iters):
            start = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=use_amp):
                _ = model(dummy_input)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

    latencies = np.array(latencies)
    mean_latency_ms = float(np.mean(latencies))
    p95_latency_ms = float(np.percentile(latencies, 95))
    p99_latency_ms = float(np.percentile(latencies, 99))
    mean_fps = 1000.0 / mean_latency_ms if mean_latency_ms > 0 else 0.0

    return {
        "backend": "pytorch",
        "mean_fps": round(mean_fps, 2),
        "mean_latency_ms": round(mean_latency_ms, 3),
        "p95_latency_ms": round(p95_latency_ms, 3),
        "p99_latency_ms": round(p99_latency_ms, 3),
        "min_latency_ms": round(float(np.min(latencies)), 3),
        "max_latency_ms": round(float(np.max(latencies)), 3),
        "std_latency_ms": round(float(np.std(latencies)), 3),
        "warmup_iters": warmup_iters,
        "timed_iters": timed_iters,
        "batch_size": batch_size,
        "input_size": list(input_size),
        "use_amp": use_amp,
    }


def benchmark_onnx(
    onnx_path: str,
    batch_size: int = 4,
    input_size: Tuple[int, int] = (256, 256),
    warmup_iters: int = 50,
    timed_iters: int = 500,
) -> dict:
    """Benchmark ONNX Runtime inference latency.

    Args:
        onnx_path: Path to the ONNX model.
        batch_size: Batch size for inference.
        input_size: (H, W) input dimensions.
        warmup_iters: Number of warmup iterations.
        timed_iters: Number of timed iterations.

    Returns:
        Dictionary with benchmark results.
    """
    if not ORT_AVAILABLE:
        return {"backend": "onnxruntime", "error": "onnxruntime not available"}

    providers = [("CUDAExecutionProvider", {"device_id": 0})]
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name

    dummy_input = np.random.randn(batch_size, 3, *input_size).astype(np.float32)

    # Warmup
    print(f"  Warmup: {warmup_iters} iterations...")
    for _ in range(warmup_iters):
        _ = session.run(None, {input_name: dummy_input})

    # Timed iterations
    print(f"  Timed: {timed_iters} iterations...")
    latencies = []
    for _ in range(timed_iters):
        start = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    latencies = np.array(latencies)
    mean_latency_ms = float(np.mean(latencies))
    p95_latency_ms = float(np.percentile(latencies, 95))
    p99_latency_ms = float(np.percentile(latencies, 99))
    mean_fps = 1000.0 / mean_latency_ms if mean_latency_ms > 0 else 0.0

    return {
        "backend": "onnxruntime",
        "mean_fps": round(mean_fps, 2),
        "mean_latency_ms": round(mean_latency_ms, 3),
        "p95_latency_ms": round(p95_latency_ms, 3),
        "p99_latency_ms": round(p99_latency_ms, 3),
        "min_latency_ms": round(float(np.min(latencies)), 3),
        "max_latency_ms": round(float(np.max(latencies)), 3),
        "std_latency_ms": round(float(np.std(latencies)), 3),
        "warmup_iters": warmup_iters,
        "timed_iters": timed_iters,
        "batch_size": batch_size,
        "input_size": list(input_size),
    }


def benchmark_tensorrt(
    engine_path: str,
    batch_size: int = 4,
    input_size: Tuple[int, int] = (256, 256),
    warmup_iters: int = 50,
    timed_iters: int = 500,
) -> dict:
    """Benchmark TensorRT engine inference latency.

    Uses TensorRT 11.x API (set_input_shape, execute_async_v3).
    Uses cuda.bindings.driver for GPU memory management (no torch dependency).

    Args:
        engine_path: Path to the TensorRT engine.
        batch_size: Batch size for inference.
        input_size: (H, W) input dimensions.
        warmup_iters: Number of warmup iterations.
        timed_iters: Number of timed iterations.

    Returns:
        Dictionary with benchmark results.
    """
    if not TRT_AVAILABLE:
        return {"backend": "tensorrt", "error": "tensorrt not available"}

    if not os.path.exists(engine_path):
        return {"backend": "tensorrt", "error": f"engine not found: {engine_path}"}

    from cuda.bindings import driver as cudadr
    cudadr.cuInit(0)
    _, ctx = cudadr.cuCtxCreate(None, 0, 0)

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Identify input/output tensors
    input_names = []
    output_names = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)

    # Set input shape
    for name in input_names:
        context.set_input_shape(name, (batch_size, 3, *input_size))

    # Allocate host/device buffers using cuda.bindings.driver
    _, stream = cudadr.cuStreamCreate(0)
    host_inputs = {}
    device_inputs = {}
    host_outputs = {}
    device_outputs = {}

    for name in input_names:
        shape = context.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size = trt.volume(shape)
        host_inputs[name] = np.empty(size, dtype=dtype)
        _, dptr = cudadr.cuMemAlloc(host_inputs[name].nbytes)
        device_inputs[name] = dptr

    for name in output_names:
        shape = context.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size = trt.volume(shape)
        host_outputs[name] = np.empty(size, dtype=dtype)
        _, dptr = cudadr.cuMemAlloc(host_outputs[name].nbytes)
        device_outputs[name] = dptr

    dummy_input = np.random.randn(batch_size, 3, *input_size).astype(np.float32)

    def _run_once() -> None:
        """Execute one inference step."""
        for name in input_names:
            np.copyto(host_inputs[name], dummy_input.ravel())
            cudadr.cuMemcpyHtoDAsync(device_inputs[name], host_inputs[name].ctypes.data, host_inputs[name].nbytes, stream)
            context.set_tensor_address(name, int(device_inputs[name]))

        for name in output_names:
            context.set_tensor_address(name, int(device_outputs[name]))

        context.execute_async_v3(int(stream))
        cudadr.cuStreamSynchronize(stream)

    # Warmup
    print(f"  Warmup: {warmup_iters} iterations...")
    for _ in range(warmup_iters):
        _run_once()

    # Timed iterations
    print(f"  Timed: {timed_iters} iterations...")
    latencies = []
    for _ in range(timed_iters):
        start = time.perf_counter()
        _run_once()
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    latencies = np.array(latencies)
    mean_latency_ms = float(np.mean(latencies))
    p95_latency_ms = float(np.percentile(latencies, 95))
    p99_latency_ms = float(np.percentile(latencies, 99))
    mean_fps = 1000.0 / mean_latency_ms if mean_latency_ms > 0 else 0.0

    return {
        "backend": "tensorrt",
        "engine_path": engine_path,
        "mean_fps": round(mean_fps, 2),
        "mean_latency_ms": round(mean_latency_ms, 3),
        "p95_latency_ms": round(p95_latency_ms, 3),
        "p99_latency_ms": round(p99_latency_ms, 3),
        "min_latency_ms": round(float(np.min(latencies)), 3),
        "max_latency_ms": round(float(np.max(latencies)), 3),
        "std_latency_ms": round(float(np.std(latencies)), 3),
        "warmup_iters": warmup_iters,
        "timed_iters": timed_iters,
        "batch_size": batch_size,
        "input_size": list(input_size),
    }


def main() -> None:
    """Run latency benchmarks for all backends."""
    config = load_config()
    bench_config = config["benchmark"]
    model_config = config["model"]
    trt_config = config["tensorrt"]

    batch_size = bench_config["batch_size"]
    input_size = tuple(bench_config["input_size"])
    warmup_iters = bench_config["warmup_iters"]
    timed_iters = bench_config["timed_iters"]

    git_sha = get_git_sha()
    gpu_info = get_gpu_info()

    print(f"GPU: {gpu_info['gpu_name']} (Driver {gpu_info['driver_version']})")
    print(f"Input: {input_size}, Batch: {batch_size}")
    print(f"Warmup: {warmup_iters}, Timed: {timed_iters}\n")

    # Build model
    print("Building model...")
    model = build_model_for_export(num_classes=model_config["num_classes"])

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "seed": 42,
        **gpu_info,
        "benchmarks": {},
    }

    # 1. PyTorch benchmark (skip if CUDA not available)
    if torch.cuda.is_available():
        print("\n=== PyTorch Benchmark ===")
        pytorch_result = benchmark_pytorch(
            model=model,
            batch_size=batch_size,
            input_size=input_size,
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )
        results["benchmarks"]["pytorch"] = pytorch_result
        print(f"  Mean FPS: {pytorch_result['mean_fps']}")
        print(f"  P95 Latency: {pytorch_result['p95_latency_ms']} ms")
    else:
        print("\n=== PyTorch Benchmark === SKIPPED (CUDA not available)")

    # 2. ONNX Runtime benchmark (FP32)
    onnx_path = config["onnx"]["fp32_path"]
    if os.path.exists(onnx_path):
        print("\n=== ONNX Runtime Benchmark ===")
        ort_result = benchmark_onnx(
            onnx_path=onnx_path,
            batch_size=batch_size,
            input_size=input_size,
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )
        results["benchmarks"]["onnxruntime"] = ort_result
        if "mean_fps" in ort_result:
            print(f"  Mean FPS: {ort_result['mean_fps']}")
            print(f"  P95 Latency: {ort_result['p95_latency_ms']} ms")
    else:
        print(f"\nSkipping ONNX Runtime: {onnx_path} not found")

    # 3. TensorRT benchmarks
    for precision in ["fp32", "fp16", "int8"]:
        engine_path = trt_config[f"{precision}_path"]
        if os.path.exists(engine_path):
            print(f"\n=== TensorRT {precision.upper()} Benchmark ===")
            trt_result = benchmark_tensorrt(
                engine_path=engine_path,
                batch_size=batch_size,
                input_size=input_size,
                warmup_iters=warmup_iters,
                timed_iters=timed_iters,
            )
            results["benchmarks"][f"tensorrt_{precision}"] = trt_result
            if "mean_fps" in trt_result:
                print(f"  Mean FPS: {trt_result['mean_fps']}")
                print(f"  P95 Latency: {trt_result['p95_latency_ms']} ms")
        else:
            print(f"\nSkipping TensorRT {precision.upper()}: {engine_path} not found")

    # Save results
    os.makedirs(bench_config["results_dir"], exist_ok=True)
    result_path = os.path.join(bench_config["results_dir"], f"latency_{git_sha}.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {result_path}")

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Backend':<25} {'Mean FPS':>10} {'P95 Latency (ms)':>18} {'P99 Latency (ms)':>18}")
    print("-" * 75)
    for name, res in results["benchmarks"].items():
        if "mean_fps" in res:
            print(f"{name:<25} {res['mean_fps']:>10.2f} {res['p95_latency_ms']:>18.3f} {res['p99_latency_ms']:>18.3f}")
        else:
            print(f"{name:<25} {'ERROR':>10} {res.get('error', 'unknown'):>18}")


if __name__ == "__main__":
    main()
