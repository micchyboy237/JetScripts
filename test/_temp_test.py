# import numpy as np
# print(np.__config__.show(mode='dicts'))

import json
import time
import numpy as np
from typing import List, Optional
import logging
import subprocess
import sys
import argparse
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx
import signal

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info(
    f"Numpy config:\n{json.dumps(np.__config__.show(mode='dicts'), indent=2)}")


class TimeoutException(Exception):
    """Custom exception for timeout handling."""
    pass


def timeout_handler(signum, frame):
    """Raise TimeoutException on signal."""
    raise TimeoutException("Operation timed out")


def check_accelerate_usage() -> dict:
    """Check BLAS configuration for NumPy."""
    config = np.__config__.show(mode='dicts')
    blas_info = config.get('Build Dependencies', {}).get('blas', {})
    logger.debug(f"BLAS Info: {blas_info}")
    return blas_info


def list_micromamba_envs() -> List[str]:
    """List available micromamba environments."""
    try:
        result = subprocess.run(
            ['micromamba', 'env', 'list'], capture_output=True, text=True, check=True)
        lines = result.stdout.splitlines()
        envs = [line.split()[0] for line in lines[2:] if line.strip()
                and not line.startswith('─')]
        logger.debug(f"Micromamba environments: {envs}")
        return envs
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to list micromamba environments: {e}")
        return []


def matrix_multiply_test(size: int = 100) -> tuple[Optional[float], Optional[float]]:
    """Test matrix multiplication performance with and without BLAS (Accelerate)."""
    logger.debug(f"Matrix size: {size}x{size}")

    # Create random matrices
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)

    # BLAS-enabled matrix multiplication (should use Accelerate)
    logger.debug("Running BLAS-enabled matrix multiplication...")
    start_time = time.time()
    _ = np.matmul(A, B)
    blas_time = time.time() - start_time
    logger.debug(f"BLAS matrix multiply completed in {blas_time:.2f}s")

    # Non-BLAS simulation with timeout
    logger.debug("Running non-BLAS matrix multiplication...")
    non_blas_time = None
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)  # 10-second timeout
        start_time = time.time()
        C = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    C[i, j] += A[i, k] * B[k, j]
        non_blas_time = time.time() - start_time
        logger.debug(
            f"Non-BLAS matrix multiply completed in {non_blas_time:.2f}s")
    except TimeoutException:
        logger.warning(
            f"Non-BLAS matrix multiplication timed out after 10 seconds")
    finally:
        signal.alarm(0)  # Disable timeout

    logger.info(
        f"BLAS matrix multiply: {blas_time:.2f}s | Non-BLAS: {non_blas_time if non_blas_time is not None else 'timeout'}s")
    return blas_time, non_blas_time


def run_inference(model, tokenizer, prompts: List[str], max_tokens: int = 100, temp: float = 0.0) -> tuple[List[str], float]:
    """Run inference on a list of prompts and measure execution time."""
    logger.debug("Starting prompt generation...")
    start_time = time.time()
    sampler = make_sampler(temp)
    responses = [generate(model, tokenizer, prompt=p,
                          max_tokens=max_tokens, sampler=sampler) for p in prompts]
    execution_time = time.time() - start_time
    logger.info(f"Completed prompt generation in {execution_time:.2f}s")
    return responses, execution_time


def main(micromamba_env: Optional[str] = None) -> None:
    """Run inference and BLAS tests, optionally in a micromamba environment."""
    # Check if running in micromamba environment
    if micromamba_env:
        if micromamba_env not in list_micromamba_envs():
            logger.error(
                f"Micromamba environment '{micromamba_env}' not found.")
            sys.exit(1)
        logger.info(f"Running in micromamba environment: {micromamba_env}")
        cmd = ['micromamba', 'run', '-n', micromamba_env, 'python', __file__]
        result = subprocess.run(cmd, capture_output=True, text=True)
        logger.debug(f"Micromamba run output: {result.stdout}")
        if result.stderr:
            logger.error(f"Micromamba run error: {result.stderr}")
        sys.exit(result.returncode)

    # Check BLAS configuration
    blas_info = check_accelerate_usage()
    if blas_info.get('name') != 'accelerate':
        logger.warning(
            f"NumPy is not using Accelerate framework (found: {blas_info.get('name')}). "
            "Run 'micromamba create -n mlx_accelerate python=3.9 numpy -c apple' to enable Accelerate."
        )
    else:
        logger.info(
            "NumPy is using Accelerate framework. Optimal performance expected.")

    # Run matrix multiplication test
    logger.info("Running matrix multiplication test...")
    blas_time, non_blas_time = matrix_multiply_test()

    # Load model
    logger.debug("Loading model...")
    model, tokenizer = load("mlx-community/Qwen3-1.7B-4bit-DWQ-053125")

    # Batch of prompts
    prompts = ["Write a story about Einstein",
               "Explain quantum mechanics", "Summarize relativity"]

    # Run inference on CPU
    logger.info("Running inference with CPU-only settings...")
    try:
        mx.set_default_device(mx.cpu)
        responses_cpu, cpu_time = run_inference(model, tokenizer, prompts)
        logger.info(f"CPU execution time: {cpu_time:.2f}s")
    except Exception as e:
        logger.error(f"CPU inference failed: {e}")
        responses_cpu, cpu_time = None, None

    # Run inference on GPU
    logger.info("Running inference with GPU settings...")
    try:
        mx.set_default_device(mx.gpu)
        responses_gpu, gpu_time = run_inference(model, tokenizer, prompts)
        logger.info(f"GPU execution time: {gpu_time:.2f}s")
    except Exception as e:
        logger.error(f"GPU inference failed: {e}")
        responses_gpu, gpu_time = None, None

    # Final comparison
    if cpu_time is not None and gpu_time is not None:
        logger.info(f"CPU time: {cpu_time:.2f}s | GPU time: {gpu_time:.2f}s")
        logger.info(
            f"Matrix multiply - BLAS: {blas_time:.2f}s | Non-BLAS: {non_blas_time if non_blas_time is not None else 'timeout'}s")
    elif cpu_time is not None:
        logger.info(f"Only CPU inference succeeded: {cpu_time:.2f}s")
    elif gpu_time is not None:
        logger.info(f"Only GPU inference succeeded: {gpu_time:.2f}s")
    else:
        logger.error("Both CPU and GPU inference failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with optional micromamba environment.")
    parser.add_argument("--micromamba-env", type=str,
                        help="Name of micromamba environment to use")
    args = parser.parse_args()
    main(micromamba_env=args.micromamba_env)
