#!/usr/bin/env python3
"""
快速检查 CUDA 和 PyTorch 支持情况
用于诊断 ARM 架构上的 CUDA 支持问题
"""

import sys
import os
import ctypes

def check_cuda_libraries():
    """检查 CUDA 库是否存在且可加载"""
    print("=" * 60)
    print("CUDA Library Check")
    print("=" * 60)
    
    cuda_libs = [
        "/usr/lib/aarch64-linux-gnu/libcudart.so.12",
        "/usr/lib/aarch64-linux-gnu/libcudart.so.12.8.57",
        "/usr/local/cuda/lib64/libcudart.so",
    ]
    
    found = False
    for lib_path in cuda_libs:
        if os.path.exists(lib_path):
            try:
                lib = ctypes.CDLL(lib_path)
                print(f"✓ Found and loadable: {lib_path}")
                found = True
            except Exception as e:
                print(f"✗ Found but not loadable: {lib_path} ({e})")
        else:
            print(f"✗ Not found: {lib_path}")
    
    return found


def check_pytorch_cuda():
    """检查 PyTorch CUDA 支持"""
    print("\n" + "=" * 60)
    print("PyTorch CUDA Support Check")
    print("=" * 60)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA compiled: {torch.version.cuda}")
        print(f"cuDNN available: {torch.backends.cudnn.is_available()}")
        
        if torch.cuda.is_available():
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
                print(f"    Compute Capability: {props.major}.{props.minor}")
            return True
        else:
            print("\n⚠️  PyTorch does not have CUDA support")
            if "+cpu" in torch.__version__:
                print("   This is a CPU-only build")
            if torch.version.cuda is None:
                print("   PyTorch was not compiled with CUDA")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False


def check_system_cuda():
    """检查系统 CUDA 信息"""
    print("\n" + "=" * 60)
    print("System CUDA Check")
    print("=" * 60)
    
    # Check nvidia-smi
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,compute_cap', '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("GPU Information:")
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")
        else:
            print("✗ nvidia-smi failed")
    except Exception as e:
        print(f"✗ Cannot run nvidia-smi: {e}")
    
    # Check CUDA paths
    cuda_paths = [
        "/usr/local/cuda",
        "/usr/lib/cuda",
        "/usr/local/cuda-12.8",
    ]
    print("\nCUDA Installation Paths:")
    for path in cuda_paths:
        if os.path.exists(path):
            print(f"  ✓ {path}")
            lib_path = os.path.join(path, "lib64")
            if os.path.exists(lib_path):
                print(f"    Library path: {lib_path}")
        else:
            print(f"  ✗ {path} (not found)")


def check_environment_variables():
    """检查环境变量"""
    print("\n" + "=" * 60)
    print("Environment Variables")
    print("=" * 60)
    
    env_vars = ['CUDA_HOME', 'LD_LIBRARY_PATH', 'PATH']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        if value != 'Not set' and len(value) > 100:
            value = value[:100] + "..."
        print(f"{var}: {value}")


def main():
    print("\n" + "=" * 60)
    print("CUDA Support Diagnostic Tool")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Architecture: {os.uname().machine}")
    
    cuda_libs_ok = check_cuda_libraries()
    pytorch_cuda_ok = check_pytorch_cuda()
    check_system_cuda()
    check_environment_variables()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"CUDA libraries: {'✓ Available' if cuda_libs_ok else '✗ Not found'}")
    print(f"PyTorch CUDA: {'✓ Available' if pytorch_cuda_ok else '✗ Not available'}")
    
    if cuda_libs_ok and not pytorch_cuda_ok:
        print("\n⚠️  CUDA libraries are available, but PyTorch was not compiled with CUDA support.")
        print("   For ARM architecture, you may need to:")
        print("   1. Wait for official PyTorch ARM CUDA builds")
        print("   2. Compile PyTorch from source with CUDA support")
        print("   3. Use NVIDIA-provided builds (if available)")
    
    if not cuda_libs_ok:
        print("\n⚠️  CUDA libraries not found. Please install CUDA toolkit.")


if __name__ == "__main__":
    main()

