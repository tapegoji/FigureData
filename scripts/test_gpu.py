#!/usr/bin/env python3
"""
GPU Test Script

This script tests if CUDA and GPU are properly configured for PyTorch.
"""

import torch
import sys

def test_gpu():
    """Test GPU availability and functionality."""
    print("=" * 60)
    print("GPU AVAILABILITY TEST")
    print("=" * 60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        
        # Test basic GPU operations
        try:
            print("\nTesting GPU operations...")
            device = torch.device('cuda:0')
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            z = torch.mm(x, y)
            print("✅ GPU operations successful!")
            print(f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        except Exception as e:
            print(f"❌ GPU operations failed: {e}")
    else:
        print("❌ CUDA not available")
        print("Note: You may need to reboot after installing NVIDIA drivers")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_gpu() 