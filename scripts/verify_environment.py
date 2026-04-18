#!/usr/bin/env python3
"""
Environment verification script for RoboRenForce

Checks if all required dependencies are installed and working.
"""

import sys
from pathlib import Path

def check_import(module_name, display_name=None):
    """Check if a module can be imported."""
    display_name = display_name or module_name
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {display_name:20s} {version}")
        return True
    except ImportError as e:
        print(f"✗ {display_name:20s} NOT FOUND")
        print(f"  Error: {e}")
        return False

def main():
    print("=" * 60)
    print("RoboRenForce Environment Verification")
    print("=" * 60)
    print()

    # Check Python version
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.executable}")
    print()

    # Core dependencies
    print("Core Dependencies:")
    print("-" * 60)
    checks = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("transformers", "Transformers"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("pyarrow", "PyArrow"),
        ("safetensors", "SafeTensors"),
        ("einops", "Einops"),
    ]

    all_passed = True
    for module, name in checks:
        if not check_import(module, name):
            all_passed = False

    print()

    # Optional dependencies
    print("Optional Dependencies:")
    print("-" * 60)
    optional_checks = [
        ("peft", "PEFT (LoRA)"),
        ("accelerate", "Accelerate"),
        ("av", "PyAV (video)"),
    ]

    for module, name in optional_checks:
        check_import(module, name)

    print()

    # Project imports
    print("Project Imports:")
    print("-" * 60)

    # Add source to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root / "source" / "RoboRenForce"))

    project_checks = [
        ("RoboRenForce.dataset.lerobot", "LeRobot Dataset"),
        ("RoboRenForce.networks.vlm", "VLM Networks"),
    ]

    for module, name in project_checks:
        if not check_import(module, name):
            all_passed = False

    print()
    print("=" * 60)

    if all_passed:
        print("✓ All required dependencies are installed!")
        print()
        print("Next steps:")
        print("  1. Generate test data: python scripts/data/generate_dummy_dataset.py")
        print("  2. Test dataset: python -c 'from RoboRenForce.dataset.lerobot import LeRobotDatasetCfg; ...'")
        print("  3. See TODO-NEXT.md for implementation tasks")
        return 0
    else:
        print("✗ Some dependencies are missing!")
        print()
        print("To install missing dependencies:")
        print("  uv sync --extra all")
        return 1

if __name__ == "__main__":
    sys.exit(main())
