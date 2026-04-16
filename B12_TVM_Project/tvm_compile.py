from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

try:
    import onnx
except ImportError as exc:
    print("Missing dependency: onnx")
    print("Install it with: pip install onnx")
    raise

try:
    import tvm # type: ignore[import]
    from tvm import relay # type: ignore[import]
except ImportError as exc:
    print("Apache TVM is not available in this Python environment.")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print("")
    print("If you are on Windows, run this script inside your Multipass Ubuntu VM")
    print("or inside WSL/Linux where TVM is built and installed.")
    print("")
    print("Example:")
    print("  multipass shell tvm-ubuntu")
    print("  source /home/ubuntu/hd-project/.venv-linux-tvm/bin/activate")
    print("  python tvm_compile.py --onnx models/resnet18.onnx --batch 1 --out_dir artifacts/resnet18/batch_1")
    raise SystemExit(1)