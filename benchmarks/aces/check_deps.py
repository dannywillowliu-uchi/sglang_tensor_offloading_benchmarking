#!/usr/bin/env python3
"""Check all SGLang dependencies."""
import sys

print("=== COMPREHENSIVE DEPENDENCY CHECK ===\n")

missing = []

# Test the full import chain for sglang serve
print("Testing SGLang multimodal serve import chain...")
try:
    from sglang.multimodal_gen.runtime.entrypoints.cli.serve import serve
    print("[OK] sglang.multimodal_gen.runtime.entrypoints.cli.serve")
except ImportError as e:
    print(f"[FAIL] {e}")
    err = str(e)
    if "No module named" in err:
        mod = err.replace("No module named ", "").strip("\"'")
        missing.append(mod.split(".")[0])

# Direct dependencies
print("\nChecking direct dependencies...")
deps = {
    "torch": "torch",
    "diffusers": "diffusers",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "safetensors": "safetensors",
    "imageio": "imageio",
    "remote_pdb": "remote_pdb",
    "requests": "requests",
    "numpy": "numpy",
    "PIL": "pillow",
    "sentencepiece": "sentencepiece",
    "einops": "einops",
    "scipy": "scipy",
    "cv2": "opencv-python",
    "tqdm": "tqdm",
    "pydantic": "pydantic",
    "uvicorn": "uvicorn",
    "fastapi": "fastapi",
}

for mod, pkg in deps.items():
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "OK")
        print(f"[OK] {pkg}: {ver}")
    except ImportError:
        print(f"[MISSING] {pkg}")
        missing.append(pkg)

print()
if missing:
    print(f"MISSING PACKAGES: {' '.join(missing)}")
    print(f"\nRun: pip install {' '.join(missing)}")
    sys.exit(1)
else:
    print("ALL DEPENDENCIES INSTALLED!")
    sys.exit(0)
