"""
install.py — runs once when the extension is first installed.
Checks for sd-dynamic-prompts and warns if missing.
"""
import importlib.util
import sys


def check_dep(package, pip_name=None):
    if importlib.util.find_spec(package) is None:
        pip_name = pip_name or package
        print(
            f"[SmartLoRACache] Optional dependency '{pip_name}' not found. "
            f"Install it for wildcard pre-resolution support:\n"
            f"  pip install {pip_name}"
        )
    else:
        print(f"[SmartLoRACache] Found: {package} ✓")


check_dep("dynamicprompts", "sd-dynamic-prompts")
