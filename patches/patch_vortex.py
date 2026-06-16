#!/usr/bin/env python3
"""
Patch the installed `vortex` package (PyPI distribution: `vtx`) for macOS/MPS.

These edits fix three CUDA-isms in the StripedHyena 2 reference implementation
that crash on Apple Silicon. They are device-aware: when CUDA is available the
patched code falls back to the original CUDA path.

The patcher is idempotent — re-running it on already-patched files is a no-op.
It writes a `.bak` next to each file the first time it edits that file.

Run after `pip install vtx`:

    python patches/patch_vortex.py

To undo:

    python patches/patch_vortex.py --restore
"""

from __future__ import annotations

import argparse
import importlib
import re
import shutil
import sys
from pathlib import Path


PATCH_MARK = "# evo2Mac patch"


def vortex_root() -> Path:
    """Resolve the installed `vortex` package directory."""
    try:
        spec = importlib.util.find_spec("vortex")
    except Exception as e:
        sys.exit(f"Could not import vortex: {e}\nDid you `pip install vtx` first?")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("`vortex` package not found. Run `pip install vtx` first.")
    return Path(next(iter(spec.submodule_search_locations)))


def back_up(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)


def restore(root: Path) -> None:
    n = 0
    for bak in root.rglob("*.bak"):
        target = bak.with_suffix("")
        shutil.copy2(bak, target)
        bak.unlink()
        print(f"  restored {target.relative_to(root)}")
        n += 1
    print(f"restored {n} file(s)")


def edit(path: Path, edits: list[tuple[str, str, str]]) -> int:
    """
    Apply a list of (pattern, replacement, label) edits to `path`.

    Each edit is skipped if the file already contains the patch mark for that
    label, so re-running is safe. Returns the number of edits actually applied.
    """
    if not path.exists():
        print(f"  ! missing: {path}")
        return 0
    text = path.read_text()
    applied = 0
    for pattern, replacement, label in edits:
        mark = f"{PATCH_MARK}: {label}"
        if mark in text:
            print(f"  - {path.name}: '{label}' already patched")
            continue
        new_text, n = re.subn(pattern, replacement, text, count=1)
        if n == 0:
            print(f"  ! {path.name}: pattern for '{label}' not found")
            continue
        text = new_text
        applied += 1
        print(f"  + {path.name}: applied '{label}'")
    if applied:
        back_up(path)
        path.write_text(text)
    return applied


def patch_engine(root: Path) -> int:
    """
    vortex/model/engine.py
      1. torch.autocast("cuda")     -> device-aware autocast
      2. torch.fft.fft().repeat()   -> .unsqueeze().expand()
         (MPS doesn't support .repeat on complex tensors in PT 2.x)
    """
    p = root / "model" / "engine.py"
    edits = [
        (
            r'with torch\.autocast\(\s*"cuda"\s*\):',
            (
                'with torch.autocast(  # evo2Mac patch: autocast\n'
                '            "cuda" if torch.cuda.is_available()\n'
                '            else ("mps" if torch.backends.mps.is_available() else "cpu")\n'
                '        ):'
            ),
            "autocast",
        ),
        (
            r'state_S\s*=\s*torch\.fft\.fft\(\s*state_s\s*,\s*n=fft_size\s*\)\.repeat\(\s*bs\s*,\s*1\s*,\s*1\s*,\s*1\s*\)',
            (
                "state_S = torch.fft.fft(state_s, n=fft_size)  # evo2Mac patch: fft_repeat\n"
                "        state_S = state_S.unsqueeze(0).expand(bs, -1, -1, -1)"
            ),
            "fft_repeat",
        ),
    ]
    return edit(p, edits)


def patch_generation(root: Path) -> int:
    """
    vortex/model/generation.py
      torch.cuda.memory_allocated(device=x.device)  ->  torch.memory_allocated(x.device)
    """
    p = root / "model" / "generation.py"
    edits = [
        (
            r"torch\.cuda\.memory_allocated\(\s*device\s*=\s*x\.device\s*\)",
            "torch.memory_allocated(x.device)  # evo2Mac patch: mem_allocated",
            "mem_allocated",
        ),
    ]
    return edit(p, edits)


def patch_model(root: Path) -> int:
    """
    vortex/model/model.py
      torch.cuda.empty_cache()  ->  device-aware empty_cache
    """
    p = root / "model" / "model.py"
    edits = [
        (
            r"torch\.cuda\.empty_cache\(\)",
            (
                "(torch.cuda.empty_cache() if torch.cuda.is_available()  # evo2Mac patch: empty_cache\n"
                "         else (torch.mps.empty_cache() if torch.backends.mps.is_available() else None))"
            ),
            "empty_cache",
        ),
    ]
    return edit(p, edits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore", action="store_true", help="restore .bak files")
    args = ap.parse_args()

    root = vortex_root()
    print(f"vortex root: {root}")

    if args.restore:
        restore(root)
        return 0

    total = 0
    total += patch_engine(root)
    total += patch_generation(root)
    total += patch_model(root)
    print(f"\napplied {total} edit(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
