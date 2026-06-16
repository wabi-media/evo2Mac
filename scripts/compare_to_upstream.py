#!/usr/bin/env python3
"""
Compare evo2Mac (MPS) numerical output against upstream's published reference
values for arcinstitute/evo2 on CUDA.

Upstream `evo2/test/test_evo2.py` runs a forward pass over a bundled
prompts.csv (8K-context DNA prompts) and bakes in reference loss / accuracy
numbers measured on H100 / FP8 + flash-attn for each checkpoint:

    Evo 2 1B base:  Loss ~0.502, Accuracy ~79.56%
    Evo 2 7B base:  Loss ~0.352, Accuracy ~85.92%
    Evo 2 7B:       Loss ~0.348, Accuracy ~86.35%
    Evo 2 20B:      Loss ~0.217, Accuracy ~91.67%
    Evo 2 40B:      Loss ~0.216, Accuracy ~91.67%

Our Mac port runs in bf16 with SDPA (no FP8, no flash-attn) so a small
numerical drift is expected. The argmax-based accuracy should be nearly
identical; the cross-entropy loss may drift by O(1e-3) to O(1e-2) but a
correct port should NOT drift by more than that.

Usage:
    conda activate evo2Mac
    python scripts/compare_to_upstream.py --model evo2_1b_base
    python scripts/compare_to_upstream.py --model evo2_7b_base   # 32GB+ Mac

Exit code:
    0 — within tolerance (port matches upstream)
    1 — outside fail tolerance (port may be broken)
    2 — usage / load error
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from importlib import resources
from pathlib import Path

# Numbers from upstream evo2/test/test_evo2.py docstring + expected_metrics
# (commit 3a4d1d0). Measured on H100 with FP8 + flash-attn.
UPSTREAM_REFERENCE = {
    "evo2_1b_base": {"loss": 0.501953125,  "acc": 79.556},
    "evo2_7b_base": {"loss": 0.3520508,    "acc": 85.921},
    "evo2_7b":      {"loss": 0.3476563,    "acc": 86.346},
    "evo2_20b":     {"loss": 0.2166748046875, "acc": 91.666},
    "evo2_40b":     {"loss": 0.2159424,    "acc": 91.673},
    "evo2_40b_base":{"loss": 0.2149658,    "acc": 91.741},
}

# Tolerance bands. bf16+SDPA on MPS vs bf16+FP8+FlashAttn on H100 are not
# bit-equal; some drift is healthy. These thresholds are deliberately loose
# enough to absorb that but tight enough to catch a real bug.
LOSS_WARN  = 0.05    # 5e-2 absolute drift in cross-entropy nats
LOSS_FAIL  = 0.15    # 1.5e-1 — well outside numerical noise
ACC_WARN   = 1.5     # percentage points
ACC_FAIL   = 5.0     # 5 pp would mean the model is meaningfully degraded

MAC_FEASIBLE = {"evo2_1b_base", "evo2_7b", "evo2_7b_base", "evo2_7b_262k", "evo2_7b_microviridae"}


def read_prompts() -> list[str]:
    """Load upstream's bundled prompts.csv (same one its own test uses)."""
    with resources.path("evo2.test.data", "prompts.csv") as p:
        path = Path(p)
    seqs: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if row and row[0].strip():
                seqs.append(row[0].strip())
    return seqs


def forward_pass(model, sequences, max_seqs: int | None) -> tuple[list[float], list[float]]:
    """Replicates upstream test_forward_pass but device-aware."""
    import torch
    import torch.nn.functional as F

    if max_seqs is not None:
        sequences = sequences[:max_seqs]

    losses: list[float] = []
    accuracies: list[float] = []

    for i, seq in enumerate(sequences, 1):
        ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.int).to(model.device)
        with torch.inference_mode():
            out = model.model.forward(ids.unsqueeze(0))
        logits = out[0] if isinstance(out, tuple) else out

        target_ids = ids[1:].long()
        pred_logits = logits[0, :-1, :]

        # bf16 -> fp32 for loss to avoid underflow in cross-entropy.
        loss = F.cross_entropy(pred_logits.float(), target_ids)
        pred_tokens = torch.argmax(pred_logits, dim=-1)
        acc = (target_ids == pred_tokens).float().mean().item()

        losses.append(loss.item())
        accuracies.append(acc)
        print(f"  seq {i:>2}/{len(sequences)}: loss={loss.item():.4f}  acc={acc*100:.2f}%")

    return accuracies, losses


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="evo2_1b_base",
                    choices=sorted(UPSTREAM_REFERENCE))
    ap.add_argument("--max-seqs", type=int, default=None,
                    help="cap number of prompts (default: all)")
    ap.add_argument("--device", default=None,
                    help="override device (mps/cpu/cuda:0)")
    args = ap.parse_args()

    if args.model not in MAC_FEASIBLE:
        print(f"NOTE: {args.model} is not expected to run on Mac (FP8/Hopper required).")
        print(f"      Run anyway for diagnostics, but expect a load error.")

    import numpy as np
    import torch

    print(f"torch:         {torch.__version__}")
    print(f"cuda avail:    {torch.cuda.is_available()}")
    print(f"mps avail:     {torch.backends.mps.is_available()}")

    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(1)

    print(f"\nloading {args.model} ...")
    t0 = time.time()
    try:
        from evo2 import Evo2
        model = Evo2(args.model)
        if args.device is not None:
            model.device = args.device
            model.model = model.model.to(args.device)
    except Exception as e:
        print(f"FAIL: could not load model: {e}")
        traceback.print_exc()
        return 2
    print(f"  device:    {model.device}")
    print(f"  loaded in: {time.time() - t0:.1f}s\n")

    print("loading upstream prompts.csv ...")
    seqs = read_prompts()
    if args.max_seqs:
        seqs = seqs[: args.max_seqs]
    print(f"  {len(seqs)} prompts\n")

    print("running forward pass on each prompt ...")
    t1 = time.time()
    accs, losses = forward_pass(model, seqs, args.max_seqs)
    print(f"\ntotal forward-pass time: {time.time() - t1:.1f}s")

    mean_loss = float(np.mean(losses))
    mean_acc_pct = float(np.mean(accs) * 100)

    ref = UPSTREAM_REFERENCE[args.model]
    loss_delta = abs(mean_loss - ref["loss"])
    acc_delta = abs(mean_acc_pct - ref["acc"])

    print("\n" + "=" * 62)
    print(f"  results for {args.model} on {model.device}")
    print("=" * 62)
    print(f"  upstream (H100, FP8, flash-attn):")
    print(f"    loss  = {ref['loss']:.4f}")
    print(f"    acc   = {ref['acc']:.3f}%")
    print(f"  evo2Mac (this run):")
    print(f"    loss  = {mean_loss:.4f}    Δ = {loss_delta:.4f}")
    print(f"    acc   = {mean_acc_pct:.3f}%  Δ = {acc_delta:.3f} pp")
    print("=" * 62)

    failed = False
    if loss_delta > LOSS_FAIL:
        print(f"  FAIL: loss drift {loss_delta:.4f} exceeds {LOSS_FAIL} fail threshold")
        failed = True
    elif loss_delta > LOSS_WARN:
        print(f"  WARN: loss drift {loss_delta:.4f} exceeds {LOSS_WARN} warn threshold "
              "(bf16/SDPA can drift this much vs FP8+flash-attn — review)")
    else:
        print(f"  loss within ±{LOSS_WARN}: OK")

    if acc_delta > ACC_FAIL:
        print(f"  FAIL: accuracy drift {acc_delta:.3f}pp exceeds {ACC_FAIL}pp fail threshold")
        failed = True
    elif acc_delta > ACC_WARN:
        print(f"  WARN: accuracy drift {acc_delta:.3f}pp exceeds {ACC_WARN}pp warn threshold")
    else:
        print(f"  accuracy within ±{ACC_WARN}pp: OK")

    if failed:
        print("\nport appears to be producing wrong outputs.")
        return 1
    print("\nport matches upstream within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
