# evo2Mac

A macOS / Apple Silicon (MPS) port of [Evo 2](https://github.com/arcinstitute/evo2)
— Arc Institute's DNA language model — for local inference on Mac.

> This is a fork of [arcinstitute/evo2](https://github.com/arcinstitute/evo2)
> with edits to the device handling, FP8 fallback, and config defaults so the
> 1B and 7B Evo 2 checkpoints can run on Apple Silicon via MPS.
>
> Upstream documentation is preserved in [`README.upstream.md`](README.upstream.md).

## Why a port?

Upstream Evo 2 depends on `flash-attn` and NVIDIA Transformer Engine, both of
which are CUDA-only. This fork:

1. Disables `use_flash_attn` and `use_fp8_input_projections` in the YAML
   configs (PyTorch SDPA + bf16 work on MPS).
2. Adds MPS-aware device detection in `evo2/models.py` and `evo2/scoring.py`.
3. Extends the bf16 fallback (when Transformer Engine is missing) to also
   cover the 1B model — upstream only falls back for 7B.
4. Provides a runtime patcher (`patches/patch_vortex.py`) that fixes three
   CUDA-isms in the installed `vortex` (`vtx` on PyPI) package:
   - `torch.autocast("cuda")` → device-aware autocast
   - `torch.fft.fft(...).repeat(...)` → `.unsqueeze().expand()`
     (MPS doesn't support `.repeat` on complex tensors in PT 2.x)
   - `torch.cuda.empty_cache()` / `torch.cuda.memory_allocated()` → device-aware

The patcher writes `.bak` files and is idempotent — re-running is safe, and
`python patches/patch_vortex.py --restore` puts the originals back.

## Models

| Checkpoint            | Size (bf16) | Runs on Mac?              |
|-----------------------|-------------|---------------------------|
| `evo2_1b_base`        | ~4 GB       | ✓ 16 GB unified mem OK    |
| `evo2_7b`             | ~14 GB      | ✓ needs 32 GB+ Mac        |
| `evo2_7b_base`        | ~14 GB      | ✓ needs 32 GB+ Mac        |
| `evo2_7b_262k`        | ~14 GB      | ✓ needs 32 GB+ Mac        |
| `evo2_7b_microviridae`| ~14 GB      | ✓ needs 32 GB+ Mac        |
| `evo2_20b`            | ~40 GB      | ✗ requires FP8 + Hopper   |
| `evo2_40b`            | ~80 GB      | ✗ requires FP8 + Hopper   |
| `evo2_40b_base`       | ~80 GB      | ✗ requires FP8 + Hopper   |

The 20B/40B exclusion is a *runtime* constraint (Transformer Engine + Hopper
GPUs), not just a memory one. They will not run on Apple Silicon even if it
fit. Long-context 7B (`evo2_7b`, 1M context) is technically loadable but the
prefill cost on MPS will be painful — start with `evo2_7b_262k` or
`evo2_7b_base` (8K context).

## Quick start

Prerequisites: Apple Silicon Mac, macOS 14+, [Homebrew](https://brew.sh).

```bash
git clone https://github.com/wabi-media/evo2Mac.git
cd evo2Mac
./scripts/setup.sh
conda activate evo2Mac

# 1. Quick smoke test (one forward pass)
python scripts/smoke_test.py --model evo2_1b_base

# 2. Full DNA pipeline: tokenize -> forward -> embed -> score -> generate
python scripts/test_dna.py --model evo2_1b_base

# 3. Compare numerical results to upstream's CUDA reference values
python scripts/compare_to_upstream.py --model evo2_1b_base
```

`setup.sh` will:
1. Install miniforge via Homebrew (skip if present).
2. Create a Python 3.11 conda env named `evo2Mac`.
3. Install PyTorch with MPS support.
4. Install `vtx` (the StripedHyena 2 runtime; imported as `vortex`).
5. Install this package in editable mode (`pip install -e . --no-deps`).
6. Apply the runtime patches to the installed `vortex` package.

On first model load, the checkpoint is downloaded into your HuggingFace cache
(`~/.cache/huggingface/`). Change with `HF_HOME=/path/to/cache`.

## Verifying correctness vs upstream

`scripts/compare_to_upstream.py` runs upstream's own bundled `prompts.csv`
through the model and compares the mean cross-entropy and next-token
accuracy against the reference numbers baked into upstream's
`evo2/test/test_evo2.py`. Those reference values were measured on
H100 + FP8 + flash-attn. Our port runs in bf16 + SDPA on MPS, so a small
drift is expected:

| Tolerance | Loss (cross-entropy) | Accuracy (pp) |
|-----------|----------------------|---------------|
| OK        | drift ≤ 0.05         | drift ≤ 1.5   |
| WARN      | 0.05 < drift ≤ 0.15  | 1.5 < drift ≤ 5 |
| FAIL      | drift > 0.15         | drift > 5     |

A failure here means the port is producing meaningfully different outputs
and something is wrong — it's the canary that should run on every fresh
install.

## Usage

```python
import torch
from evo2 import Evo2

m = Evo2("evo2_1b_base")          # auto-detects MPS / CUDA / CPU
print("device:", m.device)

ids = torch.tensor(m.tokenizer.tokenize("ACGTACGT"), dtype=torch.int).unsqueeze(0)
logits, _ = m(ids)
print(logits.shape)               # (1, 8, 512)

# Scoring
scores = m.score_sequences(["ACGTACGT", "GATTACA"])

# Generation (cached sampling works on MPS)
out = m.generate(prompt_seqs=["ACGT"], n_tokens=64, temperature=1.0, top_k=4)
print(out.sequences[0])
```

## Keeping in sync with upstream

```bash
git remote -v
# origin    https://github.com/wabi-media/evo2Mac.git    (your fork)
# upstream  https://github.com/arcinstitute/evo2.git     (Arc Institute)

git fetch upstream
git merge upstream/main         # or rebase, your call
```

When upstream lands changes to `evo2/models.py`, `evo2/scoring.py`, or the
configs, you may have to redo the Mac edits — they're small and well-marked
with `# evo2Mac:` comments.

## What this port does *not* do

- It does **not** redistribute model weights — those come from HuggingFace on
  first use.
- It does **not** train / fine-tune. Inference only.
- It does **not** make 20B/40B run on Mac. Those need Hopper GPUs.

## Credits

- Upstream model + reference code: [arcinstitute/evo2](https://github.com/arcinstitute/evo2)
  (Arc Institute, Michael Poli, Stanford University). Apache 2.0.
- The Mac compatibility notes that informed this fork's patches: the
  [hakyimlab/evo2-mac](https://github.com/hakyimlab/evo2-mac) effort by the
  Im Lab at UChicago.
- StripedHyena 2 / Vortex runtime: Together. See [`NOTICE.upstream`](NOTICE.upstream).

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). Modifications and attribution
in [`NOTICE`](NOTICE).
