# Running Qwen3-30B-A3B on the 3090

> ⚠️ **vLLM is currently BROKEN on WSL2** (2026-05-31). Latest vLLM's V2 model
> runner requires UVA (unified virtual addressing / zero-copy host memory),
> which WSL2's GPU paravirtualization doesn't support → `RuntimeError: UVA is
> not available` at engine-core init. Known open bug
> (github.com/vllm-project/vllm/issues/43381); fix PR #43348 not yet merged.
>
> **Until that merges, run Qwen3-30B on Ollama instead** (it serves the same
> model). vLLM's edge is batched-concurrency throughput, which our evaluator
> doesn't use yet (serial loop) - so we lose nothing meaningful by staying on
> Ollama for now. Revisit vLLM when (a) the PR merges AND (b) we make the
> evaluator concurrent.
>
> ### Qwen3-30B on Ollama (the working path today)
> ```bash
> ollama pull qwen3:30b-a3b-thinking-2507-q4_K_M   # ~18-19GB Q4, fits 24GB
> ```
> Then in .env:
> ```
> DEFAULT_MODEL=ollama/qwen3:30b-a3b-thinking-2507-q4_K_M
> OLLAMA_API_BASE=http://localhost:11434
> ```
> The rest of this doc (vLLM) is kept for when the WSL bug is fixed.

---

# Running Qwen3-30B-A3B on the 3090 via vLLM (blocked on WSL - see warning above)

Upgrade path from Ollama/qwen2.5-14B to a stronger reasoning model. vLLM serves
an OpenAI-compatible endpoint; polyevolve's LiteLLM path routes to it via the
`hosted_vllm/` provider - no agent code changes, just a model id + env var.

## Why this model

`Qwen3-30B-A3B-Instruct-2507` (AWQ 4-bit): 30B total / **3.3B active** (MoE), so
it has ~30B knowledge at ~3B-active speed, has a native thinking mode, and fits
24GB (~17GB weights + room for ~24-32K context KV cache). The newer Qwen3.6
27-35B variants went multimodal and their 4-bit weights are 21-24GB - they do
NOT fit a single 3090. This is the single-card sweet spot.

It will be slower per call than qwen2.5-14B (more weights to read) but should be
materially better at calibrated reasoning. We measure that directly - see step 5.

## ⚠️ vLLM vs Ollama coexistence

vLLM grabs most of the 24GB at startup. **You cannot run vLLM and an Ollama
model simultaneously.** Stop Ollama models first:

```bash
ollama stop qwen2.5:14b 2>/dev/null || true   # frees VRAM
```

(Ollama the service can stay running; just don't have it holding a model.)

---

## One-time setup (YOUR machine - large download + GPU install)

### 1. Install vLLM in its own venv

vLLM has heavy, CUDA-matched deps - keep it isolated from polyevolve's venv.

```bash
# from anywhere
~/.local/bin/uv venv ~/.venvs/vllm --python 3.12
~/.local/bin/uv pip install --python ~/.venvs/vllm/bin/python vllm
```

(First install pulls ~several GB of CUDA wheels. If it errors on CUDA version,
see vLLM's install docs for the matching `--torch-backend`/index.)

**WSL / "Could not find nvcc" fix:** vLLM JIT-compiles a kernel at load and needs
`nvcc` (CUDA *compiler*), which the NVIDIA driver/runtime alone doesn't provide.
You do NOT need a system CUDA toolkit - the pip install already bundles one in
the venv. Point `CUDA_HOME` at it (path matches your torch cuda version, e.g.
`cu13`):

```bash
export CUDA_HOME=$HOME/.venvs/vllm/lib/python3.12/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
# verify: nvcc --version  -> should print release 13.x
```
Prefix the `vllm serve` command with these (or export them in the serving shell).

### 2. Serve the model

```bash
~/.venvs/vllm/bin/vllm serve stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --reasoning-parser qwen3 \
  --served-model-name qwen3-30b \
  --port 8000
```

- First run downloads the ~17GB model from HuggingFace (one time, cached in
  `~/.cache/huggingface`).
- **Do NOT pass `--quantization awq_marlin`.** Despite the "AWQ" in the repo
  name, this model is quantized with `compressed-tensors` (llm-compressor INT4).
  Passing `awq_marlin` fails with a quant-method mismatch. Omit the flag and
  vLLM auto-detects the correct method (and still uses Marlin kernels on Ampere).
- `--served-model-name qwen3-30b` gives it a clean id we reference below.
- If it OOMs: drop `--max-model-len 16384`, or `--gpu-memory-utilization 0.90`.
- Leave this running in its own terminal (or `tmux`).

### 3. Verify it's up

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

---

## Point polyevolve at vLLM (one config change)

Edit `.env`:

```bash
DEFAULT_MODEL=hosted_vllm/qwen3-30b
VLLM_API_BASE=http://localhost:8000/v1
```

The `hosted_vllm/` prefix routes through LiteLLM to the vLLM server. The model
name after the slash must match `--served-model-name`.

---

## 4. Smoke test through polyevolve

```bash
uv run python -m polyevolve.cli evaluate --set fp_smoke --model hosted_vllm/qwen3-30b
```

Should produce a prediction with cache misses (new model = new genome_hash key
is unaffected, but new model_name = fresh cache rows).

## 5. The real experiment - re-baseline on the SAME frozen set

```bash
uv run python -m polyevolve.cli evaluate --set fp_v1 --model hosted_vllm/qwen3-30b
```

Compare `edge_holdout` directly against the qwen2.5-14B baseline:

| Model | edge_holdout (fp_v1) |
|---|---|
| ollama/qwen2.5:14b | **−0.219** (loses to market) |
| hosted_vllm/qwen3-30b | ??? ← this run answers it |

This is the highest-value single data point right now: does a stronger reasoning
model alone move the edge toward zero/positive? The snapshot is identical and
frozen, so it's a clean apples-to-apples comparison.

## Then: evolution on the better model

If Qwen3-30B's baseline is promising, run the evolution loop (`polyevolve.evolve`) with
`POLYEVOLVE_MODEL=hosted_vllm/qwen3-30b` instead of the Ollama id. Everything else
is identical - the harness is model-agnostic by design.

## Notes / gotchas

- No published calibration benchmark for Qwen3 - better reasoning ≠ guaranteed
  better calibration. We measure it ourselves (that's the whole point of fp_v1).
- vLLM thinking mode emits `<think>` blocks; `--reasoning-parser qwen3` strips
  them so our tool-call parsing sees clean output. If predictions come back
  malformed, that parser flag is the first thing to check.
- WSL: vLLM uses the same CUDA passthrough as Ollama (confirmed working on this
  3090), so no extra WSL config.
