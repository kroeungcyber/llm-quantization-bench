# llm-quantization-bench — Design Spec

Date: 2026-08-12
Status: Approved by user (design review in chat)

## One-liner

Throughput/latency/quality trade-offs of quantized open-source LLMs on commodity hardware.

## Goal

A standalone portfolio repo (`~/llm-quantization-bench`, GitHub: `kroeungcyber/llm-quantization-bench`) that measures the throughput / latency / RAM / quality of Qwen 2.5 3B Instruct at four quantization levels (F16, Q8_0, Q5_K_M, Q4_K_M) on commodity hardware (Apple M4 / 16GB / Metal), and publishes the results as a clean table + matplotlib chart with a production recommendation — directly demonstrating the "Performance Optimization" duty from the job ad.

## Non-Goals

- Training/fine-tuning; GPU training; multi-model comparison beyond one model.
- Serving (this is a measurement repo, not a deployment).
- LLM-as-judge quality scoring (objective auto-scoring chosen).

## Architecture

New repo `~/llm-quantization-bench`. Structure:

```
llm-quantization-bench/
├── models/                    # gitignored — 4 downloaded GGUF files (~14GB)
├── scripts/download_models.sh # pinned HF URLs, verifies each file (hash/size)
├── eval/
│   ├── prompts.json           # 20 fixed prompts (arithmetic / exact-fact / code / instruction)
│   ├── answers.json           # reference answers for auto-scoring
│   └── run_quality.py         # per quant: runs all 20 via llama-cli, times + RAM, scores
├── bench/run_bench.py         # llama-bench wrapper → tokens/sec + TTFT per quant
├── results/results.json       # the measured data (committed)
├── plot.py                    # matplotlib chart: quality vs throughput
├── requirements.txt           # matplotlib
├── README.md                  # table + chart + conclusion
└── .gitignore                 # models/, *.gguf, __pycache__, .venv
```

### Model + quantization

- Model: **Qwen 2.5 3B Instruct** (a strong small open model, per the spec).
- GGUF files downloaded from a trusted HF repo (e.g. `bartowski/Qwen2.5-3B-Instruct-GGUF` or `unsloth`), one per quant:
  - `qwen2.5-3b-instruct-f16.gguf` (F16, ~6GB)
  - `qwen2.5-3b-instruct-q8_0.gguf` (~3.3GB)
  - `qwen2.5-3b-instruct-q5_k_m.gguf` (~2.6GB)
  - `qwen2.5-3b-instruct-q4_k_m.gguf` (~2.1GB)
- `download_models.sh` pins the URLs + checks file size/hash; re-runnable and idempotent.

### Tools

- **llama.cpp via Homebrew**, version pinned (e.g. `llama.cpp v-bXXXX`) and recorded in the README. `llama-bench` measures prompt-eval tok/s and generation tok/s; `llama-cli` runs the quality-eval prompts.
- `/usr/bin/time -l` (macOS) measures peak RSS for the quality harness.

### Measurements (per quant, same prompts / params / seed — fair)

| Metric | Source |
|---|---|
| Prompt-eval tokens/sec | `llama-bench` |
| Generation tokens/sec | `llama-bench` |
| Time-to-first-token (TTFT) | derived from `llama-bench` prompt-eval latency (first-token latency for the eval prompts) |
| Peak RAM (RSS) | `/usr/bin/time -l` around the quality harness |
| Quality % | 20-prompt auto-score (see below) |

### Quality eval (20 fixed prompts, objective auto-scoring)

Four categories × 5 prompts each:
- **Arithmetic** (5): exact numeric answers (e.g. `47 × 23`, speed/ratio word problems).
- **Exact-fact** (5): known answers (e.g. capitals, dates, constants).
- **Code** (5): write a small function → the answer is *run* and checked for correct output on test inputs (e.g. Fibonacci, is_palindrome, fizzbuzz, sum of a list, dict inversion).
- **Instruction-following** (5): format-compliance checks (e.g. "output only JSON", "list exactly three", "reply in ≤ 10 words").

Scoring: per prompt correct/incorrect (deterministic rules in `eval/score.py`), score = `% correct` per quant. Genuinely quant-sensitive — a 3B model shows measurable arithmetic/code degradation at lower quants.

`run_quality.py`: for each quant, runs all 20 prompts via `llama-cli` (fixed seed/params), records per-prompt output + pass/fail + latency + peak RAM, writes `results/quality_<quant>.json`.

### `bench/run_bench.py`

Wraps `llama-bench` per quant with fixed params (`-p 512 -n 128` style fixed prompt/generation lengths), parses the JSON output, and merges into `results/results.json` with the RAM + quality measurements.

### Results + README

- `results/results.json` (committed): all measurements — the single source of truth.
- `plot.py`: matplotlib chart — X = quality %, Y = generation tok/s (or TTFT), one point per quant, annotated. Saves `results/quant_vs_quality.png`.
- `README.md`:
  1. One-liner + what/why.
  2. Hardware & versions (M4 / 16GB / Metal, llama.cpp pinned, model files + hashes).
  3. **Results table** (from `results.json`): quant | file size | prompt tok/s | gen tok/s | TTFT | peak RAM | quality %.
  4. **matplotlib chart** embedded.
  5. **Conclusion**: which quant for a high-throughput, low-latency production deployment and why — framed as the "Performance Optimization" duty (e.g. "Q4_K_M for maximal throughput at ~X quality; Q5_K_M if you need the extra points; F16 only when accuracy is non-negotiable and RAM is abundant").
  6. Reproducibility: commands to re-run, pinned versions.
  7. Trade-offs/limitations: single model, single hardware, synthetic prompts vs real traffic, llama-bench vs end-to-end.

## Data flow

1. `download_models.sh` → 4 GGUFs in `models/`.
2. `bench/run_bench.py` → llama-bench per quant → throughput/TTFT.
3. `eval/run_quality.py` → llama-cli per quant → 20 prompts → score + RAM.
4. Merge → `results/results.json` → `plot.py` → chart → README table + conclusion.

## Error handling

- Download: hash/size verification; clear failure if a URL 404s (fall back to a documented alternate HF repo).
- Benchmark: fixed params so runs are comparable; report stderr from llama.cpp if a quant fails to load.
- Scoring: prompts/answers are fixed and deterministic; code prompts executed in a temp dir with a timeout.

## Testing

- Unit-ish: `eval/score.py` scoring rules tested against known outputs (correct/incorrect/edge).
- `bench/run_bench.py` + `eval/run_quality.py` run a smoke pass (1 prompt, small `-n`) to prove the harness works before the full run.
- The real benchmark is the definition of done — actual numbers in `results.json`.

## Verification (definition of done)

1. Full benchmark runs on this M4: all 4 quants → `results/results.json` populated with real tok/s, TTFT, RAM, quality.
2. `plot.py` produces the chart.
3. README has the table (accurate to results.json), the chart, and the production conclusion.
4. Repo pushed, Pages optional.

## Docker compatibility

Not applicable — this is a measurement repo, not a service. (llama.cpp runs natively on the host.)
