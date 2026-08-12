# llm-quantization-bench — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure throughput/latency/RAM/quality of Qwen 2.5 3B Instruct at F16/Q8_0/Q5_K_M/Q4_K_M on this M4, publish the results as a table + matplotlib chart + a production recommendation.

**Architecture:** New repo `~/llm-quantization-bench`. llama.cpp (Homebrew, pinned) with `llama-bench` for tok/s + TTFT; a 20-prompt objective auto-scored quality harness (arithmetic / exact-fact / code / instruction) with peak-RAM via `/usr/bin/time -l`; `results/results.json` as the single source of truth; `plot.py` (matplotlib) and a README with the table + chart + conclusion.

**Tech Stack:** llama.cpp (llama-bench/llama-cli), Python 3.11 venv (matplotlib), Qwen 2.5 3B Instruct GGUF (4 quants), Apple M4 / 16GB / Metal.

**Reference spec:** `docs/superpowers/specs/2026-08-12-llm-quantization-bench-design.md`

## Global Constraints

- One model (Qwen 2.5 3B Instruct), four quants: F16, Q8_0, Q5_K_M, Q4_K_M.
- Fixed eval params across quants: `--temp 0 --seed 42`, same 20 prompts, same max-token budget — the ONLY variable is the quant (fair comparison).
- All results go into `results/results.json` (committed) — the README table/chart derive from it, never hand-typed numbers.
- Reproducibility: pin the llama.cpp version + record hardware/model hashes in the README.
- `models/` is gitignored (14GB); `results/` is committed.
- Use a python3.11 venv at `/Users/bila/llm-quantization-bench/.venv` for the harness/plot scripts (matplotlib).

---

### Task 1: Scaffold + llama.cpp + model download

**Files:** `requirements.txt`, `pyproject.toml`, `.gitignore`, `.flake8`, `scripts/download_models.sh`, venv.

- [ ] **Step 1: Scaffold files**

`requirements.txt`:
```
matplotlib>=3.8
httpx>=0.27
```

`pyproject.toml` (name `llm-quantization-bench`, requires-python >=3.9, dev extras pytest/mypy/flake8/httpx, `[tool.pytest.ini_options] testpaths=["tests"]`, mypy override for `matplotlib.*`).

`.flake8` (max-line-length 120, E302/E303/E203/E127).

`.gitignore`:
```
models/
*.gguf
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.venv/
dist/
```

Create the venv: `/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv /Users/bila/llm-quantization-bench/.venv` then `.venv/bin/pip install -r requirements.txt pytest pytest-mock mypy flake8`.

- [ ] **Step 2: Verify llama.cpp installed (it may already be installing in the background)**

```bash
command -v llama-bench llama-cli
llama-bench --version 2>&1 | head -1 || llama-cli --version
```
If not installed: `brew install llama.cpp`. Record the version (e.g. from `llama-bench --version` or `brew list --versions llama.cpp`).

- [ ] **Step 3: Create `scripts/download_models.sh`**

Pinned HF URLs for the four GGUFs (from a trusted repo, e.g. `bartowski/Qwen2.5-3B-Instruct-GGUF` — verify the exact filenames during the task; common names):

```bash
#!/usr/bin/env bash
set -euo pipefail
# Qwen 2.5 3B Instruct GGUF files, pinned URLs. Verify each is a real file (~size).
BASE="https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main"
DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
mkdir -p "$DIR"
declare -A FILES=(
  ["Qwen2.5-3B-Instruct-F16.gguf"]="~6GB"
  ["Qwen2.5-3B-Instruct-Q8_0.gguf"]="~3.3GB"
  ["Qwen2.5-3B-Instruct-Q5_K_M.gguf"]="~2.6GB"
  ["Qwen2.5-3B-Instruct-Q4_K_M.gguf"]="~2.1GB"
)
for f in "${!FILES[@]}"; do
  if [ ! -f "$DIR/$f" ] || [ ! -s "$DIR/$f" ]; then
    echo "downloading $f (${FILES[$f]})"
    curl -L --fail --retry 3 -o "$DIR/$f" "$BASE/$f"
  else
    echo "exists: $f"
  fi
done
echo "done:"
ls -lh "$DIR"
```

If a URL 404s, find the correct filename (e.g. check the HF repo file list via the API: `curl -s https://huggingface.co/api/models/bartowski/Qwen2.5-3B-Instruct-GGUF` and grep the gguf names) and fix the script. All four files must download and be non-empty.

- [ ] **Step 4: Run the download (~14GB — long)**

`bash scripts/download_models.sh` — report the final sizes. Verify each with `file models/*.gguf` → "GGUF model".

- [ ] **Step 5: Smoke test llama-cli on the Q4 file**

```bash
llama-cli -m models/Qwen2.5-3B-Instruct-Q4_K_M.gguf -p "What is 2+2?" -n 8 --temp 0 --seed 42 -no-cnv 2>/dev/null | tail -3
```
Expected: an answer (4) in a few seconds. Also smoke `llama-bench -m models/Qwen2.5-3B-Instruct-Q4_K_M.gguf -p 32 -n 16 -r 1` works (small run) and prints tok/s.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: scaffold, llama.cpp, download Qwen 2.5 3B GGUFs"
```

---

### Task 2: Eval prompts + answers + scoring (`eval/`)

**Files:** `eval/prompts.json`, `eval/answers.json`, `eval/score.py`, `tests/test_score.py`.

- [ ] **Step 1: Design the 20 prompts** (`eval/prompts.json`) — 4 categories × 5, each with an `id`, `category`, `prompt`, and `max_tokens`:

```json
{
  "prompts": [
    {"id": "arith_1", "category": "arithmetic", "prompt": "What is 47 multiplied by 23? Answer with just the number.", "max_tokens": 16},
    {"id": "arith_2", "category": "arithmetic", "prompt": "A train travels 120 km in 2.5 hours. What is its average speed in km/h? Answer with just the number.", "max_tokens": 16},
    {"id": "arith_3", "category": "arithmetic", "prompt": "What is 15% of 240? Answer with just the number.", "max_tokens": 16},
    {"id": "arith_4", "category": "arithmetic", "prompt": "What is 2^10? Answer with just the number.", "max_tokens": 16},
    {"id": "arith_5", "category": "arithmetic", "prompt": "If x = 3 and y = 4, what is x^2 + y^2? Answer with just the number.", "max_tokens": 16},
    {"id": "fact_1", "category": "exact-fact", "prompt": "What is the capital of Australia?", "max_tokens": 16},
    {"id": "fact_2", "category": "exact-fact", "prompt": "In what year did World War II end?", "max_tokens": 16},
    {"id": "fact_3", "category": "exact-fact", "prompt": "What is the chemical symbol for gold?", "max_tokens": 16},
    {"id": "fact_4", "category": "exact-fact", "prompt": "Who wrote 'Romeo and Juliet'?", "max_tokens": 16},
    {"id": "fact_5", "category": "exact-fact", "prompt": "What is the largest planet in our solar system?", "max_tokens": 16},
    {"id": "code_1", "category": "code", "prompt": "Write a Python function is_palindrome(s) that returns True if the string s is a palindrome, else False. Output only the Python code, no explanation.", "max_tokens": 128},
    {"id": "code_2", "category": "code", "prompt": "Write a Python function fibonacci(n) that returns the nth Fibonacci number (fibonacci(0)=0, fibonacci(1)=1). Output only the Python code.", "max_tokens": 128},
    {"id": "code_3", "category": "code", "prompt": "Write a Python function fizzbuzz(n) that returns a list where multiples of 3 are 'Fizz', multiples of 5 are 'Buzz', both are 'FizzBuzz', else the number, for 1..n. Output only the Python code.", "max_tokens": 128},
    {"id": "code_4", "category": "code", "prompt": "Write a Python function sum_list(nums) that returns the sum of a list of numbers. Output only the Python code.", "max_tokens": 128},
    {"id": "code_5", "category": "code", "prompt": "Write a Python function invert_dict(d) that returns a dict mapping values to keys. Output only the Python code.", "max_tokens": 128},
    {"id": "inst_1", "category": "instruction", "prompt": "List exactly three reasons to back up data. Output only the three items, numbered.", "max_tokens": 48},
    {"id": "inst_2", "category": "instruction", "prompt": "Output only this JSON: {\"status\": \"ok\"}", "max_tokens": 32},
    {"id": "inst_3", "category": "instruction", "prompt": "Write the word 'banana' three times separated by spaces and nothing else.", "max_tokens": 16},
    {"id": "inst_4", "category": "instruction", "prompt": "Sort these numbers ascending and output only the sorted list: 9, 1, 5, 3, 7", "max_tokens": 32},
    {"id": "inst_5", "category": "instruction", "prompt": "Answer with a single word: is the sky blue?", "max_tokens": 8}
  ]
}
```

`eval/answers.json` holds the reference answers / scoring rules keyed by prompt id.

- [ ] **Step 2: Write `eval/score.py`** with deterministic scoring functions:
  - `score_arithmetic(output, expected) -> bool` — extract the first number in the output, compare to the expected (int/float tolerance).
  - `score_fact(output, expected) -> bool` — normalized substring match (lowercase, strip punctuation) against the expected answer.
  - `score_code(output, func_name, test_cases) -> bool` — extract the Python function from the output (strip markdown fences / prose), `exec` it in a temp namespace, run the test cases, return all pass. Timeout guard.
  - `score_instruction(output, rule) -> bool` — per-rule checks (count of items, exact JSON, word count, etc.).
  - `score_all(quant_outputs) -> dict` — returns per-prompt pass/fail + the % per category + overall %.

- [ ] **Step 3: `tests/test_score.py`** (TDD) — test each scorer with a correct output, an incorrect output, and an edge case (code with markdown fences stripped; arithmetic with prose around the number; JSON with whitespace). Verify red then green.

- [ ] **Step 4: Commit**

```bash
git add eval tests
git commit -m "feat: 20-prompt objective eval and scoring rules"
```

---

### Task 3: Benchmark + quality harnesses

**Files:** `bench/run_bench.py`, `eval/run_quality.py`, `scripts/run_all.sh`.

- [ ] **Step 1: `bench/run_bench.py`** — for each quant model file:
  - runs `llama-bench -m <file> -p 512 -n 128 -r 3` (fixed, fair params; `-r 3` for stability),
  - parses the output for prompt-eval tok/s + generation tok/s (llama-bench prints a table; also try `-o json` if the installed version supports it),
  - computes TTFT ≈ prompt-eval latency for a fixed 32-token prompt (from llama-bench's prompt-eval time),
  - returns a dict per quant. Prints a summary.

- [ ] **Step 2: `eval/run_quality.py`** — for each quant:
  - for each of the 20 prompts: `llama-cli -m <file> -p <prompt> -n <max_tokens> --temp 0 --seed 42 -no-cnv` (capture stdout),
  - records the raw output + per-prompt pass/fail via `eval/score.py`,
  - measures peak RSS by wrapping the whole per-quant loop in `/usr/bin/time -l python ... ` (or by running each quant's llama-cli under `/usr/bin/time -l` and taking max),
  - writes `results/quality_<quant>.json`.

- [ ] **Step 3: Smoke test both harnesses on ONE quant with tiny params** (1-2 prompts, `-n` small) to prove the plumbing before the full run. Run the full `tests/test_score.py` too.

- [ ] **Step 4: Commit**

```bash
git add bench eval scripts
git commit -m "feat: llama-bench and quality-eval harnesses"
```

---

### Task 4: Full benchmark run (the measurement)

**Files:** `results/results.json`.

- [ ] **Step 1: Run the full benchmark** (~20-40 min for 4 quants × (bench + 20 quality prompts)):

```bash
.venv/bin/python bench/run_bench.py
.venv/bin/python eval/run_quality.py
```
(Or a `scripts/run_all.sh` that runs both and merges.)

- [ ] **Step 2: Merge into `results/results.json`** — quant, file size, prompt tok/s, gen tok/s, TTFT, peak RAM, quality % (overall + per category). Paste the actual measured values.

- [ ] **Step 3: Sanity-check the numbers** — lower quants should be faster/leaner; quality should degrade (F16 ≥ Q8 ≥ Q5 ≥ Q4, not always strictly but the trend should be visible). If a quant looks anomalous (e.g. Q4 quality way above F16), investigate.

- [ ] **Step 4: Commit**

```bash
git add results/
git commit -m "feat: benchmark results (real measurements)"
```

---

### Task 5: Chart + README

**Files:** `plot.py`, `README.md`.

- [ ] **Step 1: `plot.py`** — matplotlib scatter/line: X = quality % (overall), Y = generation tok/s (or TTFT), one point per quant, annotated with the quant name; save `results/quant_vs_quality.png`. Run it.

- [ ] **Step 2: `README.md`** with:
  1. One-liner + what/why.
  2. Hardware + versions (M4 / 16GB / Metal, llama.cpp version, model files + hashes).
  3. **Results table** (from `results.json` — derive programmatically or paste EXACT values): quant | size | prompt tok/s | gen tok/s | TTFT | peak RAM | quality %.
  4. **matplotlib chart** embedded (`results/quant_vs_quality.png`).
  5. **Conclusion** — the production recommendation (which quant for high-throughput low-latency and why), framed as the "Performance Optimization" duty (e.g. Q4_K_M for max throughput, Q5_K_M as the quality/speed sweet spot, F16 only when RAM allows and accuracy is non-negotiable — based on the ACTUAL numbers).
  6. Reproducibility (commands, pinned versions) + trade-offs/limitations.

- [ ] **Step 3: Verify** — table numbers match `results.json` exactly; chart file exists; conclusion references the real measurements.

- [ ] **Step 4: Commit**

```bash
git add plot.py README.md results/
git commit -m "docs: results table, matplotlib chart, and production recommendation"
```

---

### Task 6: Push + Pages (optional) + final review

- [ ] **Step 1: Push**

```bash
gh repo create llm-quantization-bench --public --source . --remote origin --push
```

- [ ] **Step 2: Optional Pages** — the README has a chart + mermaid-friendly content; follow the established pattern (`_config.yml`, `_includes/head-custom.html`, enable Pages) if the user wants it.

- [ ] **Step 3: Final verification** — health stack (pytest, flake8, mypy), repo pushed, results.json committed, README table/chart/conclusion accurate.
