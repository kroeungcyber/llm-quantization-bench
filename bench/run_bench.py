"""Run llama-bench per quant; return throughput + TTFT."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

MODELS = [
    ("F16", "models/Qwen2.5-3B-Instruct-f16.gguf"),
    ("Q8_0", "models/Qwen2.5-3B-Instruct-Q8_0.gguf"),
    ("Q5_K_M", "models/Qwen2.5-3B-Instruct-Q5_K_M.gguf"),
    ("Q4_K_M", "models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"),
]
# Fixed, fair: same prompt length (512) and generation length (128) for every quant.
BENCH_ARGS = ["-p", "512", "-n", "128", "-r", "3"]
# TTFT is reported for a fixed small prompt; llama-bench measures prompt-eval at -p tokens,
# so scale per-token latency to a 32-token prompt.
TTFT_TOKENS = 32


def _parse_json(out: str) -> list[dict]:
    try:
        entries = json.loads(out)
    except json.JSONDecodeError:
        return []
    return entries if isinstance(entries, list) else []


def _parse_regex(out: str) -> tuple[list[float], list[float]]:
    # Fallback for versions without -o json: parse the text table (t/s is last column).
    prompt_evals = [float(x) for x in re.findall(r"prompt eval.*?\|\s+(\d+(?:\.\d+)?)\s+", out)]
    gens = [float(x) for x in re.findall(r"generation.*?\|\s+(\d+(?:\.\d+)?)\s+", out)]
    return prompt_evals, gens


def run_bench(model: str) -> dict:
    out = subprocess.run(
        ["llama-bench", "-m", model, *BENCH_ARGS, "-o", "json"],
        capture_output=True, text=True, timeout=900,
    ).stdout
    entries = _parse_json(out)
    if entries:
        prompt_ts = [e["avg_ts"] for e in entries if e.get("n_prompt", 0) > 0 and e.get("n_gen", 0) == 0]
        gen_ts = [e["avg_ts"] for e in entries if e.get("n_gen", 0) > 0 and e.get("n_prompt", 0) == 0]
        prompt_tok = max(prompt_ts) if prompt_ts else None
        gen_tok = max(gen_ts) if gen_ts else None
        ttft_ms = None
        for e in entries:
            if e.get("n_prompt", 0) > 0 and e.get("n_gen", 0) == 0:
                ttft_ms = e["avg_ns"] / e["n_prompt"] * TTFT_TOKENS / 1e6
                break
    else:
        prompt_evals, gens = _parse_regex(out)
        prompt_tok = max(prompt_evals) if prompt_evals else None
        gen_tok = max(gens) if gens else None
        ttft_ms = None
    return {"prompt_tok_s": prompt_tok, "gen_tok_s": gen_tok, "ttft_ms": ttft_ms, "raw": out[-2000:]}


def main() -> None:
    results = {}
    for name, path in MODELS:
        if not Path(path).exists():
            print(f"SKIP {name}: {path} missing")
            continue
        print(f"bench {name}...")
        results[name] = run_bench(path)
        r = results[name]
        print(f"  {r['prompt_tok_s']} prompt tok/s, {r['gen_tok_s']} gen tok/s, {r['ttft_ms']} ms TTFT")
    Path("results").mkdir(exist_ok=True)
    (Path("results") / "bench_raw.json").write_text(json.dumps(results, indent=2))
    print("wrote results/bench_raw.json")


if __name__ == "__main__":
    main()
