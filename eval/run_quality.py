"""Run the 20 eval prompts per quant via llama-cli; score + record latency/RAM."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.score import load_answers, load_prompts, score_output  # noqa: E402

MODELS = [
    ("F16", "models/Qwen2.5-3B-Instruct-f16.gguf"),
    ("Q8_0", "models/Qwen2.5-3B-Instruct-Q8_0.gguf"),
    ("Q5_K_M", "models/Qwen2.5-3B-Instruct-Q5_K_M.gguf"),
    ("Q4_K_M", "models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"),
]


def extract_response(stdout: str, prompt: str) -> str:
    """Strip llama-cli's chat banner + timing stats, leaving the model's reply.

    In -st mode llama-cli echoes the prompt as "> <prompt>" followed by the
    reply, then prints "[ Prompt: ... | Generation: ... ]" stats.
    """
    s = stdout.split("[ Prompt:", 1)[0]
    idx = s.rfind("> " + prompt)
    if idx != -1:
        return s[idx + len(prompt) + 2:]
    idx = s.rfind("\n> ")
    if idx != -1:
        return s[idx + 3:]
    return s


def _parse_rss(stderr: str) -> int:
    """Parse macOS /usr/bin/time -l 'maximum resident set size' (bytes) -> MB."""
    m = re.search(r"([0-9]+)\s+maximum resident set size", stderr)
    if not m:
        return 0
    return int(m.group(1)) // (1024 * 1024)


def run_prompt(model: str, prompt: str, max_tokens: int) -> tuple[str, int]:
    out = subprocess.run(
        ["/usr/bin/time", "-l", "llama-cli", "-m", model, "-p", prompt,
         "-n", str(max_tokens), "--temp", "0", "--seed", "42", "-st",
         "--no-display-prompt"],
        capture_output=True, text=True, timeout=300,
    )
    rss_mb = _parse_rss(out.stderr)
    return extract_response(out.stdout, prompt), rss_mb


def run_quant(model: str, prompts: list[dict], answers: dict) -> dict:
    results = []
    passed = 0
    peak_ram_mb = 0
    for p in prompts:
        output, rss_mb = run_prompt(model, p["prompt"], p["max_tokens"])
        peak_ram_mb = max(peak_ram_mb, rss_mb)
        ok = score_output(output, answers[p["id"]])
        passed += ok
        results.append({"id": p["id"], "category": p["category"], "pass": ok,
                        "output": output.strip()})
    return {"passed": passed, "total": len(prompts), "score": passed / len(prompts),
            "peak_ram_mb": peak_ram_mb, "results": results}


def main() -> None:
    prompts = load_prompts()
    answers = load_answers()
    Path("results").mkdir(exist_ok=True)
    for name, path in MODELS:
        if not Path(path).exists():
            print(f"SKIP {name}")
            continue
        print(f"quality {name}...")
        data = run_quant(path, prompts, answers)
        (Path("results") / f"quality_{name}.json").write_text(json.dumps(data, indent=2))
        print(f"  {data['passed']}/{data['total']} ({data['score']:.2%}), "
              f"peak {data['peak_ram_mb']} MB")


if __name__ == "__main__":
    main()
