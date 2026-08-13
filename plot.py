"""Plot the quality-vs-throughput and quality-vs-RAM trade-off from results.json.

Two panels side by side, one point per quant (F16, Q8_0, Q5_K_M, Q4_K_M):
  - left:  X = quality %, Y = generation tok/s
  - right: X = quality %, Y = peak RAM (MB)
Values are loaded from results/results.json (the source of truth), never hardcoded.
Saves results/quant_tradeoffs.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

COLORS = {
    "F16": "#d62728",
    "Q8_0": "#ff7f0e",
    "Q5_K_M": "#2ca02c",
    "Q4_K_M": "#1f77b4",
}

REPO_ROOT = Path(__file__).resolve().parent
RESULTS = REPO_ROOT / "results" / "results.json"
OUT = REPO_ROOT / "results" / "quant_tradeoffs.png"

# Result order in the file; keeps annotation placement stable.
QUANT_ORDER = ["F16", "Q8_0", "Q5_K_M", "Q4_K_M"]


def main() -> None:
    data = json.loads(RESULTS.read_text())
    quants = data["quants"]
    names = [q for q in QUANT_ORDER if q in quants] or list(quants)

    quality = [quants[q]["quality_pct"] for q in names]
    gen_tok_s = [quants[q]["gen_tok_s"] for q in names]
    peak_ram_mb = [quants[q]["peak_ram_mb"] for q in names]

    fig, (ax_tp, ax_ram) = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.suptitle("Qwen 2.5 3B Instruct quantization trade-off (Apple M4, llama.cpp)",
                 fontsize=13, fontweight="bold")

    # Left: quality vs generation throughput.
    ax_tp.set_title("Higher quality vs generation throughput", fontsize=11)
    ax_tp.set_xlabel("Quality (% 20-prompt auto-score)")
    ax_tp.set_ylabel("Generation tok/s")
    ax_tp.set_xlim(80, 100)
    for q, x, y in zip(names, quality, gen_tok_s):
        ax_tp.scatter(x, y, s=90, color=COLORS[q], zorder=3)
        ax_tp.annotate(f"{q}\n{y} tok/s", (x, y), textcoords="offset points",
                       xytext=(12, 8), fontsize=9)
    ax_tp.grid(True, alpha=0.3, zorder=0)
    ax_tp.invert_xaxis()  # higher quality to the right reads naturally here

    # Right: quality vs peak RAM.
    ax_ram.set_title("Higher quality vs peak RAM", fontsize=11)
    ax_ram.set_xlabel("Quality (% 20-prompt auto-score)")
    ax_ram.set_ylabel("Peak RAM (MB)")
    ax_ram.set_xlim(80, 100)
    for q, x, y in zip(names, quality, peak_ram_mb):
        ax_ram.scatter(x, y, s=90, color=COLORS[q], zorder=3)
        ax_ram.annotate(f"{q}\n{y:,} MB", (x, y), textcoords="offset points",
                        xytext=(12, 8), fontsize=9)
    ax_ram.grid(True, alpha=0.3, zorder=0)
    ax_ram.invert_xaxis()

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
