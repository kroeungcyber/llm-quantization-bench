"""Objective scoring for the 20-prompt quality eval."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).parent


def load_prompts() -> list[dict]:
    return json.loads((ROOT / "prompts.json").read_text())["prompts"]


def load_answers() -> dict:
    return json.loads((ROOT / "answers.json").read_text())


def _first_number(text: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(m.group()) if m else None


def score_arithmetic(output: str, expected: float) -> bool:
    got = _first_number(output)
    return got is not None and abs(got - expected) < 0.01


def score_exact_fact(output: str, expected: str) -> bool:
    norm = re.sub(r"[^a-z0-9]", "", output.lower())
    exp = re.sub(r"[^a-z0-9]", "", expected.lower())
    return exp in norm


def _extract_code(output: str) -> str:
    # strip markdown fences and surrounding prose
    m = re.search(r"```(?:python)?\s*\n(.*?)```", output, re.DOTALL)
    if m:
        return m.group(1)
    return output


def _norm_key(k: object) -> object:
    if isinstance(k, str):
        try:
            return int(k)
        except ValueError:
            return k
    return k


def _values_equal(a: object, b: object) -> bool:
    # JSON cannot represent int dict keys, so compare dicts with int-like string keys
    # normalized on both sides (e.g. invert_dict {"a": 1} -> {1: "a"}).
    if isinstance(a, dict) and isinstance(b, dict):
        na = {_norm_key(k): v for k, v in a.items()}
        nb = {_norm_key(k): v for k, v in b.items()}
        if set(na) != set(nb):
            return False
        return all(_values_equal(na[k], nb[k]) for k in na)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b))
    return a == b


def score_code(output: str, func: str, tests: list[list]) -> bool:
    code = _extract_code(output)
    ns: dict = {}
    try:
        exec(code, ns)
    except Exception:
        return False
    fn = ns.get(func)
    if not callable(fn):
        return False
    for test in tests:
        args, expected = test[:-1], test[-1]
        try:
            got = fn(*args)
        except Exception:
            return False
        if not _values_equal(got, expected):
            return False
    return True


def score_instruction(output: str, rule: str) -> bool:
    out = output.strip()
    if rule == "three_items":
        lines = [line for line in out.splitlines() if line.strip()]
        return len(lines) == 3 and all(re.match(r"^\s*\d+[\.\)]", line) for line in lines)
    if rule == "exact_json":
        try:
            return json.loads(out) == {"status": "ok"}
        except Exception:
            return False
    if rule == "word_repeat":
        words = out.split()
        return words == ["banana"] * 3
    if rule == "sorted_list":
        nums = re.findall(r"\d+", out)
        return nums == ["1", "3", "5", "7", "9"]
    if rule == "single_word":
        return len(out.split()) == 1
    return False


def score_output(output: str, answer: dict) -> bool:
    t = answer["type"]
    if t == "arithmetic":
        return score_arithmetic(output, answer["expected"])
    if t == "exact-fact":
        return score_exact_fact(output, answer["expected"])
    if t == "code":
        return score_code(output, answer["func"], answer["tests"])
    if t == "instruction":
        return score_instruction(output, answer["rule"])
    return False
