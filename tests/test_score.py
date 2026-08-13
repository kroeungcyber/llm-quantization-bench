"""Tests for the deterministic objective scoring rules in eval/score.py."""
from __future__ import annotations

import eval.score as score

FENCE_IS_PALINDROME = '''```python
def is_palindrome(s):
    return s == s[::-1]
```'''

PLAIN_IS_PALINDROME = """def is_palindrome(s):
    return s == s[::-1]"""

WRONG_IS_PALINDROME = """def is_palindrome(s):
    return len(s)"""

THREE_ITEMS = "1. Backup to external media\n2. Protect against ransomware\n3. Recover from disasters"


def test_arithmetic_correct_with_prose():
    assert score.score_arithmetic("The answer is 1081.", 1081) is True


def test_arithmetic_incorrect():
    assert score.score_arithmetic("1080", 1081) is False


def test_arithmetic_edge_no_number():
    assert score.score_arithmetic("I don't know.", 1081) is False


def test_arithmetic_edge_decimal_tolerance():
    assert score.score_arithmetic("48.0 km/h", 48) is True


def test_exact_fact_correct_with_prose():
    assert score.score_exact_fact("The capital of Australia is Canberra.", "canberra") is True


def test_exact_fact_incorrect():
    assert score.score_exact_fact("Sydney", "canberra") is False


def test_exact_fact_edge_punctuation_and_case():
    assert score.score_exact_fact("AU.", "au") is True


def test_exact_fact_edge_empty():
    assert score.score_exact_fact("", "canberra") is False


def test_code_correct_plain():
    assert score.score_code(PLAIN_IS_PALINDROME, "is_palindrome",
                            [["racecar", True], ["hello", False], ["", True]]) is True


def test_code_wrong_function():
    assert score.score_code(WRONG_IS_PALINDROME, "is_palindrome",
                            [["racecar", True], ["hello", False], ["", True]]) is False


def test_code_markdown_fences_stripped():
    assert score.score_code(FENCE_IS_PALINDROME, "is_palindrome",
                            [["racecar", True], ["hello", False], ["", True]]) is True


def test_code_edge_not_python():
    assert score.score_code("this is not python code !!!", "is_palindrome",
                            [["racecar", True]]) is False


def test_code_edge_missing_function():
    assert score.score_code("def other():\n    return 1", "is_palindrome",
                            [["racecar", True]]) is False


def test_code_edge_function_raises():
    assert score.score_code("def is_palindrome(s):\n    raise ValueError('boom')", "is_palindrome",
                            [["racecar", True]]) is False


def test_instruction_three_items_ok():
    assert score.score_instruction(THREE_ITEMS, "three_items") is True


def test_instruction_three_items_only_two():
    assert score.score_instruction("1. One\n2. Two", "three_items") is False


def test_instruction_exact_json_ok():
    assert score.score_instruction('{"status": "ok"}', "exact_json") is True


def test_instruction_exact_json_whitespace():
    assert score.score_instruction('  {"status": "ok"}\n', "exact_json") is True


def test_instruction_exact_json_wrong():
    assert score.score_instruction('{"status": "fail"}', "exact_json") is False


def test_instruction_exact_json_not_json():
    assert score.score_instruction("status ok", "exact_json") is False


def test_instruction_word_repeat_ok():
    assert score.score_instruction("banana banana banana", "word_repeat") is True


def test_instruction_word_repeat_wrong():
    assert score.score_instruction("banana banana", "word_repeat") is False


def test_instruction_sorted_list_ok():
    assert score.score_instruction("1, 3, 5, 7, 9", "sorted_list") is True


def test_instruction_sorted_list_wrong():
    assert score.score_instruction("1,2,3", "sorted_list") is False


def test_instruction_single_word_ok():
    assert score.score_instruction("yes", "single_word") is True


def test_instruction_single_word_two_words():
    assert score.score_instruction("yes no", "single_word") is False


def test_instruction_unknown_rule():
    assert score.score_instruction("anything", "no_such_rule") is False


def test_score_output_dispatches_all_types():
    assert score.score_output("1081", {"type": "arithmetic", "expected": 1081}) is True
    assert score.score_output("Canberra", {"type": "exact-fact", "expected": "canberra"}) is True
    assert score.score_output(FENCE_IS_PALINDROME,
                              {"type": "code", "func": "is_palindrome",
                               "tests": [["racecar", True], ["hello", False], ["", True]]}) is True
    assert score.score_output(THREE_ITEMS, {"type": "instruction", "rule": "three_items"}) is True


def test_score_output_unknown_type():
    assert score.score_output("anything", {"type": "unknown"}) is False


def test_prompts_are_exactly_twenty_in_four_categories():
    prompts = score.load_prompts()
    assert len(prompts) == 20
    counts: dict[str, int] = {}
    for p in prompts:
        assert {"id", "category", "prompt", "max_tokens"} <= set(p)
        counts[p["category"]] = counts.get(p["category"], 0) + 1
    assert counts == {"arithmetic": 5, "exact-fact": 5, "code": 5, "instruction": 5}
    assert len({p["id"] for p in prompts}) == 20


def test_answers_cover_all_prompts():
    prompts = score.load_prompts()
    answers = score.load_answers()
    assert set(answers) == {p["id"] for p in prompts}


def test_every_scorer_returns_bool():
    answers = score.load_answers()
    for pid, answer in answers.items():
        for out in ("", "junk", "42"):
            assert isinstance(score.score_output(out, answer), bool), pid


def test_code_invert_dict_int_keys_via_answers():
    answer = score.load_answers()["code_5"]
    code = "def invert_dict(d):\n    return {v: k for k, v in d.items()}"
    assert score.score_output(code, answer) is True


def test_code_invert_dict_wrong_keys():
    answer = score.load_answers()["code_5"]
    code = "def invert_dict(d):\n    return d"
    assert score.score_output(code, answer) is False


def test_code_fizzbuzz_via_answers():
    answer = score.load_answers()["code_3"]
    code = ("def fizzbuzz(n):\n"
            "    out = []\n"
            "    for i in range(1, n + 1):\n"
            "        if i % 15 == 0:\n"
            "            out.append('FizzBuzz')\n"
            "        elif i % 3 == 0:\n"
            "            out.append('Fizz')\n"
            "        elif i % 5 == 0:\n"
            "            out.append('Buzz')\n"
            "        else:\n"
            "            out.append(str(i))\n"
            "    return out")
    assert score.score_output(code, answer) is True


def test_every_answer_passes_its_gold_solution():
    answers = score.load_answers()
    assert score.score_output("1081", answers["arith_1"])
    assert score.score_output("48", answers["arith_2"])
    assert score.score_output("36", answers["arith_3"])
    assert score.score_output("1024", answers["arith_4"])
    assert score.score_output("25", answers["arith_5"])
    assert score.score_output("Canberra is the capital.", answers["fact_1"])
    assert score.score_output("It ended in 1945.", answers["fact_2"])
    assert score.score_output("Au", answers["fact_3"])
    assert score.score_output("William Shakespeare wrote it.", answers["fact_4"])
    assert score.score_output("Jupiter", answers["fact_5"])
    assert score.score_output(
        "1. Backup\n2. Security\n3. Recovery", answers["inst_1"])
    assert score.score_output('{"status": "ok"}', answers["inst_2"])
    assert score.score_output("banana banana banana", answers["inst_3"])
    assert score.score_output("1, 3, 5, 7, 9", answers["inst_4"])
    assert score.score_output("yes", answers["inst_5"])
