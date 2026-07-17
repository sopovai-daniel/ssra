"""Unit tests for scripts/needle_gen.py (M2 G2-lite assignment §2 V5).

Coverage required by the assignment: exact target sequence-length control;
depth placement; tokenizer round-trip of the key digits (frozen tokenizer
sha 019568a2...); determinism under the suite seed.
"""

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from needle_gen import (KEY_RE, build_prompt, draw_key,  # noqa: E402
                        generate_suite, trial_sha256)

TOKENIZER = ROOT / "artifacts" / "tokenizer" / "fineweb-edu-bpe-16384.json"
TOKENIZER_SHA = ("019568a206fe6ccc4bc2e90c750d660979d3fd3add159e302"
                 "a0dfa4be0d669a0")
SEED = 20260717


@pytest.fixture(scope="module")
def tok():
    assert hashlib.sha256(TOKENIZER.read_bytes()).hexdigest() == TOKENIZER_SHA
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOKENIZER))


def test_exact_length_control(tok):
    """Prompt token count is EXACTLY n - headroom for every cell shape."""
    for n in (1024, 2048, 32768):
        for depth in (0.1, 0.5, 0.9):
            p = build_prompt(tok, n, depth, key=12345, headroom=16)
            assert len(p["ids"]) == n - 16


def test_depth_placement(tok):
    """Needle start lands at round(depth * budget) (grid depths never
    clamp at N >= 1024: preamble+question+needle fit well inside)."""
    for n in (1024, 4096, 32768):
        budget = n - 16
        for depth in (0.1, 0.5, 0.9):
            p = build_prompt(tok, n, depth, key=54321, headroom=16)
            assert p["needle_start"] == round(depth * budget)


def test_key_roundtrip(tok):
    """The key survives tokenize -> decode and is recovered by the SAME
    first-5-digit rule the M2 metric applies to generated text."""
    for key in (10000, 12345, 90210, 99999):
        p = build_prompt(tok, 1024, 0.5, key=key, headroom=16)
        decoded = tok.decode(p["ids"])
        m = KEY_RE.search(decoded)
        assert m is not None and m.group(0) == str(key)
        # exactly the needle's two key mentions carry digits
        assert decoded.count(str(key)) == 2


def test_prompt_ends_with_question(tok):
    p = build_prompt(tok, 1024, 0.5, key=11111, headroom=16)
    assert tok.decode(p["ids"]).endswith("What is the pass key? "
                                         "The pass key is")


def test_keys_are_five_digit():
    for n in (1024, 32768):
        for t in range(50):
            k = draw_key(SEED, n, 0.5, t)
            assert 10000 <= k <= 99999


def test_determinism_under_seed(tok):
    """Same seed -> byte-identical suite; different seed -> different keys."""
    kw = dict(grid=[1024], depths=[0.1, 0.9], trials=3, headroom=16)
    a = [trial_sha256(r["ids"]) for r in generate_suite(tok, SEED, **kw)]
    b = [trial_sha256(r["ids"]) for r in generate_suite(tok, SEED, **kw)]
    c = [r["key"] for r in generate_suite(tok, SEED + 1, **kw)]
    base = [r["key"] for r in generate_suite(tok, SEED, **kw)]
    assert a == b
    assert base != c


def test_budget_too_small_raises(tok):
    with pytest.raises(ValueError, match="too small"):
        build_prompt(tok, 32, 0.5, key=12345, headroom=16)
