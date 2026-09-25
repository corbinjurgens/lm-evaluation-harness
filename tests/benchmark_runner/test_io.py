"""Tests for benchmark_runner.io."""

from __future__ import annotations

import pytest

from benchmark_runner.io import StreamRedactor, redact


def stream(chunks, secrets):
    redactor = StreamRedactor(secrets)
    pieces = [redactor.feed(chunk) for chunk in chunks]
    pieces.append(redactor.finish())
    return pieces


def test_secret_split_across_two_chunks_is_redacted():
    pieces = stream([b"key=priv", b"ate-openai done\n"], ["private-openai"])
    assert "".join(pieces) == "key=[REDACTED] done\n"
    assert all("priv" not in piece for piece in pieces)


def test_multibyte_character_split_across_two_chunks_survives():
    encoded = "café █\n".encode()
    split = encoded.index(b"\xc3") + 1
    pieces = stream([encoded[:split], encoded[split:]], [])
    assert "".join(pieces) == "café █\n"
    assert "�" not in "".join(pieces)


def test_multibyte_split_and_secret_split_together():
    encoded = "éprivate-openaié".encode()
    chunks = [encoded[i : i + 1] for i in range(len(encoded))]
    assert "".join(stream(chunks, ["private-openai"])) == "é[REDACTED]é"


def test_complete_secret_straddling_hold_boundary_is_redacted_whole():
    # One chunk holding a whole secret that starts inside the emitted part.
    pieces = stream([b"ab-secret", b"!"], ["b-secret"])
    assert pieces[0] == "a[REDACTED]"
    assert "".join(pieces) == "a[REDACTED]!"


def test_longest_overlapping_secret_wins():
    assert redact("x-token-long y", ["token", "token-long"]) == "x-[REDACTED] y"


@pytest.mark.parametrize("secrets", [[], [""]])
def test_no_secrets_passes_text_through(secrets):
    assert "".join(stream([b"plain ", b"text"], secrets)) == "plain text"
    assert redact("plain", secrets) == "plain"
