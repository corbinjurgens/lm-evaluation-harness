"""Host file and text helpers for the benchmark runner.

Standard library only.
"""

from __future__ import annotations

import codecs
import json
import re
from pathlib import Path


REDACTED = "[REDACTED]"


def private_text(path, text):
    with Path(path).open("x", encoding="utf-8") as stream:
        Path(path).chmod(0o600)
        stream.write(text)


def json_read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def safe_path(path):
    path = Path(path).expanduser().absolute()
    if any(c in str(path) for c in (",", "\n", "\r", "\x00")):
        raise ValueError(
            "Docker mount paths cannot contain commas or control characters"
        )
    # Reject symlink ancestors as well as the leaf, including dangling links.
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Use real paths, not symlinks")
    return path


def _secret_pattern(secrets):
    """Match any secret, preferring the longest where several start together."""
    secrets = sorted({secret for secret in secrets if secret}, key=len, reverse=True)
    return re.compile("|".join(map(re.escape, secrets))) if secrets else None


def redact(text, secrets):
    pattern = _secret_pattern(secrets)
    return text if pattern is None else pattern.sub(REDACTED, text)


class StreamRedactor:
    """Decode and redact a byte stream chunk by chunk.

    The last ``max(len(secret)) - 1`` characters are held back until the next
    chunk or the end of the stream, so a secret split across chunks is still
    redacted; the incremental decoder keeps split UTF-8 characters intact.
    """

    def __init__(self, secrets):
        self.pattern = _secret_pattern(secrets)
        self.hold = max((len(s) for s in secrets if s), default=1) - 1
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.pending = ""

    def feed(self, chunk, final=False):
        text = self.pending + self.decoder.decode(chunk, final)
        cut = len(text) if final else max(0, len(text) - self.hold)
        parts, start = [], 0
        if self.pattern is not None:
            for match in self.pattern.finditer(text):
                if match.start() >= cut:
                    break
                parts += [text[start : match.start()], REDACTED]
                start = match.end()
        # A complete secret that straddles the cut is emitted redacted in full.
        cut = max(cut, start)
        parts.append(text[start:cut])
        self.pending = text[cut:]
        return "".join(parts)

    def finish(self):
        return self.feed(b"", final=True)
