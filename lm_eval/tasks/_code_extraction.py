"""Conservative Markdown envelope removal, without rewriting Python programs."""

import ast
import io
import re
import tokenize


_FENCE = re.compile(r"^[ \t]{0,3}(?P<mark>`{3,}|~{3,})(?P<label>[^\r\n]*)\r?\n?$")
_PYTHON = {"", "python", "python3", "py"}
_LEAD_INS = {
    "here is the solution:",
    "here is the code:",
    "solution:",
    "python:",
    "code:",
}


def _parses(source: str) -> bool:
    try:
        ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    return True


def _physical_lines(source: str) -> list[str]:
    """Split Python's physical newlines, retaining their original spelling."""
    return re.findall(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+$", source)


def _string_lines(source: str) -> set[int]:
    """Protect even unfinished multiline literals from fence recognition."""
    # tokenize's readline consumes LF; Python source also permits CR and CRLF.
    # Normalize only this classification copy, never the emitted program.
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    line_count = len(_physical_lines(source))
    protected = set()
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.STRING or tokenize.tok_name[
                token.type
            ].startswith("FSTRING_"):
                protected.update(range(token.start[0], token.end[0] + 1))
    except tokenize.TokenError as error:
        if "multi-line string" in error.args[0]:
            protected.update(range(error.args[1][0], line_count + 1))
    except (IndentationError, SyntaxError) as error:
        # Classification must never turn malformed Python into repaired code.
        # Later Markdown prose cannot invalidate an earlier closing delimiter.
        protected.update(range(error.lineno or 1, line_count + 1))
    return protected


def extract_python(response: str, *, allow_body: bool = False) -> str:
    """Keep raw source or join closed outer Python blocks in source order.

    Only explicit lead-ins introduce a Markdown envelope. Other unfenced text
    stays untouched, including malformed code and prose. No string contents or
    program whitespace are normalized. A dangling accepted block rejects the
    entire envelope, rather than substituting an earlier answer.
    """
    if not response.strip():
        return ""
    if _parses(response):
        return response

    lines = _physical_lines(response)
    first = 0
    while first < len(lines) and (
        not lines[first].strip() or lines[first].strip().casefold() in _LEAD_INS
    ):
        first += 1
    if first == len(lines) or not _FENCE.fullmatch(lines[first]):
        # Older assistant-prefill clients returned a body and only its closing
        # fence. Do not recognize a fence inside a Python string as that suffix.
        protected = _string_lines(response)
        for index, line in enumerate(lines):
            fence = _FENCE.fullmatch(line)
            if fence and index + 1 not in protected:
                prefix = "".join(lines[:index])
                body = prefix.lstrip("\r\n")
                if not fence["label"].strip() and (
                    _parses(prefix) or (allow_body and body.startswith((" ", "\t")))
                ):
                    if any(_FENCE.fullmatch(tail) for tail in lines[index + 1 :]):
                        return response
                    return prefix
                break
        return response

    blocks = []
    index = first
    while index < len(lines):
        opening = _FENCE.fullmatch(lines[index])
        if not opening:
            index += 1
            continue
        mark, label = opening["mark"], opening["label"].strip().casefold()
        accepted = label in _PYTHON
        start = index + 1
        protected = _string_lines("".join(lines[start:])) if accepted else set()
        index = start
        while index < len(lines):
            closing = _FENCE.fullmatch(lines[index])
            if (
                closing
                and not closing["label"].strip()
                and closing["mark"][0] == mark[0]
                and len(closing["mark"]) >= len(mark)
                and index - start + 1 not in protected
            ):
                break
            index += 1
        if index == len(lines):
            return "" if accepted else "\n".join(blocks)
        if accepted:
            blocks.append("".join(lines[start:index]))
        index += 1
    return "\n".join(blocks)
