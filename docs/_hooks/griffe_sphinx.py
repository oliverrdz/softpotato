"""Griffe extension that converts Sphinx/RST inline roles and math blocks in docstrings to Markdown."""

from __future__ import annotations

import re
from typing import Any

import griffe

_INLINE_MATH_RE = re.compile(r":math:`([^`]+)`")
_XREF_ROLE_RE = re.compile(
    r":(?:func|class|mod|data|attr|meth|exc|obj):`(~?)([^`<>]+?)(?:\s*<([^>]+)>)?`"
)
_DOUBLE_BACKTICK_RE = re.compile(r"``([^`\n]+)``")
_MATH_DIRECTIVE_RE = re.compile(
    r"(?m)^([ \t]*)\.\.\s+math::[ \t]*\n((?:\1[ \t]+[^\n]*\n|[ \t]*\n)+)"
)


def _convert_math_directive(match: re.Match[str]) -> str:
    indent = match.group(1)
    body = match.group(2)
    lines = body.splitlines()
    # Determine minimum indentation of non-empty lines in the math block
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return ""
    min_indent = min(len(line) - len(line.lstrip()) for line in non_empty)
    dedented = [
        f"{indent}{line[min_indent:]}" if line.strip() else "" for line in lines
    ]
    math_content = "\n".join(dedented).strip()
    return f"{indent}$$\n{indent}{math_content}\n{indent}$$\n\n"


def _convert_xref(match: re.Match[str]) -> str:
    tilde, label_or_target, explicit_target = (
        match.group(1),
        match.group(2).strip(),
        match.group(3),
    )
    target = explicit_target.strip() if explicit_target else label_or_target
    if explicit_target:
        display = label_or_target
    elif tilde:
        display = target.split(".")[-1]
    else:
        display = target
    return (
        f"`` `{display}` ``"
        if not target.startswith("softpotato")
        else f"[`{display}`][{target}]"
    )


def convert_rst_to_markdown(text: str) -> str:
    """Convert common Sphinx RST roles and math directives in a docstring to Markdown."""
    # 1. Convert .. math:: blocks to $$ ... $$
    text = _MATH_DIRECTIVE_RE.sub(_convert_math_directive, text)
    # 2. Convert :math:`...` to $...$
    text = _INLINE_MATH_RE.sub(r"$\1$", text)

    # 3. Convert Sphinx cross-reference roles (:func:, :class:, etc.)
    def _xref_repl(m: re.Match[str]) -> str:
        tilde, label_or_target, explicit_target = (
            m.group(1),
            m.group(2).strip(),
            m.group(3),
        )
        target = explicit_target.strip() if explicit_target else label_or_target
        display = (
            label_or_target
            if explicit_target
            else (target.split(".")[-1] if tilde else target)
        )
        if target.startswith("softpotato"):
            return f"[`{display}`][{target}]"
        return f"`{display}`"

    text = _XREF_ROLE_RE.sub(_xref_repl, text)
    # 4. Convert double backticks ``foo`` to single backticks `foo`
    text = _DOUBLE_BACKTICK_RE.sub(r"`\1`", text)
    return text


class SphinxDocstringExtension(griffe.Extension):
    """Automatically converts RST math and roles inside Python docstrings to Markdown."""

    def on_instance(self, *, obj: griffe.Object, **kwargs: Any) -> None:
        if obj.docstring and obj.docstring.value:
            obj.docstring.value = convert_rst_to_markdown(obj.docstring.value)
