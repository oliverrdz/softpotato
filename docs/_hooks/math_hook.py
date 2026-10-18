"""MkDocs hook to ensure equations inside $$ are properly rendered as blocks or inline math."""

import re


def fix_math(text: str) -> str:
    # 1. Standalone single-line $$math$$ on its own line (indented or not)
    text = re.sub(
        r"(?m)^([ \t]*)\$\$([^\n]+?)\$\$[ \t]*$",
        r"\n\n\1$$\n\1\2\n\1$$\n\n",
        text,
    )
    # 2. Inline $$math$$ with other text on the same line -> convert to single $math$
    text = re.sub(r"(?<=[^\n\$])\$\$([^\n\$]+?)\$\$(?=[^\n\$])", r"$\1$", text)
    text = re.sub(r"(?<=[^\n\$])\$\$([^\n\$]+?)\$\$", r"$\1$", text)
    text = re.sub(r"\$\$([^\n\$]+?)\$\$(?=[^\n\$])", r"$\1$", text)
    return text


def fix_math_outside_code(markdown: str) -> str:
    # Split text by fenced code blocks: ```...``` or ~~~...~~~
    parts = re.split(r"(?ms)(^```[^\n]*\n.*?^```[ \t]*$)", markdown)
    for i in range(0, len(parts), 2):
        parts[i] = fix_math(parts[i])
    return "".join(parts)


def on_page_markdown(markdown: str, page, config, files) -> str:
    return fix_math_outside_code(markdown)
