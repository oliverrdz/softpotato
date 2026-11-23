"""MkDocs hook to rewrite relative .md and .ipynb links in Jupyter notebooks."""

from __future__ import annotations

import posixpath
import re
import urllib.parse

_FILES = None
_HREF_RE = re.compile(r'(<a\b[^>]*?\bhref=)(["\'])([^"\']*)\2', re.IGNORECASE)


def on_files(files, config):
    """Cache the MkDocs file collection for resolving notebook links."""
    global _FILES
    _FILES = files
    return files


def on_post_page(output: str, page, config) -> str:
    """Rewrite relative .md and .ipynb hrefs in rendered notebook HTML."""
    if _FILES is None or not page.file.src_path.endswith(".ipynb"):
        return output

    source_dir = posixpath.dirname(page.file.src_path)

    def _replace(match: re.Match[str]) -> str:
        prefix, quote, href = match.groups()
        if href.startswith(
            ("http://", "https://", "mailto:", "javascript:", "data:", "#", "/")
        ):
            return match.group(0)

        path_part, sep, fragment = href.partition("#")
        query_part = ""
        if "?" in path_part:
            path_part, q_sep, query_part = path_part.partition("?")
            query_part = q_sep + query_part

        decoded_path = urllib.parse.unquote(path_part)
        if not decoded_path.endswith((".md", ".ipynb")):
            return match.group(0)

        target_src = posixpath.normpath(posixpath.join(source_dir, decoded_path))
        target_file = _FILES.get_file_from_path(target_src)
        if target_file is None:
            return match.group(0)

        new_url = target_file.url_relative_to(page.file)
        if query_part:
            new_url += query_part
        if sep:
            new_url += sep + fragment
        return f"{prefix}{quote}{new_url}{quote}"

    return _HREF_RE.sub(_replace, output)
