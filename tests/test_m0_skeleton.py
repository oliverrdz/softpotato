from __future__ import annotations

import re
import sys
from importlib import metadata, resources
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="strict")


def _ensure_src_on_path() -> None:
    """Allow running `pytest` from a clean checkout without requiring editable install."""
    root = _repo_root()
    src = root / "src"
    if src.is_dir():
        sys.path.insert(0, str(src))


def _import_softpotato():
    try:
        import softpotato  # type: ignore

        return softpotato
    except ModuleNotFoundError:
        _ensure_src_on_path()
        try:
            import softpotato  # type: ignore

            return softpotato
        except ModuleNotFoundError as e:
            root = _repo_root()
            hint = (
                "Could not import 'softpotato'. Expected either:\n"
                "  (a) editable install: `pip install -e .`\n"
                "  (b) src layout present: `src/softpotato/__init__.py`\n"
                f"Repo root: {root}\n"
                f"Checked for: {root / 'src' / 'softpotato' / '__init__.py'}"
            )
            raise AssertionError(hint) from e


def _find_ci_workflow(root: Path) -> Path:
    candidates = [
        root / ".github" / "workflows" / "ci.yml",
        root / ".github" / "workflows" / "ci.yaml",
        root / ".github" / "workflows" / "tests.yml",
        root / ".github" / "workflows" / "tests.yaml",
    ]
    for p in candidates:
        if p.is_file():
            return p
    pytest.fail(
        "Missing CI workflow. Expected one of: "
        + ", ".join(str(p.relative_to(root)) for p in candidates)
    )
    raise AssertionError


# --- Acceptance: Install + Tests (import + version) ---


def test_imports_and_exposes_version_string() -> None:
    """
    AC:
      - Install: `python -c "import softpotato; print(softpotato.__version__)"` prints a version string.
      - Tests: at least one smoke test asserts import + expected top-level symbols.
    """
    softpotato = _import_softpotato()

    assert hasattr(softpotato, "__version__"), "softpotato.__version__ must exist"
    v = softpotato.__version__
    assert isinstance(v, str), "__version__ must be a string"
    assert v.strip(), "__version__ must be non-empty"
    assert re.match(
        r"^\d+\.\d+\.\d+(" r"(a|b|rc)\d+|" r"([.-][0-9A-Za-z.]+)" r")?$",
        v,
    ), f"Unexpected version: {v}"


def test_runtime_version_matches_distribution_metadata() -> None:
    """
    AC:
      - Install: printed version should be consistent with installed package metadata.
    Notes:
      - If you're running pytest without `pip install -e .`, skip (CI should install).
    """
    softpotato = _import_softpotato()
    try:
        dist_v = metadata.version("softpotato")
    except metadata.PackageNotFoundError:
        pytest.skip(
            "softpotato distribution metadata not found; run `pip install -e .` to enforce this locally"
        )

    assert softpotato.__version__ == dist_v, (
        "softpotato.__version__ should match the installed distribution version "
        f"(module={softpotato.__version__!r}, dist={dist_v!r})"
    )


# --- Acceptance: Lint/Types intent (typed marker) ---


def test_package_is_typed_marker_present() -> None:
    """
    AC:
      - Lint / Format / Types: skeleton intends to be type-checked; typed marker should exist.
    """
    _import_softpotato()
    root = resources.files("softpotato")
    marker = root.joinpath("py.typed")
    assert marker.is_file(), "Missing 'py.typed' marker in softpotato package"


# --- Acceptance: Docs skeleton ---


def test_docs_skeleton_exists() -> None:
    """
    AC:
      - Documentation skeleton: docs/ exists with minimal landing page + mkdocs stub.
    """
    root = _repo_root()
    mkdocs_yml = root / "mkdocs.yml"
    docs_index = root / "docs" / "index.md"

    assert mkdocs_yml.is_file(), "mkdocs.yml missing (docs build config stub required)"
    assert docs_index.is_file(), "docs/index.md missing (minimal landing page required)"
    assert _read_text(docs_index).strip(), "docs/index.md should not be empty"


# --- Acceptance: CI workflow presence + gates ---


@pytest.mark.parametrize("needle", ["push", "pull_request"])
def test_ci_workflow_triggers_present(needle: str) -> None:
    """
    AC:
      - CI: GitHub Actions workflow runs on push + pull_request.
    """
    root = _repo_root()
    wf = _find_ci_workflow(root)
    text = _read_text(wf)

    assert (
        "on:" in text or "\non\n" in text
    ), "Workflow should define triggers under `on:`"
    assert (
        needle in text
    ), f"Workflow trigger '{needle}' not found in {wf.relative_to(root)}"


@pytest.mark.parametrize(
    "cmd_fragment", ["ruff check", "black --check", "mypy", "pytest"]
)
def test_ci_workflow_includes_quality_gates(cmd_fragment: str) -> None:
    """
    AC:
      - CI: workflow enforces ruff, black, mypy, pytest.
    """
    root = _repo_root()
    wf = _find_ci_workflow(root)
    text = _read_text(wf)

    assert cmd_fragment in text, (
        f"Expected CI workflow to run '{cmd_fragment}' "
        f"(missing in {wf.relative_to(root)})"
    )


# --- Acceptance: Packaging/toolchain invariants (PEP 517/518 + python>=3.10) ---


def test_pyproject_declares_hatchling_and_python310_plus() -> None:
    """
    AC:
      - Constraints: Python >=3.10, packaging via PEP 517/518 (hatchling).
    """
    try:
        import tomllib  # py>=3.11
    except ModuleNotFoundError:  # py<=3.10
        import tomli as tomllib

    root = _repo_root()
    pyproject = root / "pyproject.toml"
    assert pyproject.is_file(), "pyproject.toml missing"

    data = tomllib.loads(_read_text(pyproject))

    build = data.get("build-system", {})
    assert (
        build.get("build-backend") == "hatchling.build"
    ), "Expected hatchling build backend"
    requires = build.get("requires", [])
    assert any(
        str(req).startswith("hatchling") for req in requires
    ), "Expected hatchling in build requirements"

    project = data.get("project", {})
    assert (
        project.get("requires-python") == ">=3.10"
    ), "Expected requires-python to be '>=3.10'"
