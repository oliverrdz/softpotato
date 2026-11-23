# Contributing to Soft Potato

Thank you for your interest in contributing to **Soft Potato** (`softpotato`), an open-source Python library for 1D electrochemical simulations and analytical electrochemical equations. Whether you are reporting a bug, adding a new electrochemical mechanism, improving numerical solvers, or expanding tutorials and documentation, your contributions are welcome.

By participating in this project, you agree to abide by the repository's Code of Conduct and license terms (BSD 3-Clause).

---

## Generative AI Policy

Soft Potato v3+ was planned and developed using agentic AI workflows in **Antigravity**, and we welcome contributors who use generative AI tools (such as Antigravity, GitHub Copilot, Cursor, or LLMs) to assist with writing code, tests, and documentation.

To maintain scientific rigor and codebase quality, all AI-assisted contributions must follow these rules:

* **Human Ownership:** You are the sole author of record for every line of code you submit. You must read, understand, and be able to explain your changes during code review. *"The AI wrote it"* is not an acceptable explanation.
* **Transparent Disclosure:** If generative AI materially contributed to your pull request, disclose the tool or model used in your pull request description or via a commit trailer (for example, `Assisted-by: Antigravity`). Disclosure carries no penalty; it helps maintainers focus review on architectural fit and edge cases.
* **Scientific and Mathematical Verification:** LLMs frequently hallucinate equations, flip electrochemical sign conventions (such as IUPAC vs. US polarographic conventions), or introduce unit mismatches. You must manually verify all physics, kinetics, mass transport equations, and physical constants against standard electrochemistry literature or analytical solutions, and validate them with unit tests.
* **Local Verification:** Never submit unverified AI output. All existing and new tests, formatting checks, and linters must pass locally before you open a pull request.
* **Small, Focused Pull Requests:** Avoid massive, single-PR AI-generated refactors. Keep pull requests atomic and scoped to a single issue. If you plan a large AI-assisted architectural change, open an issue to discuss the design first.
* **No Automated Spam:** Automated issue-claiming bots, unverified "drive-by" pull requests, and AI-generated review comments on other contributors' pull requests will be closed immediately.
* **Licensing and Data Privacy:** Ensure your AI tool's Terms of Service are compatible with Soft Potato's BSD 3-Clause license, and never paste private credentials or unpublished, embargoed experimental data into external AI prompts.
* **Use Repository Context:** Configure your AI environment to load Soft Potato's repository context and skill instructions (such as `CONTEXT.md`, `AGENTS.md`, `SKILL.md`, and `pyproject.toml`) so generated code adheres to our architecture and conventions out of the box.

---

## Development Setup

Follow these steps to set up a local development environment on Linux, macOS, or WSL:

* Fork the repository on GitHub (`https://github.com/oliverrdz/softpotato`) and clone your fork locally:

  ```bash
  git clone git@github.com:<your-username>/softpotato.git
  cd softpotato
  ```

* Create and activate an isolated Python virtual environment:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

* Install Soft Potato in editable mode with development dependencies:

  ```bash
  pip install --upgrade pip
  pip install -e ".[dev]"
  ```

* Create a dedicated feature or bugfix branch from `main` before making changes:

  ```bash
  git checkout -b feat/your-feature-name
  ```

---

## Architectural Principles and Coding Standards

Soft Potato v3+ is designed around clarity, modularity, and educational transparency for experimental and computational electrochemists:

* **Decoupled Architecture:** Keep physical chemistry, electrode kinetics, and mass transport formulations strictly decoupled from numerical discretization grids and PDE/ODE solvers.
* **Code Style and Linting:** All Python code must comply with `ruff` formatting and linting rules configured in `pyproject.toml`. Run `ruff check .` and `ruff format --check .` before committing.
* **Type Validation and Docstrings:** Use clear type annotations and NumPy-style docstrings explaining physical parameters, units, and return values.
* **Scientific Plotting:** Any figures added to tutorials, examples, or documentation must use color-blind friendly palettes, distinct line styles or markers, and quantity calculus axis labels formatted as `variable / unit` (for example, `E / V`, `I / A`, `t / s`).

---

## Testing and Quality Checks

Every bug fix or new feature must include comprehensive unit tests written with `pytest`:

* **Analytical Benchmarks:** Whenever adding or modifying numerical solvers or electrochemical mechanisms (such as `CV` or `CA`), include tests verifying numerical convergence against closed-form analytical solutions in `softpotato.analytical` (for example, the Randles-Ševčík or Cottrell equations) within documented error tolerances.
* **Run the Test Suite:** Execute `pytest` locally and confirm all tests pass:

  ```bash
  pytest
  ```

* **Run Linter and Formatter:** Ensure zero lint warnings or formatting diffs remain:

  ```bash
  ruff check .
  ruff format .
  ```

---

## Documentation and Tutorials

Documentation and educational guides live in the repository and power `https://softpotato.xyz`:

* Keep explanations accessible to experimental electrochemists who may be new to numerical simulation.
* Include runnable, self-contained example scripts or notebooks when introducing new public APIs.
* Update `CHANGELOG.md` under the `[Unreleased]` section following [Keep a Changelog](https://keepachangelog.com/) conventions for any user-facing addition, change, or fix.

---

## Submitting a Pull Request

* Open an issue first if you are proposing a major new feature, breaking API change, or architectural refactor.
* Commit your changes with clear, descriptive commit messages. Include an `Assisted-by:` trailer if generative AI materially contributed to the commit.
* Push your branch to your fork and open a Pull Request against `main` on `oliverrdz/softpotato`.
* Fill out the Pull Request template completely, summarizing the physical or technical motivation, how the changes were tested, and any AI tools used.