# Soft Potato Project Management & Workflow Guidelines

This document outlines the standard operating procedures for managing tasks, tracking changes, maintaining the changelog, and handling version releases for **Soft Potato**.

---

## 1. Commit Messages (Conventional Commits)

We follow the **Conventional Commits** specification to maintain a clean, readable history and automate release notes. Every commit message must follow this structure:

`type(scope): short summary description`

### Allowed Types
* `feat`: A new feature or module implementation (e.g., `feat(mesh): add Uniform1DMesh class`).
* `fix`: A bug fix (e.g., `fix(discretizer): resolve stencil index off-by-one error`).
* `test`: Adding or updating unit tests (e.g., `test(cv): add Randles-Sevcik peak current validation`).
* `docs`: Documentation or memory updates (e.g., `docs: update GEMINI.txt task checklist`).
* `refactor`: Code changes that neither fix a bug nor add a feature.
* `chore`: Maintenance tasks, dependency updates, or CI configuration.

---

## 2. Task Tracking & Completion

We track micro-level progress directly within the codebase using the AI project memory file and git branches.

1. **Active Task Lists (`GEMINI.txt`):**
   - When starting a subtask from the checklist in `GEMINI.txt`, ensure it is marked as active.
   - When a task is fully implemented and tested via `pytest`, update the line from `[ ]` to `[x]`.

2. **Branching Workflow:**
   - Keep `main` stable and strictly reserved for tested, working code.
   - For new features or bug fixes, create short-lived feature branches (e.g., `feature/fdm-discretizer` or `fix/nernst-bc`).
   - Merge back to `main` via pull requests or clean merges once tests pass.

---

## 3. Changelog Maintenance (`CHANGELOG.md`)

We adhere to the [Keep a Changelog](https://keepachangelog.com/) standard. 

* **The `[Unreleased]` Section:** 
  Whenever you complete a feature, bug fix, or test suite, immediately add a bullet point under the `### Added`, `### Changed`, or `### Fixed` section of `[Unreleased]` in `CHANGELOG.md`.
* **Categories to Use:**
  * `Added` for new features.
  * `Changed` for changes in existing functionality.
  * `Deprecated` for soon-to-be removed features.
  * `Removed` for removed features.
  * `Fixed` for bug fixes.
  * `Security` for vulnerability patches.

---

## 4. Releasing & Versioning (Semantic Versioning)

Soft Potato uses **Semantic Versioning (`MAJOR.MINOR.PATCH`)**:
* **MAJOR (`1.0.0`)**: Incompatible API breaking changes.
* **MINOR (`0.1.0`)**: New backwards-compatible features or functional MVP phases.
* **PATCH (`0.1.1`)**: Backwards-compatible bug fixes.

### Release Checklist
When preparing a formal release (such as moving from MVP to a public version):
1. **Update `pyproject.toml`:** Bump the version string (e.g., `version = "0.1.0"`).
2. **Update `src/softpotato/__init__.py`:** Update `__version__ = "0.1.0"`.
3. **Finalize `CHANGELOG.md`:** 
   - Rename the `[Unreleased]` header to the new version and release date (e.g., `## [0.1.0] - YYYY-MM-DD`).
   - Create a fresh, empty `[Unreleased]` section at the top.
   - Update the comparison link URLs at the bottom of the changelog.
4. **Tag the Commit in Git:**
   Run the following commands in order:
   - `git add pyproject.toml src/softpotato/__init__.py CHANGELOG.md GEMINI.txt`
   - `git commit -m "chore(release): bump version to 0.1.0"`
   - `git tag -a v0.1.0 -m "Soft Potato Version 0.1.0 Release"`
   - `git push origin main --tags`
5. **Publish on GitHub:** Go to GitHub Releases, draft a new release from the `v0.1.0` tag, and paste the changelog notes for that version.
