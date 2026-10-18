# Setting Up a Python Virtual Environment

Before writing code or running simulations, establishing an isolated **virtual environment** is the single most important best practice in scientific computing.

In this tutorial, you will learn:
1. Why virtual environments are essential for scientific computing.
2. How to create and activate a virtual environment with `venv`.
3. How to install packages using `pip` and the ultra-fast `uv`.
4. How to manage project dependencies and avoid environment drift.

---

## 1. Why Use Virtual Environments?

When you install packages directly to your operating system's global Python:
- Different projects may require incompatible versions of dependencies (e.g., `scipy 1.11` vs `scipy 1.14`).
- System package managers (like `apt` or Homebrew) can conflict with `pip`.
- Linux distributions (such as Ubuntu 24.04+ or Debian 12+) block direct `pip install` commands with `externally-managed-environment` errors to protect the operating system.

A **virtual environment** is an isolated self-contained directory containing its own Python interpreter and `site-packages` directory. Anything you install inside it never interferes with your system or other projects.

---

## 2. Creating a Virtual Environment

Open your terminal, navigate to your project directory, and use Python's built-in `venv` module:

```bash
# Navigate to your project folder
cd ~/projects/my-electrochem-project

# Create a virtual environment named .venv
python3 -m venv .venv
```

The name `.venv` (with a leading dot) is standard practice. It keeps the directory hidden in file explorers.

---

## 3. Activating the Environment

Before installing any packages or executing scripts, you must activate the environment:

=== "Linux / macOS (Bash / Zsh)"
    ```bash
    source .venv/bin/activate
    ```

=== "Windows (Command Prompt)"
    ```cmd
    .venv\Scripts\activate.bat
    ```

=== "Windows (PowerShell)"
    ```powershell
    .venv\Scripts\Activate.ps1
    ```

Once activated, your terminal prompt will display the environment name in parentheses, for example:

```text
(.venv) user@machine:~/projects/my-electrochem-project$
```

To deactivate when you are finished working:
```bash
deactivate
```

---

## 4. Installing Scientific Packages

With your virtual environment activated, install the standard scientific stack along with **Soft Potato**:

```bash
# Upgrade pip to the latest version
pip install --upgrade pip

# Install scientific libraries and Soft Potato
pip install numpy scipy matplotlib softpotato
```

### Blazing Fast Setup with `uv`
If you frequently create virtual environments, consider using [`uv`](https://github.com/astral-sh/uv), an extremely fast Python package manager written in Rust:

```bash
# Create a venv in 10 milliseconds
uv venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies 10x-100x faster than pip
uv pip install numpy scipy matplotlib softpotato
```

---

## 5. Verifying Your Installation

You can quickly verify that your environment has access to Python and Soft Potato:

```bash
python -c "import softpotato; print('Soft Potato version:', softpotato.__version__)"
```

Output:
```text
Soft Potato version: 3.1.0
```

---

## 6. Saving & Sharing Dependencies

To reproduce your exact simulation environment on another computer or lab workstation, generate a `requirements.txt` lockfile:

```bash
# Export all installed package versions
pip freeze > requirements.txt
```

On a new machine, recreate the identical environment with a single command:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

!!! chip "Chip's Tip: Always Gitignore Your `.venv`"
    Never commit the `.venv` directory to Git! Virtual environments contain compiled platform-specific binaries that should not be tracked in version control. Always add `.venv/` to your project's `.gitignore` file, and share your `requirements.txt` or `pyproject.toml` instead.
