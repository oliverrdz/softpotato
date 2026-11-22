# Installing Python & Soft Potato

If you want to run Soft Potato locally on your own computer—offline, in VS Code, or in JupyterLab—follow this minimal setup guide.

---

## 1. Install Python (3.10 or Newer)

=== "Windows"

    1. Download the latest Python installer from [python.org/downloads](https://www.python.org/downloads/).
    2. **Important:** Check **"Add python.exe to PATH"** at the bottom of the first installer screen, then click **Install Now**.
    3. Open **Command Prompt** or **PowerShell** and verify:

        ```powershell
        python --version
        ```

=== "macOS"

    1. Download the macOS installer from [python.org/downloads](https://www.python.org/downloads/) (or install via [Homebrew](https://brew.sh/): `brew install python`).
    2. Open **Terminal** and verify:

        ```bash
        python3 --version
        ```

=== "Linux"

    Python 3 is pre-installed on most distributions. Open your terminal and verify you have Python 3.10 or newer:

    ```bash
    python3 --version
    ```

    If `venv` or `pip` is missing on Ubuntu/Debian, install them with:

    ```bash
    sudo apt update && sudo apt install -y python3-pip python3-venv
    ```

---

## 2. Create a Virtual Environment & Install Soft Potato

Using a virtual environment keeps Soft Potato's dependencies isolated from other Python projects on your machine.

=== "Windows (PowerShell)"

    ```powershell
    python -m venv sp-env
    .\sp-env\Scripts\Activate.ps1
    pip install softpotato
    ```

=== "macOS / Linux"

    ```bash
    python3 -m venv sp-env
    source sp-env/bin/activate
    pip install softpotato
    ```

!!! tip "Optional: Jupyter Notebooks Locally"
    If you also want to run `.ipynb` notebooks locally, install JupyterLab in the same environment:

    ```bash
    pip install softpotato jupyterlab
    ```

---

## 3. Verify Your Installation

Run this one-liner in your activated terminal:

```bash
python -c "import softpotato as sp; print(sp.__version__)"
```

You should see the installed version number (e.g. `3.4.1`).

---

## Next Steps

- [First Simulation (CV)](recipes/first-simulation.md) — simulate and plot a cyclic voltammogram in 5 minutes.
- [Analytical Baselines](recipes/analytical.md) — compute Cottrell, Levich, Randles–Ševčík, and SECM curves.
- [Fitting Experimental Data](recipes/fitting.md) — extract $D$, $RG$, and $d_{\text{offset}}$ with 95% confidence intervals.
