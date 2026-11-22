# Google Colab Quickstart

<div class="sp-section-banner" markdown>
<div markdown>
**Zero-install workflow.** Run Soft Potato directly in your web browser using Google Colab—no Python installation, terminal commands, or local setup required.
</div>
<img src="../../assets/images/chip-lab.png" alt="Chip in the Laboratory" />
</div>

## 1. Launch the Starter Notebook

Click the button below to open the official **Soft Potato v3 Starter** notebook in Google Colab:

[:octicons-play-24: Open Soft Potato v3 Starter in Google Colab](https://drive.google.com/drive/folders/1aHJVZaPPRcIK2uy6OgobL8FOKzcMjjLx?usp=drive_link){ .md-button .md-button--primary target="_blank" rel="noopener" }

---

## 2. How to Run a Notebook in Colab

1. **Sign in** with any Google account (required by Google Colab to allocate a free cloud runtime).
2. Click **Runtime → Run all** in the top menu bar (or press ++ctrl+f9++ / ++cmd+f9++).
3. If Colab shows a *"This notebook was not authored by Google"* prompt, click **Run anyway**.
4. Wait ~10 seconds for the first cell to install `softpotato` from PyPI, then scroll down to view your plots.

!!! chip "Chip's Tip"
    To save your own copy of a notebook so you can edit parameters and keep your results, click **File → Save a copy in Drive**.

---

## 3. Starting from a Blank Colab Notebook

If you create a new notebook at [colab.research.google.com](https://colab.research.google.com/), add this line to your very first code cell to install Soft Potato:

```python
!pip install -q softpotato
```

Then paste any of the [5-Minute Copy-Paste Recipes](recipes/first-simulation.md) into the next cell and press **Shift + Enter**.

---

## 4. Beginner "Run All" Notebooks

Prefer a self-contained walkthrough for a specific technique? Open either starter notebook below:

- [Chronoamperometry (CA) Quickstart](../tutorials/start/ca_quickstart.ipynb) — simulate a potential step, compare with the Cottrell equation, and fit $D$.
- [Cyclic Voltammetry (CV) Quickstart](../tutorials/start/cv_quickstart.ipynb) — simulate a CV, compare with Randles–Ševčík theory, and vary $k^0$.
