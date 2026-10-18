# Linux Command Line Basics

Working comfortably in the Linux terminal makes running simulations, managing Python environments, and automating data processing much faster.

---

## 1. Navigating the Filesystem

When you open a terminal, you are inside a **working directory**. These essential commands let you inspect and move around the directory tree:

| Command | Description | Example |
| :--- | :--- | :--- |
| `pwd` | Print the current working directory | `pwd` $\rightarrow$ `/home/user/projects` |
| `ls` | List files and directories (`-la` shows hidden files and details) | `ls -la` |
| `cd` | Change directory (`..` moves up one level, `~` goes to home) | `cd ~/projects/softpotato` |
| `mkdir` | Create a new directory (`-p` creates parent directories if needed) | `mkdir -p data/cv_scans` |

!!! chip "Chip's Tip: Use Tab Autocompletion!"
    Press the <kbd>Tab</kbd> key while typing any file or folder name in the Linux terminal to autocomplete it automatically. This saves typing and prevents typos in long file paths!

---

## 2. Viewing and Managing Files

```bash
# Copy a file
cp experiment_01.csv experiment_01_backup.csv

# Move or rename a file
mv experiment_01_backup.csv archive/

# View the first or last 10 lines of a data file
head -n 10 experiment_01.csv
tail -n 10 experiment_01.csv

# Search for text inside files using grep
grep -rn "diffusion" src/
```

---

## 3. Running Python Scripts & Redirecting Output

You can run any `.py` script directly from the terminal and save its output to a log file using `>`:

```bash
# Activate your virtual environment first
source .venv/bin/activate

# Run a simulation script and save output to simulation.log
python run_simulation.py > simulation.log 2>&1
```

---

## 4. Connecting to Remote Servers with SSH

If you run computations on a workstation, lab server, or VPS, use `ssh` to connect and `rsync` to transfer data:

```bash
# Connect to a remote server
ssh user@softpotato.xyz

# Sync a local results folder to a remote server
rsync -avz ./results/ user@softpotato.xyz:~/results/
```
