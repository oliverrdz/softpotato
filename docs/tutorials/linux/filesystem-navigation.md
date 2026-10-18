# Navigating the Linux Filesystem

Working smoothly in the Linux terminal begins with understanding how the filesystem is structured and how to navigate between directories.

In this tutorial, you will learn:
1. The Linux directory tree hierarchy.
2. Essential commands: `pwd`, `ls`, `cd`, and `mkdir`.
3. Absolute paths versus relative paths.
4. Productive navigation shortcuts and Tab autocompletion.

---

## 1. The Linux Filesystem Tree

Unlike Windows with drive letters (`C:\`, `D:\`), Linux organizes all disks, folders, and devices under a single root directory denoted by `/`.

Key directories in the hierarchy include:
- `/` — The root directory.
- `/home/<username>` or `~` — Your personal home directory where your code, virtual environments, and data live.
- `/tmp` — Temporary files, cleared on system reboot.
- `/etc` — System-wide configuration files.

When you open a terminal, you start inside a **current working directory** (typically your home directory `~`).

---

## 2. Essential Navigation Commands

| Command | Description | Practical Example |
| :--- | :--- | :--- |
| `pwd` | Print Working Directory (shows your current absolute location) | `pwd` $\rightarrow$ `/home/oliver/projects/softpotato` |
| `ls` | List contents of the current directory | `ls -la` (lists hidden files with details) |
| `cd <path>` | Change directory | `cd docs/tutorials` |
| `cd ..` | Move up one directory level | `cd ..` |
| `cd ~` or `cd` | Return immediately to your home directory | `cd` |
| `mkdir <dir>` | Make a new directory | `mkdir -p simulations/cv_scans` |

---

## 3. Absolute vs. Relative Paths

Understanding paths prevents file not found errors:

- **Absolute path**: Starts from the root `/` or home `~`. It always points to the exact same location regardless of where you are:
  ```bash
  cd /home/oliver/gitHub/softpotato/docs
  ```
- **Relative path**: Starts from your current directory. It depends on where you are:
  ```bash
  # Move into a subfolder of current directory
  cd docs/tutorials

  # Move up two folders, then into another
  cd ../../tests
  ```

---

## 4. Creating Nested Directories with `mkdir -p`

When setting up simulation project folders, you often need nested subdirectories. Use the `-p` (parents) flag so you don't have to create each level one by one:

```bash
# Creates the entire folder hierarchy at once
mkdir -p project/data/raw_cv project/results/figures project/scripts
```

---

## 5. Navigation Shortcuts

```bash
# Toggle back to your previous directory
cd -

# Go to the parent directory
cd ..

# Go back two levels
cd ../..
```

!!! chip "Chip's Tip: Master the Tab Key!"
    Never type out long directory or file names by hand! Press the <kbd>Tab</kbd> key after typing the first 2 or 3 letters of any path. Linux will automatically autocomplete the name. If multiple files match, pressing <kbd>Tab</kbd> twice lists all possibilities.
