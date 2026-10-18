# File Management & Data Inspection

Managing experimental datasets, copying simulation outputs, and quickly inspecting CSV files without loading heavy text editors are core skills for any computational researcher.

In this tutorial, you will learn:
1. Creating, copying, moving, and removing files safely.
2. Fast data inspection using `cat`, `head`, `tail`, and `less`.
3. Searching across code and datasets using `grep`.
4. Inspecting file sizes and disk usage with `ls -lh` and `du -sh`.

---

## 1. Creating, Copying, and Moving Files

```bash
# Create an empty file or update its timestamp
touch simulation_config.yaml

# Copy a single file
cp cottrell_run_01.csv cottrell_backup.csv

# Copy an entire folder recursively (-r)
cp -r results/ results_archive/

# Move or rename a file
mv cottrell_backup.csv archive/cottrell_run_01_backup.csv

# Safely remove a file
rm old_temp_data.csv

# Remove an empty folder
rmdir empty_folder/

# Remove a non-empty directory and its contents (-r: recursive, -f: force)
rm -rf old_experiment/
```

!!! danger "Caution with `rm -rf`"
    In Linux, there is no "Recycle Bin" or "Trash" for terminal `rm` commands! When you execute `rm -rf`, files are permanently deleted. Always double-check your path before pressing Enter.

---

## 2. Inspecting Data Files in the Terminal

Opening a 200 MB CSV file in a graphical text editor can freeze your computer. The command line provides instant streaming tools:

### Peeking at the First Lines with `head`
View the header and top data rows:

```bash
# Display the first 10 lines of a data file
head -n 10 cv_experimental_data.csv
```

### Checking the Last Lines with `tail`
Useful for checking if a long simulation has finished or errored:

```bash
# Display the last 15 lines of a simulation log
tail -n 15 simulation.log

# Follow a live log file in real-time as new lines are appended (-f)
tail -f simulation.log
```

### Scrolling Through Files with `less`
Use `less` to scroll up and down through large files without loading the whole file into memory:

```bash
less big_mesh_output.txt
```
- Press <kbd>Space</kbd> to scroll down a page.
- Press <kbd>b</kbd> to scroll up a page.
- Press <kbd>/</kbd> followed by a query to search text.
- Press <kbd>q</kbd> to exit.

---

## 3. Searching Files with `grep`

`grep` searches for words or patterns inside files:

```bash
# Search for the word 'diffusion' in all Python files in the current folder
grep "diffusion" *.py

# Recursively search all files (-r) with line numbers (-n)
grep -rn "sand_transition_time" src/

# Case-insensitive search (-i)
grep -i "cottrell" changelog.md
```

---

## 4. Checking File Sizes & Disk Space

```bash
# Detailed list with human-readable file sizes (-lh)
ls -lh

# Measure the total size of a directory (-s: summary, -h: human-readable)
du -sh results/
```

!!! chip "Chip's Tip: Use Pipes to Combine Commands"
    You can pipe the output of one command into another using `|`. For example, to find all occurrences of `"error"` in a log and only see the first 5:
    ```bash
    grep "error" simulation.log | head -n 5
    ```
