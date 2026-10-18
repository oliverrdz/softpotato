# Running Python Scripts & Redirection

Executing Python simulation scripts from the command line allows you to automate runs, log output, and keep long simulations running even after closing your terminal.

In this tutorial, you will learn:
1. Executing Python scripts from the terminal.
2. Understanding the standard input, output, and error streams (`stdin`, `stdout`, `stderr`).
3. Redirecting simulation logs to files (`>`, `>>`, `2>&1`).
4. Running long-running tasks in the background with `nohup` and `&`.
5. Monitoring and terminating processes (`ps`, `top`, `kill`).

---

## 1. Running Scripts from the Terminal

Ensure your virtual environment is activated, then invoke `python`:

```bash
# Activate your project's virtual environment
source .venv/bin/activate

# Execute the simulation script
python run_cv_simulation.py
```

### Making a Script Executable
You can run a script directly without explicitly typing `python` by adding a **shebang** on the first line of your `.py` file:

```python
#!/usr/bin/env python3
print("Running automated electrochemistry batch...")
```

Then grant execute permissions using `chmod`:

```bash
chmod +x run_cv_simulation.py
./run_cv_simulation.py
```

---

## 2. Standard Streams & Redirection

Every Linux process automatically connects to three standard I/O streams:
- `stdin` (File descriptor 0) — Standard input.
- `stdout` (File descriptor 1) — Normal output (e.g. `print()`).
- `stderr` (File descriptor 2) — Error and warning messages.

### Overwriting with `>`
Redirect standard output to a file (creates the file or overwrites existing content):

```bash
python run_simulation.py > output.log
```

### Appending with `>>`
Append output to an existing file without deleting its previous contents:

```bash
python run_simulation.py >> history.log
```

### Capturing Both Output & Errors (`2>&1`)
To ensure that error tracebacks and warning logs are captured along with standard prints:

```bash
python run_simulation.py > simulation.log 2>&1
```
Here, `2>&1` redirects stream 2 (`stderr`) into stream 1 (`stdout`), writing everything into `simulation.log`.

---

## 3. Running Tasks in the Background

Simulations that take minutes or hours to finish shouldn't freeze your terminal prompt.

### Backgrounding with `&`
Append an ampersand `&` to run the command in the background immediately:

```bash
python run_heavy_pde.py > pde.log 2>&1 &
```
The terminal prints a job number and Process ID (PID), e.g. `[1] 34821`, and immediately returns your shell prompt.

### Detaching with `nohup`
If you are logged into a remote machine over SSH and want the script to continue executing even if your Wi-Fi disconnects, prefix the command with `nohup` (no hang up):

```bash
nohup python run_heavy_pde.py > pde.log 2>&1 &
```

---

## 4. Monitoring & Killing Processes

```bash
# Find your running Python simulation processes
ps aux | grep python

# Launch an interactive task monitor
top
# or if installed:
htop

# Terminate a stuck process by its PID
kill 34821

# Force kill a process that refuses to close
kill -9 34821
```

!!! chip "Chip's Tip: Live Monitor with `tail -f`"
    When you start a background simulation with `python run.py > run.log 2>&1 &`, follow its progress in real-time in another terminal tab with:
    ```bash
    tail -f run.log
    ```
    Press <kbd>Ctrl</kbd> + <kbd>C</kbd> at any time to stop following the log without interrupting the simulation itself!
