# Remote Computing with SSH & Rsync

Complex multi-dimensional electrochemical models, parameter fits, and large batch scans frequently run on remote lab workstations, university computing clusters, or cloud virtual machines.

In this tutorial, you will learn:
1. Connecting securely to remote computers with `ssh`.
2. Setting up passwordless authentication using SSH key pairs.
3. Synchronizing code and simulation datasets with `rsync`.
4. Using terminal multiplexers (`tmux`) to keep long sessions persistent.

---

## 1. Connecting via SSH

**SSH** (Secure Shell) provides an encrypted terminal connection to a remote machine:

```bash
# Connect to a remote machine with username and hostname or IP
ssh oliver@lab-cluster.university.edu

# If the server uses a non-standard port (e.g. 2222)
ssh -p 2222 oliver@192.168.1.50
```

Once connected, your terminal prompt changes to reflect the remote machine. Any command you type runs on the remote server's CPU!

---

## 2. Passwordless Authentication with SSH Keys

Instead of typing your password every time, generate an SSH cryptographic key pair:

```bash
# 1. Generate an modern Ed25519 key pair on your local computer
ssh-keygen -t ed25519 -C "your_email@example.com"
# Press Enter to accept default location ~/.ssh/id_ed25519

# 2. Copy your public key to the remote server
ssh-copy-id oliver@lab-cluster.university.edu
```

Now you can log in instantly without entering a password, and automated scripts can securely connect.

---

## 3. Transferring Simulation Data with `rsync`

While `scp` copies entire files, `rsync` (remote sync) compares local and remote folders and **only transfers the modified blocks**, saving gigabytes of bandwidth.

```bash
# Push local code and scripts to remote server
# -a: archive mode (preserves permissions, times, symlinks)
# -v: verbose output
# -z: compress data during transfer
# -P: show progress bar and allow resume
rsync -avzP ./my_simulation_code/ oliver@lab-cluster:~/simulations/

# Pull simulation results and figures back to your local computer
rsync -avzP oliver@lab-cluster:~/simulations/results/ ./local_results/
```

!!! note "The Trailing Slash in `rsync`"
    In `rsync`, a trailing slash on the source folder (`./data/`) means "copy the *contents* of this folder". Omitting the trailing slash (`./data`) means "copy the folder itself into the destination".

---

## 4. Persistent Sessions with `tmux`

If your connection drops, an SSH session will terminate and kill any foreground processes. Use `tmux` to keep your environment running on the server indefinitely:

```bash
# 1. Start a new named tmux session on the remote server
tmux new -s electrochem

# 2. Run your simulation inside tmux
python run_heavy_simulation.py

# 3. Detach from tmux and safely disconnect your SSH connection:
# Press Ctrl+B, release, then press D
```

When you log back into the server tomorrow from anywhere:

```bash
# Reattach to your running session
tmux attach -t electrochem
```
Your session, terminals, and outputs will be right where you left them!

!!! chip "Chip's Tip: Use `rsync` with `--exclude`"
    When syncing project directories to remote clusters, exclude large temporary caches and virtual environments:
    ```bash
    rsync -avzP --exclude '.venv' --exclude '__pycache__' ./ oliver@lab-cluster:~/project/
    ```
