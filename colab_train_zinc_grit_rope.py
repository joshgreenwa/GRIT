"""
Colab driver: train + eval GritRoPETransformer on ZINC (subset, 12k).

Assumptions
-----------
- You have uploaded the working folder (the one that contains `grit/` with
  `main.py`, `configs/`, `grit/` package, etc.) to your Google Drive.
- You will run this script cell-by-cell (or top-to-bottom) in a Colab notebook.
- A GPU runtime is selected (Runtime -> Change runtime type -> GPU).

What it does
------------
1. Mounts Google Drive and cd's into the working folder.
2. Installs the pinned dependencies GRIT needs.
3. Writes a new config YAML that is byte-for-byte the official
   `zinc-GRIT-RRWP.yaml` except `gt.layer_type` is swapped to
   `GritRoPETransformer` (and optionally the number of epochs is reduced,
   because 2000 epochs on a free Colab T4 takes ~half a day).
4. Launches `main.py` with that config as a subprocess and streams its
   stdout live so you can watch per-epoch train/val/test MAE.
5. At the end, parses the final results file and prints a tidy summary.

Experimental setup (from `configs/GRIT/zinc-GRIT-RRWP.yaml`, matches Table 2 of
the GRIT paper for ZINC 12k subset):

    - Dataset:        ZINC subset (12k molecules)
    - Task:           graph-level regression, MAE loss
    - Model:          GRIT architecture with our structure-rotated attention
    - Layers:         10
    - Heads:          8
    - Hidden dim:     64  (gnn.dim_inner == gt.dim_hidden)
    - Dropout:        0.0
    - Attn dropout:   0.2
    - RRWP ksteps:    21
    - Batch size:     32
    - Optimiser:      AdamW, weight_decay=1e-5
    - LR:             1e-3 with 50-epoch warmup + cosine decay to 1e-6
    - Epochs:         2000 (paper); overridable below
    - Grad clip:      1.0
    - Pooling:        sum
    - Metric:         MAE (lower is better)
"""

# ============================================================================
# CELL 1 -- User-configurable settings
# ============================================================================

# Absolute path inside your Google Drive to the folder that contains `main.py`
# and the `grit/` Python package. Example:
#     My Drive/GRIT/grit/    <- this folder has main.py, configs/, grit/, ...
# then set:
DRIVE_WORK_DIR = "/content/drive/MyDrive/GRIT/grit"

# Where ZINC will be downloaded to (first run). Keep on Drive so reruns skip
# the download, or use /content for faster ephemeral storage.
DATA_DIR = "/content/drive/MyDrive/GRIT/datasets"

# Output directory for logs, checkpoints, final metrics.
OUT_DIR = "/content/drive/MyDrive/GRIT/results_rope_zinc"

# Random seed -- the paper reports mean over seeds 0..3. Pick one for Colab.
SEED = 0

# Max epochs. Paper uses 2000; set lower while debugging (e.g. 200) because
# Colab free tier will almost certainly disconnect before 2000 finishes on T4.
# 400 epochs is usually enough to see clear signal on ZINC-12k.
MAX_EPOCHS = 2000

# If you want a quick smoke test, set this to True to run 5 epochs.
SMOKE_TEST = False

# Override d_struct channel split (see grit_rope_layer.py). None => 3/4 of
# head dim, rounded down to even (default).
D_STRUCT = None  # e.g. 6 for head_dim=8


# ============================================================================
# CELL 2 -- Mount Google Drive
# ============================================================================

def mount_drive():
    from google.colab import drive  # only available inside Colab
    drive.mount("/content/drive")
    print("[mount] Google Drive mounted at /content/drive")


# ============================================================================
# CELL 3 -- Install dependencies
# ============================================================================
# Colab currently ships with a recent torch (e.g. 2.x). GRIT's README pins
# torch 1.12.1 + cu113, but the code itself works with newer torch as long
# as PyG extensions are built for that torch version. We install PyG + scatter
# + sparse wheels matched to whatever torch Colab is currently on.
#
# NOTE: after installing setuptools==59.5.0 you MAY need to restart the
# runtime once. If you hit a `distutils` error on first run, do
# Runtime -> Restart runtime, then re-run everything from CELL 4 onwards.

INSTALL_COMMANDS = r"""
set -e
python -c "import torch; print('[install] torch:', torch.__version__, 'cuda:', torch.version.cuda)"

TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VERSION=$(python -c "import torch; v=torch.version.cuda; print('cu'+v.replace('.','')) if v else print('cpu')")
echo "[install] Resolved torch=$TORCH_VERSION cuda=$CUDA_VERSION"

pip install -q torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install -q torch-geometric==2.2.0
pip install -q torchmetrics==0.9.1
pip install -q ogb
pip install -q yacs
pip install -q opt_einsum
pip install -q tensorboardX
pip install -q performer-pytorch
pip install -q rdkit
pip install -q pytorch-lightning
pip install -q setuptools==59.5.0

echo "[install] Done."
"""


def install_deps():
    import subprocess
    print("[install] Installing GRIT dependencies (this takes ~2 min) ...")
    proc = subprocess.run(
        ["bash", "-lc", INSTALL_COMMANDS],
        capture_output=True, text=True,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        print("[install] STDERR:\n" + proc.stderr)
        raise RuntimeError("Dependency installation failed. See stderr above.")
    print("[install] Finished installing dependencies.")


# ============================================================================
# CELL 4 -- Enter the working directory and sanity-check the repo layout
# ============================================================================

def enter_workdir():
    import os
    import sys
    if not os.path.isdir(DRIVE_WORK_DIR):
        raise FileNotFoundError(
            f"DRIVE_WORK_DIR={DRIVE_WORK_DIR!r} does not exist. "
            "Edit CELL 1 to point at the folder in Drive that contains main.py."
        )
    os.chdir(DRIVE_WORK_DIR)
    if DRIVE_WORK_DIR not in sys.path:
        sys.path.insert(0, DRIVE_WORK_DIR)
    print(f"[workdir] cwd = {os.getcwd()}")

    # Sanity: must contain main.py, grit/ package, configs/GRIT/
    needed = ["main.py", "grit", "configs/GRIT/zinc-GRIT-RRWP.yaml",
              "grit/layer/grit_rope_layer.py"]
    missing = [p for p in needed if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "The following required paths are missing in DRIVE_WORK_DIR:\n  "
            + "\n  ".join(missing)
            + "\nMake sure you uploaded the full GRIT working folder, including "
              "the new grit_rope_layer.py we added."
        )
    print("[workdir] Repo layout looks correct.")


# ============================================================================
# CELL 5 -- Build the GritRoPE config (derived from zinc-GRIT-RRWP.yaml)
# ============================================================================

def build_config(cfg_out_path: str):
    """Write a new YAML cfg that swaps layer_type -> GritRoPETransformer."""
    import os

    base_path = "configs/GRIT/zinc-GRIT-RRWP.yaml"
    with open(base_path, "r") as f:
        base_yaml = f.read()

    # We deliberately do line-level string edits rather than loading as a dict
    # and dumping, so that comments / formatting of the original file are
    # preserved byte-for-byte for every field we don't touch.
    out_lines = []
    in_attn_block = False
    attn_block_indent = None
    d_struct_injected = False

    for line in base_yaml.splitlines():
        stripped = line.rstrip()
        if stripped.startswith("  layer_type: GritTransformer"):
            out_lines.append("  layer_type: GritRoPETransformer")
            continue
        # Track the attn: block so we can inject d_struct inside it.
        if stripped.startswith("  attn:"):
            in_attn_block = True
            attn_block_indent = "    "
            out_lines.append(line)
            continue
        if in_attn_block and line and not line.startswith("    ") and not line.startswith("\t"):
            # Exited the attn: block without injecting d_struct yet.
            if D_STRUCT is not None and not d_struct_injected:
                out_lines.append(f"    d_struct: {int(D_STRUCT)}")
                d_struct_injected = True
            in_attn_block = False
        out_lines.append(line)

    # If attn: was the last block and we never injected, append now.
    if in_attn_block and D_STRUCT is not None and not d_struct_injected:
        out_lines.append(f"    d_struct: {int(D_STRUCT)}")

    # Override a few top-level fields by appending (YACS lets later keys
    # override earlier ones, but YAML doesn't -- so instead we rewrite in place).
    content = "\n".join(out_lines) + "\n"

    def replace_scalar(text, key_path, new_value):
        # key_path like "optim.max_epoch" -> find "  max_epoch: <old>" under optim:
        # Simple line replacement keyed on the leaf.
        leaf = key_path.split(".")[-1]
        lines = text.splitlines()
        for i, ln in enumerate(lines):
            if ln.strip().startswith(f"{leaf}:"):
                indent = ln[: len(ln) - len(ln.lstrip())]
                lines[i] = f"{indent}{leaf}: {new_value}"
                return "\n".join(lines) + "\n"
        raise KeyError(f"Could not find key {key_path} in config")

    epochs = 5 if SMOKE_TEST else MAX_EPOCHS
    content = replace_scalar(content, "optim.max_epoch", epochs)
    # Write out_dir into the yaml so main.py respects it.
    content = replace_scalar(content, "out_dir", OUT_DIR)

    os.makedirs(os.path.dirname(cfg_out_path), exist_ok=True)
    with open(cfg_out_path, "w") as f:
        f.write(content)

    print(f"[config] Wrote derived config to {cfg_out_path}")
    print(f"[config] layer_type=GritRoPETransformer, max_epoch={epochs}, "
          f"seed={SEED}, d_struct={D_STRUCT}")
    print("[config] --- begin config ---")
    print(content)
    print("[config] --- end config ---")


# ============================================================================
# CELL 6 -- Run main.py with live-streamed output
# ============================================================================

def run_training(cfg_path: str):
    import os
    import subprocess
    import sys
    import time

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    cmd = [
        sys.executable, "-u", "main.py",
        "--cfg", cfg_path,
        "seed", str(SEED),
        "dataset.dir", DATA_DIR,
        "wandb.use", "False",
        "accelerator", "cuda:0",
    ]
    print("[run] Launching:", " ".join(cmd))
    print("[run] Live training log follows. Watch for lines like:")
    print("      'train: ... mae: 0.xxx' and 'val: ... mae: 0.xxx'")
    print("=" * 78)

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=DRIVE_WORK_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n[run] KeyboardInterrupt -- terminating training subprocess ...")
        proc.terminate()
        proc.wait()
        raise
    proc.wait()
    dt = time.time() - t0
    print("=" * 78)
    print(f"[run] Training subprocess exited with code {proc.returncode} "
          f"after {dt/60:.1f} min.")
    if proc.returncode != 0:
        raise RuntimeError("Training failed; inspect the log above.")


# ============================================================================
# CELL 7 -- Parse final results
# ============================================================================

def summarise_results():
    """Walk OUT_DIR for the 'best' metrics written by GRIT's logger."""
    import json
    import os
    import glob

    print(f"[results] Scanning {OUT_DIR} for result files ...")
    # GRIT writes to out_dir/<cfg_name>/<seed>/{train,val,test}/stats.json
    candidates = sorted(glob.glob(os.path.join(OUT_DIR, "**", "stats.json"),
                                  recursive=True))
    if not candidates:
        print("[results] No stats.json found. You may need to scroll the log "
              "above and find the 'best' line printed by the custom trainer.")
        return

    # Show the tail of each split's stats.json.
    for path in candidates:
        rel = os.path.relpath(path, OUT_DIR)
        print(f"\n[results] --- {rel} (last entry) ---")
        try:
            with open(path) as f:
                lines = [ln for ln in f.readlines() if ln.strip()]
            if lines:
                last = json.loads(lines[-1])
                for k in ("epoch", "loss", "mae", "lr"):
                    if k in last:
                        print(f"    {k}: {last[k]}")
        except Exception as e:  # pragma: no cover
            print(f"    (failed to parse: {e})")


# ============================================================================
# CELL 8 -- Orchestrator
# ============================================================================

def main():
    import os
    mount_drive()
    enter_workdir()
    install_deps()

    cfg_path = os.path.join(
        DRIVE_WORK_DIR, "configs", "GRIT", "zinc-GRIT-RRWP-rope.yaml"
    )
    build_config(cfg_path)
    run_training(cfg_path)
    summarise_results()
    print("\n[done] All finished. Check OUT_DIR for checkpoints and full logs:")
    print(f"       {OUT_DIR}")


if __name__ == "__main__":
    # In Colab you can either (a) paste each CELL block into its own cell and
    # run them in order, or (b) put this whole file on Drive and just do
    #     !python /content/drive/MyDrive/GRIT/grit/colab_train_zinc_grit_rope.py
    # from a single Colab cell after you've already mounted Drive.
    main()
