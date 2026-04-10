"""
Colab driver: train + eval GRIT-RoPE variants on ZINC (subset, 12k).

Supported model variants (set MODEL_VARIANT below):

    "rope"        GritRoPETransformer
                  RoPE dot-product attention + GRIT-style edge stream update.

    "rope_pair"   GritRoPEPairTransformer
                  RoPE attention + AF2-style symmetric pair-MLP edge update:
                  e_ij^{l+1} = e_ij^l + MLP(LN([h_i+h_j || h_i*h_j || e_ij]))

    "rope_noedge" GritRoPENoEdgeTransformer
                  Pure RoPE attention with NO edge representation at all.
                  All structural info flows through the RoPE rotation angles.

Assumptions
-----------
- You are running this in Google Colab with a GPU runtime
  (Runtime -> Change runtime type -> GPU).
- You have a GitHub personal access token stored as a Colab secret named
  `diss_key` (left sidebar -> key icon -> Add new secret -> name=diss_key,
  value=<your PAT>, and "Notebook access" enabled).
- The repo https://github.com/joshgreenwa/GRIT contains this file and the
  variant layer files under grit/layer/.
- Google Drive is mounted ONLY so that dataset downloads and result
  checkpoints persist across Colab sessions. If you don't want that, set
  USE_DRIVE_FOR_PERSISTENCE = False below and everything will live under
  /content (ephemeral).

What it does
------------
1. (Optionally) mounts Google Drive for dataset + results persistence.
2. Reads the `diss_key` Colab secret and `git clone`s the repo into /content.
3. Installs the pinned dependencies GRIT needs.
4. Writes a new config YAML derived from the official `zinc-GRIT-RRWP.yaml`
   with gt.layer_type and model.type set to the chosen variant.
5. Launches `main.py` with that config as a subprocess and streams its
   stdout live so you can watch per-epoch train/val/test MAE.
6. At the end, parses the final results file and prints a tidy summary.

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

# ---- Model variant selector ----
# Choose one of:
#   "rope"        - RoPE attention + GRIT-style edge stream  (GritRoPETransformer)
#   "rope_pair"   - RoPE attention + AF2 pair-MLP edge update (GritRoPEPairTransformer)
#   "rope_noedge" - Pure RoPE attention, no edges at all      (GritRoPENoEdgeTransformer)
MODEL_VARIANT = "rope"

# ---- Pair-MLP settings (only used when MODEL_VARIANT == "rope_pair") ----
# MLP architecture: 3d -> hidden_mult*d -> d.  Use 1 for no expansion (3d->d->d).
PAIR_MLP_HIDDEN_MULT = 2
# Share one MLP across ALL layers (AF2 style) vs one MLP per layer.
PAIR_MLP_SHARE_ACROSS_LAYERS = False
# MLP dropout (independent of the main attn/layer dropout).
PAIR_MLP_DROPOUT = 0.0
# MLP activation: "gelu" or "relu".
PAIR_MLP_ACT = "gelu"

# ---- Variant name -> internal class name mapping (do not edit) ----
_VARIANT_MAP = {
    "rope":        "GritRoPETransformer",
    "rope_pair":   "GritRoPEPairTransformer",
    "rope_noedge": "GritRoPENoEdgeTransformer",
}

# GitHub repo to clone. Must contain main.py at its root (alongside configs/
# and the inner `grit/` Python package, which in turn contains
# `layer/grit_rope_layer.py`).
GITHUB_REPO = "joshgreenwa/GRIT"
GITHUB_BRANCH = "main"  # change if you pushed to a different branch
SECRET_NAME = "diss_key"  # name of the Colab secret holding your PAT

# Where to clone the repo inside Colab (ephemeral, under /content).
CLONE_DIR = "/content/GRIT"

# Persist dataset + results across Colab sessions by storing them in Drive.
# Set to False for a fully ephemeral run (faster I/O, no Drive mount).
USE_DRIVE_FOR_PERSISTENCE = True

# Where ZINC will be downloaded to (first run).
DATA_DIR = (
    "/content/drive/MyDrive/GRIT/datasets"
    if USE_DRIVE_FOR_PERSISTENCE else "/content/datasets"
)
# Where logs, checkpoints, and final metrics get written.
# Includes the variant name so different runs don't overwrite each other.
OUT_DIR = (
    f"/content/drive/MyDrive/GRIT/results_{MODEL_VARIANT}_zinc"
    if USE_DRIVE_FOR_PERSISTENCE else f"/content/results_{MODEL_VARIANT}_zinc"
)

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

# Filled in by enter_workdir() once we detect where main.py lives inside the
# cloned repo. Everything downstream uses this instead of CLONE_DIR directly.
WORK_DIR = None


# ============================================================================
# CELL 2 -- (Optional) Mount Google Drive for persistence
# ============================================================================

def mount_drive():
    if not USE_DRIVE_FOR_PERSISTENCE:
        print("[mount] USE_DRIVE_FOR_PERSISTENCE=False -- skipping Drive mount.")
        return
    from google.colab import drive  # only available inside Colab
    drive.mount("/content/drive")
    print("[mount] Google Drive mounted at /content/drive")


# ============================================================================
# CELL 2b -- Clone the repo using the Colab secret as a PAT
# ============================================================================

def clone_repo():
    """git clone https://<token>@github.com/<GITHUB_REPO>.git into CLONE_DIR."""
    import os
    import shutil
    import subprocess

    # Pull the PAT from Colab secrets.
    try:
        from google.colab import userdata
        token = userdata.get(SECRET_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read Colab secret {SECRET_NAME!r}. In the left sidebar "
            f"click the key icon, add a secret named {SECRET_NAME!r} with your "
            "GitHub PAT, and enable notebook access."
        ) from exc
    if not token:
        raise RuntimeError(
            f"Colab secret {SECRET_NAME!r} is empty. Paste your GitHub PAT into it."
        )

    # Fresh clone each run so we always pick up the latest commit from the
    # repo. The PAT lives only in the subprocess env, not in the remote URL
    # that gets stored in .git/config.
    if os.path.isdir(CLONE_DIR):
        print(f"[clone] Removing existing {CLONE_DIR} for a fresh clone ...")
        shutil.rmtree(CLONE_DIR)

    remote = f"https://{token}@github.com/{GITHUB_REPO}.git"
    safe_remote = f"https://***@github.com/{GITHUB_REPO}.git"
    print(f"[clone] git clone --depth 1 -b {GITHUB_BRANCH} {safe_remote} {CLONE_DIR}")
    proc = subprocess.run(
        ["git", "clone", "--depth", "1", "-b", GITHUB_BRANCH, remote, CLONE_DIR],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        # Strip the token from any error message before printing.
        msg = (proc.stdout + "\n" + proc.stderr).replace(token, "***")
        raise RuntimeError(f"git clone failed:\n{msg}")

    # Scrub the token from the stored remote URL so `git pull` later prompts
    # rather than silently using the embedded credential.
    subprocess.run(
        ["git", "-C", CLONE_DIR, "remote", "set-url", "origin",
         f"https://github.com/{GITHUB_REPO}.git"],
        check=True,
    )
    print(f"[clone] Clone complete at {CLONE_DIR}")


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

# NOTE on the infamous `pkgutil.ImpImporter` error you may have seen:
# GRIT's README pins setuptools==59.5.0, but that version calls
# pkgutil.ImpImporter, which was REMOVED in Python 3.12 (Colab's current
# Python). Any `pip install` step that pulls in a package whose metadata
# handling routes through pkg_resources from that old setuptools crashes.
# The fix below is threefold:
#   1. Upgrade pip/setuptools/wheel as the FIRST step, so every subsequent
#      pip call uses the modern setuptools that works on Python 3.12.
#   2. Drop packages we don't actually need for ZINC training
#      (performer-pytorch) and remove the hard pin on torchmetrics=0.9.1
#      (a 2022-era package with legacy setup.py metadata that was also
#      tripping the same error).
#   3. Stream pip output live instead of capturing it, so if something DOES
#      fail we see exactly which package and line.

# Each entry is ONE pip invocation so we can label progress.
INSTALL_STEPS = [
    # 1. Modernise the build toolchain before touching anything else.
    ("Upgrade pip / setuptools / wheel",
     "python -m pip install --upgrade pip setuptools wheel"),

    # 2. PyG C++ extensions, built against Colab's current torch/cuda.
    ("torch-scatter + torch-sparse (PyG wheel index)",
     'python - <<PY\n'
     'import torch, subprocess, sys\n'
     'tv = torch.__version__.split("+")[0]\n'
     'cu = "cu" + torch.version.cuda.replace(".", "") if torch.version.cuda else "cpu"\n'
     'url = f"https://data.pyg.org/whl/torch-{tv}+{cu}.html"\n'
     'print("[install] PyG wheel index:", url)\n'
     'subprocess.check_call([sys.executable, "-m", "pip", "install",\n'
     '    "torch-scatter", "torch-sparse", "-f", url])\n'
     'PY'),

    # 3. PyG itself. 2.3.x still has torch_geometric.graphgym (GRIT needs it),
    #    and installs on Python 3.12. PyG 2.5+ removed graphgym.
    ("torch-geometric==2.3.1",
     "python -m pip install torch-geometric==2.3.1"),

    # 4. Remaining pure-Python deps.
    #
    #    CRITICAL: we force-upgrade torchmetrics with `-U` to >=1.0.
    #    Colab preinstalls torchmetrics==0.9.1 (a 2022 release) which does
    #    `from pkg_resources import ...`; that resolves to the Debian system
    #    pkg_resources at /usr/lib/python3/dist-packages/, which hard-codes
    #    `pkgutil.ImpImporter` -- removed in Python 3.12. Modern torchmetrics
    #    (>=1.0) dropped pkg_resources entirely, which bypasses the bug.
    #    Without `-U` pip would say "already satisfied" and keep 0.9.1.
    ("Supporting libs (torchmetrics>=1.0, ogb, yacs, opt_einsum, tensorboardX, rdkit, lightning)",
     "python -m pip install -U 'torchmetrics>=1.0' ogb yacs opt_einsum "
     "tensorboardX rdkit pytorch-lightning"),

    # 5. Sanity-check imports, so any version mismatch dies here with a
    #    readable traceback rather than silently breaking main.py.
    ("Import sanity check",
     'python - <<PY\n'
     'import sys\n'
     'print("[install] python", sys.version)\n'
     'import torch; print("[install] torch", torch.__version__, "cuda", torch.version.cuda)\n'
     'import torch_scatter, torch_sparse\n'
     'print("[install] torch_scatter", torch_scatter.__version__)\n'
     'print("[install] torch_sparse", torch_sparse.__version__)\n'
     'import torch_geometric; print("[install] torch_geometric", torch_geometric.__version__)\n'
     'from torch_geometric.graphgym.config import cfg\n'
     'print("[install] torch_geometric.graphgym OK")\n'
     'import yacs, ogb, torchmetrics, tensorboardX, opt_einsum\n'
     'print("[install] supporting libs OK")\n'
     'PY'),
]


def install_deps():
    """Run each install step with live-streamed output."""
    import subprocess
    import sys

    print("[install] Installing GRIT dependencies "
          "(each step streams live; takes ~2-3 min total)")
    print("=" * 78)
    for label, cmd in INSTALL_STEPS:
        print(f"\n[install] >>> {label}")
        print(f"[install]     $ {cmd.splitlines()[0]}"
              + (" ..." if "\n" in cmd else ""))
        proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write("    " + line)
            sys.stdout.flush()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"[install] Step failed ({label!r}) with exit code "
                f"{proc.returncode}. Scroll up for the full log."
            )
    print("=" * 78)
    print("[install] All dependency steps completed successfully.")


# ============================================================================
# CELL 4 -- Enter the working directory and sanity-check the repo layout
# ============================================================================

def enter_workdir():
    """Locate main.py inside the cloned repo, chdir there, sanity-check."""
    import os
    import sys
    global WORK_DIR

    # main.py may live at the repo root or inside a top-level subfolder
    # (e.g. GRIT/grit/main.py). Walk up to 2 levels to find it.
    candidates = [CLONE_DIR]
    try:
        for entry in sorted(os.listdir(CLONE_DIR)):
            sub = os.path.join(CLONE_DIR, entry)
            if os.path.isdir(sub):
                candidates.append(sub)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"CLONE_DIR={CLONE_DIR!r} does not exist -- did clone_repo() run?"
        )

    found = None
    for c in candidates:
        if os.path.isfile(os.path.join(c, "main.py")):
            found = c
            break
    if found is None:
        raise FileNotFoundError(
            f"Could not find main.py in {CLONE_DIR} or any of its immediate "
            f"subdirectories. Contents: {os.listdir(CLONE_DIR)}"
        )

    WORK_DIR = found
    os.chdir(WORK_DIR)
    if WORK_DIR not in sys.path:
        sys.path.insert(0, WORK_DIR)
    print(f"[workdir] cwd = {os.getcwd()}")

    # Sanity: must contain main.py, grit/ package, configs/GRIT/, rope layer
    needed = ["main.py", "grit", "configs/GRIT/zinc-GRIT-RRWP.yaml",
              "grit/layer/grit_rope_layer.py"]
    missing = [p for p in needed if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "The following required paths are missing in WORK_DIR:\n  "
            + "\n  ".join(missing)
            + "\nThe clone is incomplete, on the wrong branch, or is missing "
              "the grit_rope_layer.py commit. Check GITHUB_BRANCH and that "
              "grit/layer/grit_rope_layer.py is committed on that branch."
        )
    print("[workdir] Repo layout looks correct.")


# ============================================================================
# CELL 5 -- Build the GritRoPE config (derived from zinc-GRIT-RRWP.yaml)
# ============================================================================

def build_config(cfg_out_path: str):
    """Write a new YAML config derived from zinc-GRIT-RRWP.yaml for the
    selected MODEL_VARIANT."""
    import os

    if MODEL_VARIANT not in _VARIANT_MAP:
        raise ValueError(
            f"Unknown MODEL_VARIANT={MODEL_VARIANT!r}. "
            f"Choose one of: {list(_VARIANT_MAP.keys())}"
        )
    class_name = _VARIANT_MAP[MODEL_VARIANT]

    base_path = "configs/GRIT/zinc-GRIT-RRWP.yaml"
    with open(base_path, "r") as f:
        base_yaml = f.read()

    # We deliberately do line-level string edits rather than loading as a dict
    # and dumping, so that comments / formatting of the original file are
    # preserved byte-for-byte for every field we don't touch.
    out_lines = []
    in_attn_block = False
    d_struct_injected = False

    for line in base_yaml.splitlines():
        stripped = line.rstrip()

        # Swap layer_type to the chosen variant.
        if stripped.startswith("  layer_type: GritTransformer"):
            out_lines.append(f"  layer_type: {class_name}")
            continue

        # Swap model.type to the chosen variant.
        if stripped.startswith("  type: GritTransformer"):
            out_lines.append(f"  type: {class_name}")
            continue

        # Track the attn: block so we can inject d_struct and edge_mlp inside it.
        if stripped.startswith("  attn:"):
            in_attn_block = True
            out_lines.append(line)
            continue
        if in_attn_block and line and not line.startswith("    ") and not line.startswith("\t"):
            # Exiting the attn: block -- inject d_struct + edge_mlp before we leave.
            if D_STRUCT is not None and not d_struct_injected:
                out_lines.append(f"    d_struct: {int(D_STRUCT)}")
                d_struct_injected = True
            if MODEL_VARIANT == "rope_pair":
                out_lines.append("    edge_mlp:")
                out_lines.append(f"      hidden_mult: {PAIR_MLP_HIDDEN_MULT}")
                out_lines.append(f"      dropout: {PAIR_MLP_DROPOUT}")
                out_lines.append(f"      act: '{PAIR_MLP_ACT}'")
                out_lines.append(f"      share_across_layers: {str(PAIR_MLP_SHARE_ACROSS_LAYERS)}")
            in_attn_block = False
        out_lines.append(line)

    # If attn: was the last block and we never injected, append now.
    if in_attn_block:
        if D_STRUCT is not None and not d_struct_injected:
            out_lines.append(f"    d_struct: {int(D_STRUCT)}")
        if MODEL_VARIANT == "rope_pair":
            out_lines.append("    edge_mlp:")
            out_lines.append(f"      hidden_mult: {PAIR_MLP_HIDDEN_MULT}")
            out_lines.append(f"      dropout: {PAIR_MLP_DROPOUT}")
            out_lines.append(f"      act: '{PAIR_MLP_ACT}'")
            out_lines.append(f"      share_across_layers: {str(PAIR_MLP_SHARE_ACROSS_LAYERS)}")

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
    print(f"[config] variant={MODEL_VARIANT}, class={class_name}, "
          f"max_epoch={epochs}, seed={SEED}, d_struct={D_STRUCT}")
    if MODEL_VARIANT == "rope_pair":
        print(f"[config] edge_mlp: hidden_mult={PAIR_MLP_HIDDEN_MULT}, "
              f"share_across_layers={PAIR_MLP_SHARE_ACROSS_LAYERS}, "
              f"dropout={PAIR_MLP_DROPOUT}, act={PAIR_MLP_ACT}")
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

    # We run main.py through a tiny shim that monkey-patches torch.load to
    # default `weights_only=False` before any PyG dataset is touched. This is
    # needed because PyTorch 2.6 changed the default of `weights_only` from
    # False to True, but PyG 2.3.1's cached ZINC dataset was pickled with
    # plain Python objects (torch_geometric.data.data.Data), which the new
    # safe loader refuses to unpickle. Upgrading PyG would drop graphgym, so
    # the monkey-patch is the least invasive fix. The patch is safe here
    # because every dataset we load is downloaded by PyG itself (trusted).
    shim = "\n".join([
        "import runpy",
        "import sys",
        "import torch",
        "_orig_load = torch.load",
        "def _patched_load(*a, **k):",
        "    k.setdefault('weights_only', False)",
        "    return _orig_load(*a, **k)",
        "torch.load = _patched_load",
        "sys.argv = ['main.py'] + sys.argv[1:]",
        "runpy.run_path('main.py', run_name='__main__')",
    ])
    cmd = [
        sys.executable, "-u", "-c", shim,
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
        cwd=WORK_DIR,
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
    clone_repo()
    enter_workdir()
    install_deps()

    cfg_path = os.path.join(
        WORK_DIR, "configs", "GRIT", f"zinc-GRIT-RRWP-{MODEL_VARIANT}.yaml"
    )
    build_config(cfg_path)
    run_training(cfg_path)
    summarise_results()
    print(f"\n[done] All finished (variant={MODEL_VARIANT}). "
          "Check OUT_DIR for checkpoints and full logs:")
    print(f"       {OUT_DIR}")


if __name__ == "__main__":
    # In Colab, after setting the `diss_key` secret and selecting a GPU
    # runtime, you can either:
    #   (a) paste each CELL block into its own cell and run them in order, or
    #   (b) upload only THIS file to /content and run it in one cell:
    #         !python /content/colab_train_zinc_grit_rope.py
    main()
