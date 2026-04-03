#!/bin/bash
#SBATCH --account=def-mchiew        # same PI account as your training jobs
#SBATCH --cpus-per-task=4           # enough for parallel numpy/scipy
#SBATCH --mem=32G                   # loading 152 × 11 × (96×96×72) float32 maps
#SBATCH --time=00:10:00             # 1 hour is plenty — no GPU needed
#SBATCH --job-name=cohort_vis
#SBATCH --output=%x-%j.out          # log → cohort_vis-<jobid>.out

# ── 1. load the same modules as the training runs ──────────────────────────
module load StdEnv/2023
module load python/3.11
module load arrow/15.0.1

# ── 2. activate the same virtual environment ───────────────────────────────
source ~/tms_env/bin/activate

# ── 3. run the visualisation script ────────────────────────────────────────
echo "Starting cohort_vis.py at $(date)"
python cohort_vis.py
echo "Finished at $(date)"
