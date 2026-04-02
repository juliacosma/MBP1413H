#!/bin/bash
#SBATCH --account=def-mchiew           # VERY IMPORTANT: Replace with your PI's account name
#SBATCH --gpus-per-node=a100:1         # Request 1 A100 GPU (Narval's standard powerhouse)
#SBATCH --cpus-per-task=8              # Matches the NUM_WORKERS = 8 in your code
#SBATCH --mem=64G                      # Plenty of RAM for loading the MRI data
#SBATCH --time=04:00:00                # Max runtime (4 hours should be plenty for 60 epochs)
#SBATCH --job-name=tms_fmri_train
#SBATCH --output=%x-%j.out             # Saves all your print statements to a log file

# 1. load Narval modules
module load StdEnv/2023
module load python/3.11
module load arrow/15.0.1

# 2. activate venv (Do NOT use --no-download or virtualenv here anymore)
source ~/tms_env/bin/activate

# 3. run code!
python BINARYCLASS-tms-fmri-classifier-narval.py