#!/bin/bash
#SBATCH --account=def-mchiew           # PI's account name
#SBATCH --gpus-per-node=1              # any gpu
#SBATCH --cpus-per-task=8              # Matches the NUM_WORKERS = 8 in your code
#SBATCH --mem=16G                      # Plenty of RAM for loading the MRI data
#SBATCH --time=03:00:00                # Max runtime 
#SBATCH --job-name=tms_fmri_train
#SBATCH --output=%x-%j.out             # Saves all your print statements to a log file
#SBATCH --qos=test

# 1. load Narval modules
module load StdEnv/2023
module load python/3.11
module load arrow/15.0.1

# 2. create + activate venv
virtualenv --no-download ~/tms_env
source ~/tms_env/bin/activate

# 3. install python packages
pip install --no-index --upgrade pip
pip install monai s3fs nibabel pandas seaborn scikit-learn tqdm scipy

# 4. run code!
python tms-fmri-classifier-narval.py