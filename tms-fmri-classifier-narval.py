# %% [markdown]
# ## 0. Setup

# %%
# imports
import datetime
import os, io, json, warnings, gc, shutil, re, platform
from copy import deepcopy
from pathlib import Path
warnings.filterwarnings("ignore")

import fsspec

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.ndimage import zoom, gaussian_filter

import s3fs
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from monai.utils import set_determinism
from monai.networks.nets import DenseNet121

from sklearn.model_selection import train_test_split

# %%
# seeds
SEED = 42
set_determinism(seed=SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# %%
# --- outputs ---
# Generate a unique string with the current date and time (e.g., 20260401_153022)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Append the timestamp to your main project folder
WORK_DIR    = Path.cwd() / f"tms_fmri_project_{timestamp}"

MAPS_DIR    = WORK_DIR / "response_maps"  
TEMP_DIR    = WORK_DIR / "temp_nifti"      
RESULTS_DIR = WORK_DIR / "results"

for d in [WORK_DIR, MAPS_DIR, TEMP_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Data and results will be saved to: {WORK_DIR}")

# %%
# dataset
DATASET_ID = "ds005498"
S3_BUCKET  = "openneuro.org"

# acquisition parameters (from paper)
TR_TMS       = 2.4   # seconds
HRF_OFFSET   = 2     # volumes after pulse before averaging
HRF_WIN      = 3     # volumes to average (HRF peak ~5–7 s post-stim)
BASELINE_WIN = 2     # volumes before pulse for baseline

# cohort labels
GROUPS    = ["NTHC", "TEHC", "NTS", "NIS"]
LABEL_MAP = {g: i for i, g in enumerate(GROUPS)}

# %%
# hyperparams updated for Cluster Compute (16GB+ VRAM)
TARGET_SHAPE = (96, 96, 72) # double spatial res
BATCH_SIZE   = 16           # larger batch for stable norm
N_EPOCHS     = 60
LR           = 1e-4

NUM_WORKERS  = 8

# %%
#note: might not need device param
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device       : {DEVICE}")
print(f"TARGET_SHAPE : {TARGET_SHAPE}")
print(f"BATCH_SIZE   : {BATCH_SIZE}")
print(f"Work dir     : {WORK_DIR}")

# %% [markdown]
# ## 1. Deal With Data
# 

# %% [markdown]
# ### 1.1 S3 Functions
# For dealing with downloading and deleting files.

# %%
def get_s3():
    # Swap out S3 for the local file system
    return fsspec.filesystem("file")

def s3_ls(fs, prefix: str) -> list:
    try:
        return fs.ls(prefix, detail=False)
    except Exception as e:
        print(f"  [s3_ls] {prefix}: {e}")
        return []

def s3_read_tsv(fs, path: str) -> pd.DataFrame:
    with fs.open(path, "rb") as f:
        return pd.read_csv(f, sep="\t", low_memory=False)

def s3_download(fs, s3_path: str, local_path: Path):
    with fs.open(s3_path, "rb") as src, open(local_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

def extract_task(fname: str):
    m = re.search(r"task-([^_]+)", Path(fname).name)
    return m.group(1) if m else None

fs  = get_s3()
pfx = f"/scratch/{os.environ.get('USER')}/{S3_BUCKET}/{DATASET_ID}"
top = s3_ls(fs, pfx)
print(f"✓ Connected — top-level entries under s3://{pfx}: {len(top)}")


# %% [markdown]
# ### 1.2 Load and Clean Data

# %%
participants_df = s3_read_tsv(fs, f"{pfx}/participants.tsv")
print(f"Raw shape: {participants_df.shape}")
print(participants_df.head(3))

# get group from id string (e.g. sub-NTHC001 → NTHC)
def get_group_from_id(sid):
    s = str(sid).upper()
    if "NTHC" in s: return "NTHC"
    if "TEHC" in s: return "TEHC"
    if "NTS"  in s: return "NTS"
    if "NIS"  in s or "TIS" in s: return "NIS"
    return None

participants_df["group"] = participants_df["participant_id"].apply(get_group_from_id)

# standardize sex column name
for col in participants_df.columns:
    if col.lower().strip() in ("sex", "gender", "sex_at_birth", "biological_sex"):
        participants_df = participants_df.rename(columns={col: "sex"})
        break

#  sex as [0, 1]
# BIDS convention: 1=Female, 2=Male 
SEX_NORM = {
    "M": 1.0, "m": 1.0, "male": 1.0, "Male": 1.0,
    "2": 1.0, 2: 1.0,
    "F": 0.0, "f": 0.0, "female": 0.0, "Female": 0.0,
    "1": 0.0, 1: 0.0,
}
participants_df["sex_num"] = participants_df["sex"].map(SEX_NORM).fillna(0.5)

# int labels and clean
participants_df["label"]   = participants_df["group"].map(LABEL_MAP)
participants_df            = participants_df[participants_df["label"].notna()].copy()
participants_df            = participants_df.rename(columns={"participant_id": "subject_id"})
participants_df["label"]   = participants_df["label"].astype(int)
participants_df            = participants_df.reset_index(drop=True)

print("\nGroup distribution:")
print(participants_df["group"].value_counts())
print("\nSex numeric distribution:")
print(participants_df["sex_num"].value_counts())

# %% [markdown]
# ### 1.3 80/20 train/test split

# %%
all_sids = participants_df["subject_id"].values
all_labs = participants_df["label"].values

TEST_FRAC = 0.2

TRAIN_SIDS, TEST_SIDS, _, _ = train_test_split(
    all_sids, all_labs,
    test_size=TEST_FRAC,
    stratify=all_labs,
    random_state=SEED,
)
TRAIN_SIDS, TEST_SIDS = list(TRAIN_SIDS), list(TEST_SIDS)

print(f"Total     : {len(all_sids)}")
print(f"Train     : {len(TRAIN_SIDS)}  ({100*(1-TEST_FRAC):.0f}%)")
print(f"Test      : {len(TEST_SIDS)}   ({100*TEST_FRAC:.0f}%)")
print()
print("Train distribution:")
print(participants_df[participants_df["subject_id"].isin(TRAIN_SIDS)]["group"].value_counts().to_string())
print()
print("Test distribution:")
print(participants_df[participants_df["subject_id"].isin(TEST_SIDS)]["group"].value_counts().to_string())


# %% [markdown]
# ### 1.4 Get TMS Task Names & Isolate Causal BOLD Signal
# Helping convert from 4D to 3D data.

# %%
# --- getting the TMS task names ---

def get_subject_func_files(fs, pfx, subject_id):
    files = []
    files.extend(s3_ls(fs, f"{pfx}/{subject_id}/func"))
    sub_items = s3_ls(fs, f"{pfx}/{subject_id}")
    for item in sub_items:
        if "ses-" in item.split("/")[-1]:
            files.extend(s3_ls(fs, f"{item}/func"))
    return files

def discover_all_tms_tasks(fs, pfx, subject_ids, max_search=20):
    all_tasks = set()
    for sid in subject_ids[:max_search]:
        files = get_subject_func_files(fs, pfx, sid)
        bold = [f for f in files if "bold.nii" in f]
        for f in bold:
            task = extract_task(f)
            if task and "rest" not in task.lower():
                all_tasks.add(task)
        # break early if we found all 11 sites mentioned in the paper
        if len(all_tasks) == 11:
            break
    return sorted(list(all_tasks))

# use the new function to scan multiple subjects instead of just one
all_subject_ids = participants_df["subject_id"].tolist()
TMS_TASKS = discover_all_tms_tasks(fs, pfx, all_subject_ids)

TASK_TO_IDX      = {t: i for i, t in enumerate(TMS_TASKS)}
N_SITES          = len(TASK_TO_IDX)
SITE_IDX_TO_TASK = {v: k for k, v in TASK_TO_IDX.items()}

print(f"\nTMS site -> channel index ({N_SITES} sites):")
for t, i in TASK_TO_IDX.items():
    print(f"  [{i:2d}] {t}")

# %% [markdown]
# ### 1.5 TMS Response Map Computation

# %%

def load_pulse_onsets(fs, events_s3: str) -> np.ndarray:
    try:
        ev = s3_read_tsv(fs, events_s3)
        if "onset" in ev.columns:
            return ev["onset"].dropna().values.astype(float)
    except Exception:
        pass
    return np.array([])


def compute_response_map(
    bold_4d: np.ndarray,
    onsets: np.ndarray,
    tr: float = TR_TMS,
    hrf_offset: int = HRF_OFFSET,
    hrf_win: int = HRF_WIN,
    base_win: int = BASELINE_WIN,
) -> np.ndarray:
    """Voxelwise average TMS response map: (post-stim mean) - (pre-stim baseline)."""
    T = bold_4d.shape[3]
    post_maps, base_maps = [], []
    for t0 in onsets:
        v0     = int(t0 / tr)
        v_post = v0 + hrf_offset
        v_end  = v_post + hrf_win
        v_base = max(0, v0 - base_win)
        if v_end <= T and v0 > v_base:
            post_maps.append(bold_4d[..., v_post:v_end].mean(axis=3))
            base_maps.append(bold_4d[..., v_base:v0].mean(axis=3))
    if not post_maps:
        return np.zeros(bold_4d.shape[:3], dtype=np.float32)
    return (np.stack(post_maps).mean(0) - np.stack(base_maps).mean(0)).astype(np.float32)


def resample_volume(vol: np.ndarray, target: tuple = TARGET_SHAPE) -> np.ndarray:
    factors = tuple(t / s for t, s in zip(target, vol.shape))
    return gaussian_filter(zoom(vol, factors, order=1, prefilter=False), sigma=0.5).astype(np.float32)


def z_score(vol: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return ((vol - vol.mean()) / (vol.std() + eps)).astype(np.float32)


# %% [markdown]
# ### 1.6 Data Preprocessing

# %%
MAX_SUBJECTS = None   # set to e.g. 10 for a quick test

def process_subject(fs, pfx, subject_id, maps_dir, temp_dir):
    sub_dir = maps_dir / subject_id
    sub_dir.mkdir(exist_ok=True)

    # use the helper function here instead of s3_ls directly
    files = get_subject_func_files(fs, pfx, subject_id)
    
    bold_by_task = {
        extract_task(f): f
        for f in files
        if "bold.nii" in f and extract_task(f) in TASK_TO_IDX
    }

    # the rest of the function stays exactly the same
    results = {}
    for task, bold_s3 in bold_by_task.items():
        site_idx = TASK_TO_IDX[task]
        npy_path = sub_dir / f"site_{site_idx:02d}.npy"

        if npy_path.exists():
            results[site_idx] = npy_path
            continue

        events_s3 = re.sub(r"bold\.nii(\.gz)?$", "events.tsv", bold_s3)
        temp_nii  = temp_dir / f"{subject_id}_{task}.nii.gz"
        try:
            s3_download(fs, bold_s3, temp_nii)
            bold_img  = nib.load(str(temp_nii))
            bold_data = bold_img.get_fdata(dtype=np.float32)
            onsets    = load_pulse_onsets(fs, events_s3)
            if len(onsets) == 0:
                n_vols = bold_data.shape[3]
                onsets = np.linspace(TR_TMS * 4, n_vols * TR_TMS - TR_TMS * 4, 68)
                print(f"      warn {task}: no events.tsv, using estimated timings")
            resp = compute_response_map(bold_data, onsets)
            resp = resample_volume(resp, TARGET_SHAPE)
            resp = z_score(resp)
            np.save(str(npy_path), resp)
            results[site_idx] = npy_path
            del bold_img, bold_data, resp
            gc.collect()
        except Exception as e:
            print(f"      error {subject_id}/{task}: {e}")
        finally:
            if temp_nii.exists():
                temp_nii.unlink()
    return results

subject_ids = participants_df["subject_id"].tolist()
if MAX_SUBJECTS:
    subject_ids = subject_ids[:MAX_SUBJECTS]

print(f"Preprocessing {len(subject_ids)} subjects (skipping already-done) …\n")
SUBJECT_MAPS: dict = {}

for sid in tqdm(subject_ids, desc="Subjects"):
    SUBJECT_MAPS[sid] = process_subject(fs, pfx, sid, MAPS_DIR, TEMP_DIR)

done      = sum(1 for v in SUBJECT_MAPS.values() if v)
site_cnts = [len(v) for v in SUBJECT_MAPS.values() if v]
print(f"\n✅ Done. {done} subjects with ≥1 map.")
if site_cnts:
    print(f"   Sites per subject — mean {np.mean(site_cnts):.1f}, min {min(site_cnts)}, max {max(site_cnts)}")


# %%
# --- 1.6b Visualize Preprocessing ---

def visualize_preprocessing(fs, pfx, subject_id, task, temp_dir, maps_dir):
    print(f"Fetching data for {subject_id} - {task}...")
    
    # 1. Get the Raw Data
    files = get_subject_func_files(fs, pfx, subject_id)
    bold_s3 = next((f for f in files if task in f and "bold.nii" in f), None)
    
    if not bold_s3:
        print("Raw file not found on S3. Try a different subject/task.")
        return
        
    temp_nii = temp_dir / f"vis_{subject_id}_{task}.nii.gz"
    s3_download(fs, bold_s3, temp_nii)
    
    raw_img = nib.load(str(temp_nii))
    raw_data = raw_img.get_fdata(dtype=np.float32)
    
    # Take the mean across the time dimension (4th dim) to get a stable structural view
    raw_mean = np.mean(raw_data, axis=3) 
    temp_nii.unlink() # Clean up the heavy raw file
    
    # 2. Get the Processed Data
    site_idx = TASK_TO_IDX.get(task)
    processed_path = maps_dir / subject_id / f"site_{site_idx:02d}.npy"
    
    if not processed_path.exists():
        print(f"Processed file not found at {processed_path}.")
        return
        
    processed_data = np.load(str(processed_path))
    
    # 3. Plot the Middle Slices (Sagittal, Coronal, Axial)
    # Find the center index for all three dimensions
    r_x, r_y, r_z = [s // 2 for s in raw_mean.shape]
    p_x, p_y, p_z = [s // 2 for s in processed_data.shape]
    
    # Need to display plots in Jupyter, so we temporarily override the 'Agg' backend
    # %matplotlib inline 
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f"Preprocessing Comparison: {subject_id} | Task: {task}", fontsize=16)
    
    # --- Row 1: Raw Mean (Grayscale) ---
    axes[0, 0].imshow(np.rot90(raw_mean[r_x, :, :]), cmap='gray')
    axes[0, 0].set_title(f"Raw (Mean) - Sagittal\nShape: {raw_mean.shape}")
    
    axes[0, 1].imshow(np.rot90(raw_mean[:, r_y, :]), cmap='gray')
    axes[0, 1].set_title(f"Raw (Mean) - Coronal")
    
    axes[0, 2].imshow(np.rot90(raw_mean[:, :, r_z]), cmap='gray')
    axes[0, 2].set_title(f"Raw (Mean) - Axial")
    
    # --- Row 2: Processed Response Map (Diverging Colormap) ---
    # Because it is z-scored, 0 is the mean. We use a blue-white-red colormap to show +/- deviations.
    vmax = np.max(np.abs(processed_data))
    
    axes[1, 0].imshow(np.rot90(processed_data[p_x, :, :]), cmap='coolwarm', vmin=-vmax, vmax=vmax)
    axes[1, 0].set_title(f"Processed (Z-Scored) - Sagittal\nShape: {processed_data.shape}")
    
    axes[1, 1].imshow(np.rot90(processed_data[:, p_y, :]), cmap='coolwarm', vmin=-vmax, vmax=vmax)
    axes[1, 1].set_title(f"Processed (Z-Scored) - Coronal")
    
    axes[1, 2].imshow(np.rot90(processed_data[:, :, p_z]), cmap='coolwarm', vmin=-vmax, vmax=vmax)
    axes[1, 2].set_title(f"Processed (Z-Scored) - Axial")
    
    for ax in axes.flatten():
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(maps_dir / f"vis_{subject_id}_{task}.png", dpi=300)
    
    # Re-enable 'Agg' backend if you have background plotting later in the script
    matplotlib.use("Agg")

# Pick a subject and task that we know successfully processed
valid_subjects = [s for s, maps in SUBJECT_MAPS.items() if len(maps) > 0]
if valid_subjects:
    sample_sub = valid_subjects[0]
    sample_task = SITE_IDX_TO_TASK[list(SUBJECT_MAPS[sample_sub].keys())[0]]
    
    visualize_preprocessing(fs, pfx, sample_sub, sample_task, TEMP_DIR, MAPS_DIR)
else:
    print("No processed subjects found yet. Run the preprocessing loop first.")

# %% [markdown]
# ### 1.7 Dataset Class for Pytorch

# %%
# --- dataset and loader ---

# --- 1.7 Dataset Class and Dataloaders ---

from monai.transforms import (
    Compose, 
    RandGaussianNoised, 
    RandAffined,
    EnsureTyped
)

class SimpleTMSDataset(Dataset):
    def __init__(self, subject_ids, subject_maps, meta_df, transforms=None):
        self.sids = subject_ids
        self.maps = subject_maps
        self.meta = meta_df.set_index("subject_id")
        self.transforms = transforms
        
    def __len__(self):
        return len(self.sids)
        
    def __getitem__(self, idx):
        sid = self.sids[idx]
        info = self.meta.loc[sid]
        
        # Empty volume for all possible sites
        volume = np.zeros((N_SITES, *TARGET_SHAPE), dtype=np.float32)
        
        for si, path in self.maps.get(sid, {}).items():
            if si < N_SITES:
                volume[si] = np.load(str(path))
                
        label = int(info["label"])
        
        # Package as a dictionary for MONAI transforms
        sample = {"volume": volume, "label": label}
        
        if self.transforms:
            sample = self.transforms(sample)
            
        # Ensure label is the correct tensor type for CrossEntropyLoss
        sample["label"] = torch.tensor(sample["label"], dtype=torch.long)
            
        return sample

# Define training augmentations
train_transforms = Compose([
    RandGaussianNoised(keys=["volume"], prob=0.5, mean=0.0, std=0.1),
    RandAffined(
        keys=["volume"], 
        prob=0.5, 
        rotate_range=(0.05, 0.05, 0.05), # ~3 degree random rotations
        translate_range=(2, 2, 2)        # +/- 2 voxel random shifts
    ),
    EnsureTyped(keys=["volume"], dtype=torch.float32)
])

# Validation doesn't get augmented, just converted to tensor
val_transforms = Compose([
    EnsureTyped(keys=["volume"], dtype=torch.float32)
])

train_sids_ready = [s for s in TRAIN_SIDS if s in SUBJECT_MAPS and SUBJECT_MAPS[s]]
test_sids_ready = [s for s in TEST_SIDS if s in SUBJECT_MAPS and SUBJECT_MAPS[s]]

train_ds = SimpleTMSDataset(train_sids_ready, SUBJECT_MAPS, participants_df, transforms=train_transforms)
test_ds = SimpleTMSDataset(test_sids_ready, SUBJECT_MAPS, participants_df, transforms=val_transforms)

# drop_last=True fixes the Batch Normalization crash
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"Ready to train with {len(train_ds)} train subjects and {len(test_ds)} test subjects")

# %% [markdown]
# ## 2. Model Setup

# %%
# init MONAI 3D classification model
model = DenseNet121(
    spatial_dims=3, 
    in_channels=N_SITES,  # 11 channels, one for each stimulation site
    out_channels=4        # 4 clinical groups (NTHC, TEHC, NTS, NIS)
).to(DEVICE)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print("Model, loss function, and optimizer initialized.")

# %% [markdown]
# ## 3. Training Loop

# %%
print("Starting training...")

best_acc = 0

for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = 0
    
    # --- Training Step ---
    for batch in train_loader:
        inputs = batch["volume"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    # --- Evaluation Step ---
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["volume"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    test_acc = correct / total
    
    # print update every 5 epochs / on first epoch
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:02d}/{N_EPOCHS} - Train Loss: {epoch_loss/len(train_loader):.4f} - Test Acc: {test_acc:.4f}")
    
    # save best version of model
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), RESULTS_DIR / "best_model.pth")

print(f"\nFinished! Best test accuracy was {best_acc:.4f}")
print(f"Best model weights saved to: {RESULTS_DIR / 'best_model.pth'}")

# %% [markdown]
# ## 4. Evaluate

# %%
# --- 4. Confusion Matrix Evaluation ---

from sklearn.metrics import confusion_matrix

print("Loading best model weights for evaluation...")
model.load_state_dict(torch.load(RESULTS_DIR / "best_model.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        inputs = batch["volume"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Generate the confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plotting
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=GROUPS, 
    yticklabels=GROUPS,
    cbar=False
)
plt.xlabel('Predicted Clinical Group', fontsize=12, fontweight='bold')
plt.ylabel('Actual Clinical Group', fontsize=12, fontweight='bold')
plt.title('Test Set Confusion Matrix', fontsize=14, pad=15)
plt.tight_layout()

# Save and show
plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=300)
plt.show()

# %% [markdown]
# notes: try doing visualization of what data looks like before going into model
# 
# 1. try higher resolution with more compute -- email mark and maged
# 2. 1.35 is pretty high train loss (still underfit) do more training, plot accuracy and loss (overfit if loss increase while acc decrease)
# 3. add in data vis - show examples, class averages
# 4. also print confusion matrix (which classes are correct and which commonly are mixed up)
# 5. see if there's augmentations people tend to use in lit for other fmri classification i.e. adding guassian noise to improve generalization
# 
# 
# other notes (maybe do if time)
# exploratory justification--discovering new biomarker
# interpretability methods --> gradcam (python packages exist for this) will show you which region of the image it's using
# 
# during testing - split cases into categories and do stimulation site specific analysis
# mini model idea could make dataset really small -- maybe just split sites for testing


