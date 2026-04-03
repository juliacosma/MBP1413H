"""
cohort_vis.py  —  MBP1413H TMS-fMRI Cohort Visualisation
=========================================================
Reads preprocessed .npy response maps already on Narval scratch,
generates cohort-averaged figures, and saves everything to
  <script_dir>/cohort-data-vis/

Run via SLURM (see cohort_vis_job.sh) or directly:
  python cohort_vis.py

Paths are discovered automatically from:
  /scratch/$USER/openneuro.org/ds005498/
  /scratch/$USER/tms_fmri_project_*/response_maps/   (most recent run)
"""

import os, re, warnings, gc
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import zoom

# ─────────────────────────────────────────────────────────────────────────────
# 0.  PATH DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

USER        = os.environ.get("USER", "")
HOME        = Path(f"/home/{USER}")
SCRATCH     = Path(f"/scratch/{USER}")
DATASET_DIR = SCRATCH / "openneuro.org" / "ds005498"
PARTICIPANTS_TSV = DATASET_DIR / "participants.tsv"

# Find the most-recently-modified tms_fmri_project_*/response_maps directory
# that actually contains .npy files — search both HOME and SCRATCH.
def find_maps_dir() -> Path:
    candidates = []
    for base in [HOME, SCRATCH]:
        candidates.extend(base.glob("tms_fmri_project_*/response_maps"))
    # sort by modification time, newest first
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for c in candidates:
        if any(c.glob("sub-*/site_*.npy")):
            return c
    raise FileNotFoundError(
        f"No tms_fmri_project_*/response_maps with .npy files found under\n"
        f"  {HOME}\n  {SCRATCH}\n"
        f"Make sure the preprocessing run has completed."
    )

MAPS_DIR = find_maps_dir()
print(f"✓ Using maps dir : {MAPS_DIR}")
print(f"✓ Participants   : {PARTICIPANTS_TSV}")

# Output folder — same directory as this script
OUT_DIR = Path(__file__).resolve().parent / "cohort-data-vis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"✓ Output dir     : {OUT_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD PARTICIPANTS
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(PARTICIPANTS_TSV, sep="\t", low_memory=False)

def get_group(sid):
    s = str(sid).upper()
    if "NTHC" in s: return "NTHC"
    if "TEHC" in s: return "TEHC"
    if "NTS"  in s: return "NTS"
    if "TIS"  in s or "NIS" in s: return "TIS"
    return None

df["group"]      = df["participant_id"].apply(get_group)
df["subject_id"] = df["participant_id"]
df = df[df["group"].notna()].reset_index(drop=True)

print("\nGroup distribution:")
print(df["group"].value_counts().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 2.  DISCOVER TASK→CHANNEL MAPPING  (mirrors the training script)
# ─────────────────────────────────────────────────────────────────────────────

# Infer number of channels from the .npy files present
def get_n_sites(maps_dir: Path) -> int:
    indices = set()
    for npy in maps_dir.rglob("site_*.npy"):
        m = re.search(r"site_(\d+)\.npy", npy.name)
        if m:
            indices.add(int(m.group(1)))
    return max(indices) + 1 if indices else 11

N_SITES      = get_n_sites(MAPS_DIR)
TARGET_SHAPE = (96, 96, 72)   # matches HPC training run
print(f"\n✓ N_SITES detected : {N_SITES}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

COHORT_COLORS = {
    "NTHC": "#20808D",
    "TEHC": "#A84B2F",
    "NTS":  "#1B474D",
    "TIS":  "#944454",
}
BINARY_COLORS = {
    "Non-Trauma":     "#20808D",
    "Trauma-Exposed": "#A84B2F",
}

def load_vol(path: Path) -> np.ndarray:
    vol = np.load(str(path)).astype(np.float32)
    if vol.shape != TARGET_SHAPE:
        factors = tuple(t / s for t, s in zip(TARGET_SHAPE, vol.shape))
        vol = zoom(vol, factors, order=1)
    return vol

def load_group_site(group: str, site_idx: int) -> list:
    """Return list of 3-D arrays for a group × site combination."""
    sids   = df[df["group"] == group]["subject_id"].tolist()
    arrays = []
    for sid in sids:
        npy = MAPS_DIR / sid / f"site_{site_idx:02d}.npy"
        if npy.exists():
            arrays.append(load_vol(npy))
    return arrays

def load_subject_all_sites(sid: str) -> np.ndarray | None:
    """Average across all available sites for one subject → (96,96,72)."""
    vols = []
    for i in range(N_SITES):
        npy = MAPS_DIR / sid / f"site_{i:02d}.npy"
        if npy.exists():
            vols.append(load_vol(npy))
    if not vols:
        return None
    return np.mean(vols, axis=0)

def mid_slices(vol: np.ndarray):
    """Return centre slices, each cropped to a square so all panels
    are the same size and no grey padding shows around the image."""
    x, y, z = vol.shape
    slices_raw = [vol[x//2, :, :], vol[:, y//2, :], vol[:, :, z//2]]
    out = []
    for s in slices_raw:
        h, w = s.shape
        side  = min(h, w)
        r0    = (h - side) // 2
        c0    = (w - side) // 2
        out.append(s[r0:r0+side, c0:c0+side])
    return out

def cmap_kw(vmax: float):
    return dict(cmap="coolwarm",
                norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))

def save(fig, name: str):
    p = OUT_DIR / name
    fig.savefig(str(p), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved  {name}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  FIGURE A  —  Four-cohort averaged response maps, one site at a time
#     Loops over all N_SITES and saves one panel per site.
# ─────────────────────────────────────────────────────────────────────────────

print("\n[A] Generating per-site group-averaged response maps …")

GROUPS_4 = ["NTHC", "TEHC", "NTS", "TIS"]
VIEW_LABELS = ["Sagittal", "Coronal", "Axial"]

for site_idx in range(N_SITES):
    avgs, ns = {}, {}
    all_vmax  = 0.0

    for g in GROUPS_4:
        arrays = load_group_site(g, site_idx)
        ns[g]  = len(arrays)
        if arrays:
            avg = np.mean(arrays, axis=0)
            avgs[g] = avg
            all_vmax = max(all_vmax, float(np.percentile(np.abs(avg), 99)))
        else:
            avgs[g] = None

    if all_vmax == 0:
        all_vmax = 1.0
    ck = cmap_kw(all_vmax)

    fig, axes = plt.subplots(4, 3, figsize=(13, 15))
    fig.suptitle(
        f"Group-Averaged Causal BOLD Response  —  Site channel {site_idx:02d}",
        fontsize=13, fontweight="bold", y=0.98
    )

    plt.subplots_adjust(left=0.18, right=0.91, top=0.93, bottom=0.04,
                        hspace=0.12, wspace=0.06)

    for row, g in enumerate(GROUPS_4):
        avg = avgs[g]
        n_total = len(df[df["group"] == g])

        if avg is None:
            for col in range(3):
                axes[row, col].text(0.5, 0.5, "No data", ha="center", va="center",
                                    transform=axes[row, col].transAxes, fontsize=10)
                axes[row, col].axis("off")
        else:
            slices = mid_slices(avg)
            for col, (slc, vlbl) in enumerate(zip(slices, VIEW_LABELS)):
                axes[row, col].imshow(np.rot90(slc), **ck, aspect="equal"); axes[row, col].set_facecolor("black")
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(vlbl, fontsize=11, pad=6)

        # use figure-level text so label is never clipped by axis boundaries
        bbox = axes[row, 0].get_position()
        fig.text(0.01, bbox.y0 + bbox.height / 2,
                 f"{g}\n(n={ns[g]}/{n_total})",
                 fontsize=13, fontweight="bold",
                 color=COHORT_COLORS[g],
                 ha="left", va="center")

    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.65])
    sm = plt.cm.ScalarMappable(**ck); sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax).set_label("Z-score (BOLD Δ)", fontsize=10)
    save(fig, f"A_group_avg_site{site_idx:02d}.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  FIGURE B  —  Binary difference maps  (TE − NT)  per site
# ─────────────────────────────────────────────────────────────────────────────

print("\n[B] Generating binary difference maps (Trauma-Exposed − Non-Trauma) …")

for site_idx in range(N_SITES):
    nt_arrays = load_group_site("NTHC", site_idx) + load_group_site("NTS", site_idx)
    te_arrays = load_group_site("TEHC", site_idx) + load_group_site("TIS", site_idx)

    if not nt_arrays or not te_arrays:
        print(f"  → skipping site {site_idx:02d} (insufficient data)")
        continue

    diff  = np.mean(te_arrays, axis=0) - np.mean(nt_arrays, axis=0)
    vmax  = float(np.percentile(np.abs(diff), 99)) or 0.01
    ck    = cmap_kw(vmax)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(
        f"Trauma-Exposed − Non-Trauma BOLD Difference  —  Site channel {site_idx:02d}\n"
        f"NT n={len(nt_arrays)}, TE n={len(te_arrays)}",
        fontsize=12, fontweight="bold"
    )
    slices = mid_slices(diff)
    for ax, slc, lbl in zip(axes, slices, VIEW_LABELS):
        ax.imshow(np.rot90(slc), **ck, aspect="equal"); ax.set_facecolor("black")
        ax.set_title(lbl, fontsize=11)
        ax.axis("off")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
    sm = plt.cm.ScalarMappable(**ck); sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax).set_label("ΔZ-score (TE−NT)", fontsize=10)
    plt.subplots_adjust(right=0.90, wspace=0.05)
    save(fig, f"B_diff_map_site{site_idx:02d}.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  FIGURE C  —  All-site averaged maps: NT vs TE side-by-side
# ─────────────────────────────────────────────────────────────────────────────

print("\n[C] Generating all-site averaged binary comparison …")

binary_groups = {
    "Non-Trauma":     df[df["group"].isin(["NTHC", "NTS"])]["subject_id"].tolist(),
    "Trauma-Exposed": df[df["group"].isin(["TEHC", "TIS"])]["subject_id"].tolist(),
}
avgs_bin, ns_bin = {}, {}
all_vmax = 0.0

for label, sids in binary_groups.items():
    vols = []
    for sid in sids:
        v = load_subject_all_sites(sid)
        if v is not None:
            vols.append(v)
    avgs_bin[label] = np.mean(vols, axis=0) if vols else None
    ns_bin[label]   = len(vols)
    if vols:
        all_vmax = max(all_vmax, float(np.percentile(np.abs(avgs_bin[label]), 99)))
if all_vmax == 0:
    all_vmax = 1.0
ck = cmap_kw(all_vmax)

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
fig.suptitle("All-Site Averaged Causal BOLD Response\nNon-Trauma vs. Trauma-Exposed",
             fontsize=13, fontweight="bold", y=0.99)

for row, label in enumerate(binary_groups):
    avg = avgs_bin[label]
    if avg is None:
        for col in range(3): axes[row, col].axis("off")
        continue
    for col, (slc, vlbl) in enumerate(zip(mid_slices(avg), VIEW_LABELS)):
        axes[row, col].imshow(np.rot90(slc), **ck, aspect="equal"); axes[row, col].set_facecolor("black")
        axes[row, col].axis("off")
        if row == 0:
            axes[row, col].set_title(vlbl, fontsize=11, pad=6)
    bbox = axes[row, 0].get_position()
    fig.text(0.01, bbox.y0 + bbox.height / 2,
             f"{label}\n(n={ns_bin[label]})",
             fontsize=13, fontweight="bold",
             color=BINARY_COLORS[label],
             ha="left", va="center")

cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.65])
sm = plt.cm.ScalarMappable(**ck); sm.set_array([])
fig.colorbar(sm, cax=cbar_ax).set_label("Z-score (BOLD Δ)", fontsize=10)
plt.subplots_adjust(left=0.22, right=0.91, top=0.93, bottom=0.04,
                    hspace=0.10, wspace=0.06)
save(fig, "C_binary_allsite_avg.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  FIGURE D  —  Within-group variability (std dev) per site
# ─────────────────────────────────────────────────────────────────────────────

print("\n[D] Generating within-group variability maps …")

for site_idx in range(N_SITES):
    stds, ns_std = {}, {}
    all_vmax = 0.0

    for g in GROUPS_4:
        arrays = load_group_site(g, site_idx)
        ns_std[g] = len(arrays)
        if len(arrays) > 1:
            s = np.std(arrays, axis=0)
            stds[g] = s
            all_vmax = max(all_vmax, float(np.percentile(s, 99)))
        else:
            stds[g] = None
    if all_vmax == 0:
        all_vmax = 1.0

    fig, axes = plt.subplots(4, 3, figsize=(13, 15))
    fig.suptitle(
        f"Within-Group Response Variability (Std Dev)  —  Site channel {site_idx:02d}\n"
        "High variability = overlapping group distributions = harder classification",
        fontsize=12, fontweight="bold", y=0.98
    )

    for row, g in enumerate(GROUPS_4):
        s = stds[g]
        n_total = len(df[df["group"] == g])

        if s is None:
            for col in range(3):
                axes[row, col].text(0.5, 0.5, "Insufficient data",
                                    ha="center", va="center",
                                    transform=axes[row, col].transAxes, fontsize=9)
                axes[row, col].axis("off")
        else:
            for col, (slc, vlbl) in enumerate(zip(mid_slices(s), VIEW_LABELS)):
                axes[row, col].imshow(np.rot90(slc), cmap="viridis",
                                      vmin=0, vmax=all_vmax, aspect="equal")
                axes[row, col].set_facecolor("black")
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(vlbl, fontsize=11, pad=6)

        bbox = axes[row, 0].get_position()
        fig.text(0.01, bbox.y0 + bbox.height / 2,
                 f"{g}\n(n={ns_std[g]}/{n_total})",
                 fontsize=13, fontweight="bold",
                 color=COHORT_COLORS[g],
                 ha="left", va="center")

    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.65])
    sm = plt.cm.ScalarMappable(cmap="viridis",
                                norm=plt.Normalize(vmin=0, vmax=all_vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax).set_label("Std Dev of Z-score", fontsize=10)
    plt.subplots_adjust(left=0.18, right=0.91, top=0.93, bottom=0.04,
                        hspace=0.12, wspace=0.06)
    save(fig, f"D_variability_site{site_idx:02d}.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  FIGURE E  —  Per-site response magnitude bar chart  (all cohorts)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[E] Generating per-site magnitude bar chart …")

results = {g: [] for g in GROUPS_4}
for site_idx in range(N_SITES):
    for g in GROUPS_4:
        arrays = load_group_site(g, site_idx)
        if arrays:
            results[g].append(float(np.mean([np.mean(np.abs(a)) for a in arrays])))
        else:
            results[g].append(float("nan"))

x       = np.arange(N_SITES)
width   = 0.18
offsets = [-1.5, -0.5, 0.5, 1.5]

fig, ax = plt.subplots(figsize=(max(10, N_SITES * 1.4), 5))
for i, g in enumerate(GROUPS_4):
    ax.bar(x + offsets[i] * width, results[g], width,
           label=g, color=COHORT_COLORS[g], alpha=0.88,
           edgecolor="white", linewidth=0.5)

ax.set_xlabel("Site channel index", fontsize=12)
ax.set_ylabel("Mean |Z-score|  (response magnitude)", fontsize=12)
ax.set_title("Per-Site Mean Causal BOLD Response Magnitude by Cohort",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xticks(x)
ax.set_xticklabels([f"ch{i}" for i in range(N_SITES)], fontsize=9)
ax.legend(frameon=False, fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
save(fig, "E_per_site_magnitude.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  FIGURE F  —  Subject-count heatmap  (which subjects have which sites)
#     Useful for understanding the missingness pattern
# ─────────────────────────────────────────────────────────────────────────────

print("\n[F] Generating site-completion heatmap …")

all_sids = df["subject_id"].tolist()
matrix   = np.zeros((len(all_sids), N_SITES), dtype=np.float32)
for r, sid in enumerate(all_sids):
    for c in range(N_SITES):
        if (MAPS_DIR / sid / f"site_{c:02d}.npy").exists():
            matrix[r, c] = 1.0

# sort rows by group for visual clarity
group_order = ["NTHC", "TEHC", "NTS", "TIS"]
sorted_sids = []
group_boundaries = [0]
for g in group_order:
    g_sids = df[df["group"] == g]["subject_id"].tolist()
    sorted_sids.extend(g_sids)
    group_boundaries.append(len(sorted_sids))

sorted_idx = [all_sids.index(s) for s in sorted_sids if s in all_sids]
matrix_sorted = matrix[sorted_idx, :]

fig, ax = plt.subplots(figsize=(max(8, N_SITES * 0.9), max(10, len(sorted_sids) * 0.12)))
ax.imshow(matrix_sorted, aspect="auto", cmap="Blues", vmin=0, vmax=1,
          interpolation="nearest")

# group dividers
for b in group_boundaries[1:-1]:
    ax.axhline(b - 0.5, color="#A84B2F", linewidth=1.2, linestyle="--")

# group labels on y-axis
prev = 0
for g, b in zip(group_order, group_boundaries[1:]):
    mid = (prev + b) / 2
    ax.text(-0.6, mid, g, ha="right", va="center", fontsize=9,
            fontweight="bold", color=COHORT_COLORS[g],
            transform=ax.get_yaxis_transform())
    prev = b

ax.set_xticks(range(N_SITES))
ax.set_xticklabels([f"ch{i}" for i in range(N_SITES)], fontsize=9)
ax.set_yticks([])
ax.set_xlabel("Site channel index", fontsize=11)
ax.set_title("Site Completion Map (blue = scan present)\nDashed lines = cohort boundaries",
             fontsize=12, fontweight="bold")
plt.tight_layout()
save(fig, "F_site_completion_map.png")

# ─────────────────────────────────────────────────────────────────────────────
# 10.  FIGURE G  —  Grand-average across all subjects and all sites
#      (single summary image for report sanity check)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[G] Generating grand-average response map …")

all_vols = []
for sid in df["subject_id"].tolist():
    v = load_subject_all_sites(sid)
    if v is not None:
        all_vols.append(v)

if all_vols:
    grand = np.mean(all_vols, axis=0)
    vmax  = float(np.percentile(np.abs(grand), 99)) or 1.0
    ck    = cmap_kw(vmax)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(
        f"Grand-Averaged Causal BOLD Response — All subjects, all sites  (N={len(all_vols)})",
        fontsize=12, fontweight="bold"
    )
    for ax, slc, lbl in zip(axes, mid_slices(grand), VIEW_LABELS):
        ax.imshow(np.rot90(slc), **ck, aspect="equal"); ax.set_facecolor("black")
        ax.set_title(lbl, fontsize=11)
        ax.axis("off")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
    sm = plt.cm.ScalarMappable(**ck); sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax).set_label("Z-score (BOLD Δ)", fontsize=10)
    plt.subplots_adjust(right=0.90, wspace=0.05)
    save(fig, "G_grand_average.png")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n✅  All figures saved to: {OUT_DIR}")
n_files = len(list(OUT_DIR.glob("*.png")))
print(f"   Total figures: {n_files}")
