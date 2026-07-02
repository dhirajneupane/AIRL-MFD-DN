"""
usad.py

USAD (UnSupervised Anomaly Detection, Audibert et al., KDD 2020) for
vibration-based machinery fault detection.

Default dataset: HUMS2023 (4095-point segments). IMS and XJTU-SY are also
supported (4096-point segments) -- see DATASET_PAIRS and EXPECTED_LEN below.
Differences between HUMS and the bearing datasets are marked inline with
"DATASET DIFFERENCE" comments; everything else is dataset-agnostic.

Architecture (mirrors the MLP-autoencoder baseline):
    Shared encoder E : in_dim -> 1024 -> 256 -> 64
    Decoder G1       : 64 -> 256 -> 1024 -> in_dim
    Decoder G2       : 64 -> 256 -> 1024 -> in_dim

Training (two-phase, per Audibert et al.):
    Phase 1 (AE1, params E+G1):
        L_AE1 = w * MSE(x, G1(E(x))) - (1-w) * MSE(x, G2(G1(E(x))))
    Phase 2 (AE2, params E+G2):
        L_AE2 = w * MSE(x, G2(E(x))) + (1-w) * MSE(x, G2(G1(E(x))))
    where w = 1/epoch, so the adversarial term dominates as training
    progresses.

Anomaly score:
    A(x) = alpha * MSE(x, G1(E(x))) + (1-alpha) * MSE(x, G2(G1(E(x))))
    Higher score = more anomalous.

Requirements: torch, numpy, scipy, scikit-learn, scikit-image
"""

import os
import json
import glob
import time
import re
import random
import csv

import numpy as np
from scipy.io import loadmat
from scipy.stats import genpareto
from numpy.lib.format import open_memmap

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================================================== #
#  USER SETTINGS
# ================================================================== #

# Each entry: (train_dir, test_dir, tag, mat_key)
# mat_key is the variable name storing the signal inside each .mat file.
DATASET_PAIRS = [
    ("./data/HUMS2023/train", "./data/HUMS2023/test", "HUMS_RF2", "xr"),

    # DATASET DIFFERENCE: IMS and XJTU-SY are 4096-point segments, not 4095.
    # To run either, set EXPECTED_LEN = 4096 below and uncomment the entry.
    # ("./data/IMS/train", "./data/IMS/test", "IMS_Set1", "IMS"),
    # ("./data/XJTUSY/train", "./data/XJTUSY/test", "XJTUSY_31", "xjtusy"),
]

OUTPUT_DIR = "./usad_runs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- model ----
HIDDEN_DIMS = [1024, 256, 64]   # encoder dims; decoders mirror these back
ALPHA       = 0.5               # anomaly score weight: alpha*AE1 + (1-alpha)*AE2_adv
USE_L1_LOSS = False              # False -> MSE, True -> L1, for both phases

# ---- training ----
LR           = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS       = 300
BATCH_SIZE   = 128
PATIENCE     = 10                # early stopping on AE1 validation reconstruction
VAL_SPLIT    = 0.10
MAX_SAMPLES  = None
GRAD_CLIP    = 1.0
SAVE_MODELS  = True

# ---- thresholds ----
USE_GPD       = True
GPD_TAIL_FRAC = 0.05
GPD_PCTS      = [0.95, 0.98, 0.99]
EPS           = 1e-12

# ---- multi-run ----
N_RUNS = 10
SEEDS  = list(range(N_RUNS))

# ---- data ----
# DATASET DIFFERENCE: 4095 for HUMS2023. Set to 4096 for IMS / XJTU-SY.
EXPECTED_LEN = 4095
USE_SCALER   = False              # data already z-normalised; True for raw signals
REMOVE_DC    = False

PRINT_MODEL_SUMMARY_ONCE = True
SEED_GLOBAL = 0


# ================================================================== #
#  HELPERS
# ================================================================== #

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_num_re = re.compile(r"(\d+)")


def _natural_key(name: str):
    """Numeric-aware sort key so '2_file' < '10_file'."""
    parts = _num_re.split(os.path.basename(str(name)))
    key = []
    for p in parts:
        if p.isdigit():
            key.append((1, int(p)))
        else:
            key.append((0, p.lower()))
    return tuple(key)


def load_folder_mat_1d(folder, key, expected_len):
    """
    Load all .mat files from `folder` in natural filename order.
    `key` is the variable name inside the .mat file (dataset-specific,
    see DATASET_PAIRS). Falls back to a lowercase key match, then to any
    array of the right size.
    """
    files = sorted(glob.glob(os.path.join(folder, "*.mat")), key=_natural_key)
    X, F = [], []
    for fp in files:
        m = loadmat(fp)
        arr = m.get(key, None)
        if arr is None:
            arr = m.get(key.lower(), None)
        if arr is None:
            for v in m.values():
                if isinstance(v, np.ndarray) and v.size == expected_len:
                    arr = v
                    break
        if arr is None:
            continue
        x = np.asarray(arr, dtype=np.float32).reshape(-1)
        if expected_len and len(x) != expected_len:
            continue
        X.append(x)
        F.append(fp)
    if not X:
        raise RuntimeError(
            f"No usable .mat files found in {folder}. Tried key='{key}'. "
            f"Check the variable name stored in your .mat files."
        )
    X = np.vstack(X)
    X = np.nan_to_num(X, copy=False)
    return X, np.array(F, dtype=object)


def standardize_train_test(Xtr_raw, Xte_raw, mode="train"):
    """Normalise using TRAIN statistics only, to prevent leakage."""
    Xtr, Xte = Xtr_raw.copy(), Xte_raw.copy()

    if REMOVE_DC:
        Xtr = Xtr - Xtr.mean(axis=1, keepdims=True)
        Xte = Xte - Xte.mean(axis=1, keepdims=True)

    if mode.lower() == "train":
        if USE_SCALER:
            sc = StandardScaler().fit(Xtr.reshape(-1, 1))
            Xtr = sc.transform(Xtr.reshape(-1, 1)).reshape(Xtr.shape)
            Xte = sc.transform(Xte.reshape(-1, 1)).reshape(Xte.shape)
            scaler_mean  = sc.mean_.astype(np.float32)
            scaler_scale = sc.scale_.astype(np.float32)
        else:
            mu  = float(Xtr.flatten().mean())
            std = float(Xtr.flatten().std())
            Xtr = (Xtr - mu) / (std + 1e-9)
            Xte = (Xte - mu) / (std + 1e-9)
            scaler_mean  = np.full(Xtr.shape[1], mu, dtype=np.float32)
            scaler_scale = np.full(Xtr.shape[1], std, dtype=np.float32)
    elif mode.lower() == "none":
        scaler_mean  = np.zeros(Xtr.shape[1], dtype=np.float32)
        scaler_scale = np.ones(Xtr.shape[1], dtype=np.float32)
    else:
        raise ValueError(f"Unknown standardization mode: {mode}")

    return Xtr, Xte, scaler_mean, scaler_scale


def subsample_rows(X, max_samples, seed):
    if max_samples is None or max_samples >= len(X):
        return X
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=max_samples, replace=False)
    return X[idx]


def write_json_safely(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# ================================================================== #
#  MODEL
# ================================================================== #

def _build_mlp(dims: list, add_relu_last: bool = False) -> nn.Sequential:
    """dims = [in, h1, ..., out]. ReLU after every layer except the last."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or add_relu_last:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class USAD(nn.Module):
    """
    USAD with a shared encoder and two independent decoders. Both decoders
    receive the same latent z = E(x); G2 also receives G1(z) during the
    adversarial phase.
    """

    def __init__(self, in_dim: int, hidden_dims: list):
        super().__init__()
        enc_dims = [in_dim] + hidden_dims
        dec_dims = list(reversed(hidden_dims)) + [in_dim]

        self.encoder  = _build_mlp(enc_dims)
        self.decoder1 = _build_mlp(dec_dims)   # G1
        self.decoder2 = _build_mlp(dec_dims)   # G2

    def encode(self, x):
        return self.encoder(x)

    def decode1(self, z):
        return self.decoder1(z)

    def decode2(self, z):
        return self.decoder2(z)

    def forward(self, x):
        """Returns (xhat1, xhat2, xhat2_adv) = (G1(E(x)), G2(E(x)), G2(G1(E(x))))."""
        z = self.encode(x)
        xhat1 = self.decode1(z)
        xhat2 = self.decode2(z)
        xhat2_adv = self.decode2(self.encode(xhat1))
        return xhat1, xhat2, xhat2_adv


def print_model_summary(model: USAD, in_dim: int):
    x = torch.randn(2, in_dim).to(DEVICE)
    print(model)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params: {total:,}  |  Trainable: {trainable:,}")
    model.eval()
    with torch.no_grad():
        h1, h2, h2a = model(x)
    print(f"Input:     {tuple(x.shape)}")
    print(f"xhat1:     {tuple(h1.shape)}")
    print(f"xhat2:     {tuple(h2.shape)}")
    print(f"xhat2_adv: {tuple(h2a.shape)}")


# ================================================================== #
#  TRAINING
# ================================================================== #

def train_one_run(Xtr: np.ndarray, seed: int, in_dim: int) -> USAD:
    """
    Train USAD for one seed. Early stopping monitors AE1 validation
    reconstruction loss. Returns the best model by val loss.
    """
    set_seed(seed)
    Xfit = subsample_rows(Xtr, MAX_SAMPLES, seed)

    n = len(Xfit)
    n_val = max(1, int(VAL_SPLIT * n))
    idx = np.random.permutation(n)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    Xtr_t  = torch.from_numpy(Xfit[tr_idx]).float()
    Xval_t = torch.from_numpy(Xfit[val_idx]).float().to(DEVICE)

    dl_tr = DataLoader(
        TensorDataset(Xtr_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )

    model = USAD(in_dim, HIDDEN_DIMS).to(DEVICE)

    # Two optimisers: opt1 updates (E, G1), opt2 updates (E, G2), so the
    # shared encoder receives gradients from both -- correct USAD setup.
    opt1 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder1.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    opt2 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder2.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )

    crit = nn.L1Loss(reduction="mean") if USE_L1_LOSS else nn.MSELoss(reduction="mean")

    best_val, best_state, bad = float("inf"), None, 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        w = 1.0 / ep   # adversarial weight; shrinks so phase 2 dominates over time

        for (xb,) in dl_tr:
            xb = xb.to(DEVICE, non_blocking=True)
            xhat1, xhat2, xhat2_adv = model(xb)

            # ---- phase 1: AE1 (E + G1) ----
            loss_ae1 = w * crit(xb, xhat1) - (1.0 - w) * crit(xb, xhat2_adv)
            opt1.zero_grad(set_to_none=True)
            loss_ae1.backward(retain_graph=True)
            if GRAD_CLIP is not None:
                nn.utils.clip_grad_norm_(
                    list(model.encoder.parameters()) + list(model.decoder1.parameters()),
                    GRAD_CLIP,
                )
            opt1.step()

            # ---- phase 2: AE2 (E + G2); re-forward since opt1.step() moved E ----
            xhat1, xhat2, xhat2_adv = model(xb)
            loss_ae2 = w * crit(xb, xhat2) + (1.0 - w) * crit(xb, xhat2_adv)
            opt2.zero_grad(set_to_none=True)
            loss_ae2.backward()
            if GRAD_CLIP is not None:
                nn.utils.clip_grad_norm_(
                    list(model.encoder.parameters()) + list(model.decoder2.parameters()),
                    GRAD_CLIP,
                )
            opt2.step()

        model.eval()
        with torch.no_grad():
            xh1_val, _, _ = model(Xval_t)
            val_loss = crit(Xval_t, xh1_val).item()

        if val_loss < best_val - 1e-6:
            best_val, best_state, bad = val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()
    return model


# ================================================================== #
#  ANOMALY SCORING
# ================================================================== #

@torch.no_grad()
def reconstruction_scores(model: USAD, X: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """A(x) = alpha * MSE(x, G1(E(x))) + (1-alpha) * MSE(x, G2(G1(E(x)))). Batched to avoid OOM."""
    model.eval()
    scores = []
    bs = 256
    mse = nn.MSELoss(reduction="none")

    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).float().to(DEVICE)
        xhat1, _, xhat2_adv = model(xb)
        err1 = mse(xb, xhat1).mean(dim=1)
        err2 = mse(xb, xhat2_adv).mean(dim=1)
        score = alpha * err1 + (1.0 - alpha) * err2
        scores.append(score.cpu().numpy().astype(np.float32))

    return np.concatenate(scores, axis=0)


# ================================================================== #
#  THRESHOLDS
# ================================================================== #

def build_thresholds(tr: np.ndarray) -> dict:
    """Data-driven thresholds from healthy training scores. Flag rule: score >= tau."""
    tr = np.asarray(tr, dtype=float)
    mu = float(tr.mean())
    sig = float(tr.std() + 1e-12)
    n = max(1, len(tr))
    se = sig / np.sqrt(n)

    tau = {}
    tau["mean_plus_2std"]  = mu + 2 * sig
    tau["mean_plus_3std"]  = mu + 3 * sig
    tau["mean_minus_3std"] = mu - 3 * sig   # lower-bound sanity check

    for p in (90, 95, 97, 98, 99):
        tau[f"p{p}"] = float(np.percentile(tr, p))

    tau["max"]           = float(tr.max())
    tau["max_minus_se"]  = float(tr.max() - se)
    tau["max_minus_2se"] = float(tr.max() - 2 * se)

    med = float(np.median(tr))
    mad = float(np.median(np.abs(tr - med)) + 1e-12)
    tau["median_plus_3mad"] = med + 3 * mad

    try:
        k2 = KMeans(n_clusters=2, n_init=10, random_state=SEED_GLOBAL).fit(tr.reshape(-1, 1))
        centers = np.sort(k2.cluster_centers_.flatten())
        tau["kmeans_mid"] = float(0.5 * (centers[0] + centers[1]))
    except Exception:
        pass

    try:
        tau["otsu"] = float(threshold_otsu(tr))
    except Exception:
        pass

    if USE_GPD:
        try:
            tail_n = max(10, int(GPD_TAIL_FRAC * len(tr)))
            tail_vals = np.sort(tr)[-tail_n:]
            shift = float(tail_vals.min())
            shape, loc, scale = genpareto.fit(tail_vals - shift, floc=0)
            for pct in GPD_PCTS:
                tau[f"gpd_tail{int(GPD_TAIL_FRAC * 100)}_{int(pct * 100)}"] = float(
                    shift + genpareto.ppf(pct, shape, loc, scale)
                )
        except Exception:
            pass

    return tau


# ================================================================== #
#  MEMMAP HELPERS
# ================================================================== #

def ensure_channel_dir(tag: str, stamp: str) -> str:
    d = os.path.join(OUTPUT_DIR, f"{tag}_{stamp}")
    os.makedirs(d, exist_ok=True)
    return d


def open_or_init_memmaps(chan_dir, tag, n_runs, n_train, n_test):
    """
    Open existing memmaps (crash-resume) or create fresh NaN-filled ones.
    Filenames include `tag` so multiple datasets can share OUTPUT_DIR
    without overwriting each other's scores.
    """
    tr_path = os.path.join(chan_dir, f"train_scores_runs_USAD_{tag}.npy")
    te_path = os.path.join(chan_dir, f"test_scores_runs_USAD_{tag}.npy")

    if os.path.exists(tr_path) and os.path.exists(te_path):
        mm_tr = open_memmap(tr_path, mode="r+")
        mm_te = open_memmap(te_path, mode="r+")
        start = mm_tr.shape[0]
        for r in range(mm_tr.shape[0]):
            if np.all(np.isnan(mm_tr[r])):
                start = r
                break
    else:
        mm_tr = open_memmap(tr_path, mode="w+", dtype="float32", shape=(n_runs, n_train))
        mm_te = open_memmap(te_path, mode="w+", dtype="float32", shape=(n_runs, n_test))
        mm_tr[:] = np.nan
        mm_te[:] = np.nan
        mm_tr.flush()
        mm_te.flush()
        start = 0

    return mm_tr, mm_te, start


# ================================================================== #
#  THRESHOLD SWEEP CSVs
# ================================================================== #

def _natural_order_indices(file_list):
    names = [os.path.basename(str(f)) for f in file_list]
    return np.array(sorted(range(len(names)), key=lambda i: _natural_key(names[i])), dtype=int)


def run_threshold_sweep(tag, stamp, te_ens, te_files, tr_ens):
    """
    Build all thresholds from ensemble training scores, apply to test scores,
    and write per-threshold CSVs plus an earliest-detection summary.
    """
    out_dir = os.path.join(OUTPUT_DIR, "threshold_results", tag)
    os.makedirs(out_dir, exist_ok=True)

    thresholds = build_thresholds(tr_ens)
    thr_names = sorted(thresholds.keys())
    nat_idx = _natural_order_indices(te_files)
    inv_nat = np.empty_like(nat_idx)
    inv_nat[nat_idx] = np.arange(len(nat_idx))

    master_csv = os.path.join(out_dir, f"{tag}_{stamp}_ALL_THRESHOLDS.csv")
    with open(master_csv, "w", newline="") as fm:
        wm = csv.writer(fm)
        wm.writerow(["filename", "score", "order_index"] + [f"flagged_{t}" for t in thr_names])
        for i in range(len(te_files)):
            row = [os.path.basename(str(te_files[i])), float(te_ens[i]), int(inv_nat[i])]
            row += [int(te_ens[i] >= (thresholds[t] - EPS)) for t in thr_names]
            wm.writerow(row)
    print(f"[saved] {master_csv}")

    counts_csv   = os.path.join(out_dir, f"{tag}_{stamp}_threshold_counts.csv")
    earliest_csv = os.path.join(out_dir, f"{tag}_{stamp}_earliest_and_after.csv")
    with open(counts_csv, "w", newline="") as fc, open(earliest_csv, "w", newline="") as fe:
        wc, we = csv.writer(fc), csv.writer(fe)
        wc.writerow(["threshold_name", "tau", "n_anomalous", "n_normal"])
        we.writerow([
            "threshold_name", "tau", "earliest_natural_index", "earliest_filename",
            "n_anom_after_inclusive", "n_norm_after_inclusive",
        ])

        for t in thr_names:
            tau = thresholds[t]
            flags = (te_ens >= (tau - EPS))

            thr_csv = os.path.join(out_dir, f"{tag}_{stamp}_{t}.csv")
            with open(thr_csv, "w", newline="") as ft:
                wt = csv.writer(ft)
                wt.writerow(["filename", "score", "flagged", "order_index"])
                for i in range(len(te_files)):
                    wt.writerow([
                        os.path.basename(str(te_files[i])), float(te_ens[i]),
                        int(flags[i]), int(inv_nat[i]),
                    ])

            n_anom, n_norm = int(flags.sum()), int((~flags).sum())
            wc.writerow([t, f"{tau:.6f}", n_anom, n_norm])
            print(f"  [{t}]  tau={tau:.6f}  anomalous={n_anom}  normal={n_norm}")

            if n_anom == 0:
                we.writerow([t, f"{tau:.6f}", "", "", 0, len(flags)])
                continue

            flags_nat = flags[nat_idx]
            first_hit = int(np.argmax(flags_nat))
            if not flags_nat[first_hit]:
                we.writerow([t, f"{tau:.6f}", "", "", 0, len(flags)])
                continue

            earliest_file = os.path.basename(str(te_files[nat_idx[first_hit]]))
            anom_after = int(flags_nat[first_hit:].sum())
            norm_after = int((~flags_nat[first_hit:]).sum())
            we.writerow([t, f"{tau:.6f}", first_hit, earliest_file, anom_after, norm_after])

    print(f"[saved] {counts_csv}")
    print(f"[saved] {earliest_csv}")
    return thresholds


# ================================================================== #
#  ENSEMBLE
# ================================================================== #

def ensemble_scores(train_runs: np.ndarray, test_runs: np.ndarray, zscore_per_run: bool = True):
    """Ensemble across N_RUNS by z-scoring each run on its own training scores, then averaging."""
    R = train_runs.shape[0]
    tr = train_runs.astype(float).copy()
    te = test_runs.astype(float).copy()

    if zscore_per_run:
        for r in range(R):
            mu = train_runs[r].mean()
            sd = train_runs[r].std() + 1e-12
            tr[r] = (train_runs[r] - mu) / sd
            te[r] = (test_runs[r] - mu) / sd

    return tr.mean(axis=0), te.mean(axis=0)


# ================================================================== #
#  MAIN
# ================================================================== #

def main():
    set_seed(SEED_GLOBAL)
    manifest = []

    for train_dir, test_dir, tag, mat_key in DATASET_PAIRS:
        stamp = tag.replace("_", "").replace("-", "")
        print(f"\n{'=' * 60}\n  Dataset: {tag}  |  mat_key='{mat_key}'\n{'=' * 60}")

        Xtr_raw, Ftr = load_folder_mat_1d(train_dir, mat_key, EXPECTED_LEN)
        Xte_raw, Fte = load_folder_mat_1d(test_dir, mat_key, EXPECTED_LEN)
        assert Xtr_raw.shape[1] == EXPECTED_LEN and Xte_raw.shape[1] == EXPECTED_LEN
        in_dim = EXPECTED_LEN

        Xtr, Xte, scaler_mean, scaler_scale = standardize_train_test(Xtr_raw, Xte_raw, mode="train")
        chan_dir = ensure_channel_dir(tag, stamp)

        meta = dict(
            tag=tag, stamp=stamp, mat_key=mat_key,
            train_dir=train_dir, test_dir=test_dir,
            n_train=int(Xtr.shape[0]), n_test=int(Xte.shape[0]),
            seeds=SEEDS, expected_len=EXPECTED_LEN,
            remove_dc=REMOVE_DC, use_scaler=USE_SCALER,
            scaler_mean=scaler_mean.tolist(), scaler_scale=scaler_scale.tolist(),
            usad_params=dict(
                hidden_dims=HIDDEN_DIMS, alpha=ALPHA, use_l1_loss=USE_L1_LOSS,
                lr=LR, weight_decay=WEIGHT_DECAY, epochs=EPOCHS,
                batch_size=BATCH_SIZE, patience=PATIENCE, val_split=VAL_SPLIT,
                max_samples=(len(Xtr) if MAX_SAMPLES is None else min(MAX_SAMPLES, len(Xtr))),
                grad_clip=GRAD_CLIP, device=str(DEVICE),
            ),
            train_files=[str(x) for x in Ftr],
            test_files=[str(x) for x in Fte],
        )
        write_json_safely(os.path.join(chan_dir, "meta.json"), meta)

        if PRINT_MODEL_SUMMARY_ONCE:
            tmp = USAD(in_dim, HIDDEN_DIMS).to(DEVICE)
            print_model_summary(tmp, in_dim=in_dim)
            del tmp

        mm_tr, mm_te, start_run = open_or_init_memmaps(chan_dir, tag, N_RUNS, Xtr.shape[0], Xte.shape[0])
        print(f"Memmaps: {mm_tr.shape} train, {mm_te.shape} test | resuming from run {start_run}/{N_RUNS}")

        prog_path = os.path.join(chan_dir, "progress.json")
        if os.path.exists(prog_path):
            prog = json.load(open(prog_path))
            start_run = max(start_run, int(prog.get("last_completed_run", -1)) + 1)

        for r in range(start_run, N_RUNS):
            seed = SEEDS[r]
            t0 = time.time()
            print(f"\n  Run {r + 1}/{N_RUNS}  (seed={seed})")

            model = train_one_run(Xtr, seed=seed, in_dim=in_dim)

            if SAVE_MODELS:
                ckpt = os.path.join(chan_dir, f"usad_{tag}_run{r:02d}_best.pt")
                torch.save(model.state_dict(), ckpt)

            mm_tr[r] = reconstruction_scores(model, Xtr, alpha=ALPHA)
            mm_te[r] = reconstruction_scores(model, Xte, alpha=ALPHA)
            mm_tr.flush()
            mm_te.flush()

            print(f"  Done in {time.time() - t0:.1f}s  |  "
                  f"train_max={mm_tr[r].max():.4f}  test_max={mm_te[r].max():.4f}")

            write_json_safely(prog_path, dict(last_completed_run=r, seeds=SEEDS))

        tr_ens, te_ens = ensemble_scores(np.array(mm_tr), np.array(mm_te), zscore_per_run=True)

        final_npz = os.path.join(chan_dir, f"usad_runs_{tag}_{stamp}.npz")
        np.savez_compressed(
            final_npz,
            train_scores_runs=np.array(mm_tr), test_scores_runs=np.array(mm_te),
            train_scores_ens=tr_ens.astype(np.float32), test_scores_ens=te_ens.astype(np.float32),
            train_files=Ftr, test_files=Fte,
            seeds=np.array(SEEDS, dtype=int),
            scaler_mean=scaler_mean, scaler_scale=scaler_scale,
            params=np.array(meta["usad_params"], dtype=object),
        )
        print(f"\n[saved] {final_npz}")

        print("\n--- Threshold sweep (ensemble scores) ---")
        run_threshold_sweep(tag, stamp, te_ens, Fte, tr_ens)

        manifest.append(dict(
            tag=tag, stamp=stamp, channel_dir=chan_dir, npz=final_npz,
            n_train=int(Xtr.shape[0]), n_test=int(Xte.shape[0]), seeds=SEEDS,
        ))

    man_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[done] Manifest: {man_path}")


if __name__ == "__main__":
    main()
