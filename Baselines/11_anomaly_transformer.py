"""
anomaly_transformer.py

Anomaly Transformer with a Conv1D stem (Wu et al., "Anomaly Transformer:
Time Series Anomaly Detection with Association Discrepancy", ICLR 2022)
for vibration-based machinery fault detection.

Default dataset: HUMS2023 (4095-point segments). IMS and XJTU-SY are also
supported (4096-point segments) -- see DATASET_PAIRS and EXPECTED_LEN below.
Differences between HUMS and the bearing datasets are marked inline with
"DATASET DIFFERENCE" comments; everything else is dataset-agnostic. Decoder
output paddings are derived automatically from EXPECTED_LEN at import time,
so no manual arithmetic is needed when switching datasets.

Architecture:
    ConvStem  : 1 -> 16 -> 32 -> 64, kernel=9, strides (3,3,5), padding=4
    Positions : fixed sinusoidal
    Blocks    : 2 x AnomalyTransformerBlock (d_model=64, heads=4, d_ff=256)
    Decoder   : ConvTranspose1d 64 -> 32 -> 16 -> 1 (strides reversed)
                + Conv1d(1->1, k=7) refinement pass

Training (two-phase minimax, mirrors USAD's two-optimiser structure):
    Phase 1 (log_sigma / prior scale only):
        L1 = recon_mse.detach() - lambda_disc * AssocDisc
    Phase 2 (all other params):
        L2 = recon_mse + lambda_disc * AssocDisc

Anomaly score:
    A(x) = alpha * recon_mse(x) + (1-alpha) * assoc_disc_per_sample(x)
    Higher score = more anomalous.

Requirements: torch, numpy, scipy, scikit-learn, scikit-image
"""

import os
import json
import glob
import time
import re
import random
import math
import csv

import numpy as np
from scipy.io import loadmat
from scipy.stats import genpareto
from numpy.lib.format import open_memmap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================================================== #
#  USER SETTINGS
# ================================================================== #

# Each entry: (train_dir, test_dir, tag, mat_key)
DATASET_PAIRS = [
    ("./data/HUMS2023/train", "./data/HUMS2023/test", "HUMS_RF2", "xr"),

    # DATASET DIFFERENCE: IMS and XJTU-SY are 4096-point segments, not 4095.
    # To run either, set EXPECTED_LEN = 4096 below and uncomment the entry.
    # ("./data/IMS/train", "./data/IMS/test", "IMS_Set1", "IMS"),
    # ("./data/XJTUSY/train", "./data/XJTUSY/test", "XJTUSY_31", "xjtusy"),
]

OUTPUT_DIR = "./atf_runs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- conv stem ----
STEM_CHANNELS = (1, 16, 32, 64)
STEM_KERNEL   = 9
STEM_STRIDES  = (3, 3, 5)
STEM_PADDING  = 4

# ---- transformer ----
D_MODEL  = 64
N_HEADS  = 4
N_LAYERS = 2
D_FF     = 256
DROPOUT  = 0.0

# ---- loss / scoring ----
LAMBDA_DISC = 3.0
ALPHA       = 0.5

# ---- training ----
LR           = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS       = 300
BATCH_SIZE   = 128
PATIENCE     = 10
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
# DATASET DIFFERENCE: 4095 for HUMS2023 (token count 91, decoder output
# paddings (4,2,2)). Set to 4096 for IMS / XJTU-SY (token count 92, output
# paddings (0,0,0) -- 4096 divides evenly by the total stride of 45).
# Both are computed automatically below; this constant is the only thing
# that needs to change.
EXPECTED_LEN = 4095
USE_SCALER   = False
REMOVE_DC    = False

PRINT_MODEL_SUMMARY_ONCE = True
SEED_GLOBAL = 0

assert D_MODEL == STEM_CHANNELS[-1]


# ================================================================== #
#  STEM ARITHMETIC
# ================================================================== #

def _forward_len(L: int, k: int, s: int, p: int) -> int:
    return (L + 2 * p - k) // s + 1


def _compute_seq_len(in_len: int) -> int:
    L = in_len
    for s in STEM_STRIDES:
        L = _forward_len(L, STEM_KERNEL, s, STEM_PADDING)
    return L


def _compute_output_paddings(in_len: int) -> tuple:
    """ConvTranspose1d output_padding values so the decoder reconstructs exactly in_len samples."""
    L0 = in_len
    L1 = _forward_len(L0, STEM_KERNEL, STEM_STRIDES[0], STEM_PADDING)
    L2 = _forward_len(L1, STEM_KERNEL, STEM_STRIDES[1], STEM_PADDING)
    L3 = _forward_len(L2, STEM_KERNEL, STEM_STRIDES[2], STEM_PADDING)

    def op(L_in, L_target, stride):
        natural = (L_in - 1) * stride - 2 * STEM_PADDING + STEM_KERNEL
        diff = L_target - natural
        assert 0 <= diff < stride, (
            f"output_padding {diff} out of range for L_in={L_in}, "
            f"L_target={L_target}, stride={stride}"
        )
        return diff

    return op(L3, L2, STEM_STRIDES[2]), op(L2, L1, STEM_STRIDES[1]), op(L1, L0, STEM_STRIDES[0])


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
#  POSITIONAL ENCODING
# ================================================================== #

class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1), :])


# ================================================================== #
#  ANOMALY ATTENTION
# ================================================================== #

class AnomalyAttention(nn.Module):
    """
    Series association S = softmax(QK^T / sqrt(d_k))
    Prior  association P = Gaussian(|i-j|; sigma_h) per head
    Association discrepancy = mean[KL(P||S) + KL(S||P)]
    log_sigma is updated only in phase 1 of the minimax training loop.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.log_sigma = nn.Parameter(torch.zeros(n_heads))
        self.dropout = nn.Dropout(dropout)

    def _gaussian_prior(self, L: int, device: torch.device) -> torch.Tensor:
        sigma = F.softplus(self.log_sigma)
        idx = torch.arange(L, dtype=torch.float, device=device)
        dist2 = (idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2
        log_P = -dist2.unsqueeze(0) / (2.0 * sigma.view(-1, 1, 1) ** 2 + 1e-8)
        return F.softmax(log_P, dim=-1)

    def forward(self, x: torch.Tensor):
        B, L, _ = x.shape

        def sh(t):
            return t.view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        Q, K, V = sh(self.W_q(x)), sh(self.W_k(x)), sh(self.W_v(x))
        S = self.dropout(F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k), dim=-1))
        P = self._gaussian_prior(L, x.device)
        out = self.W_o(torch.matmul(S, V).transpose(1, 2).contiguous().view(B, L, self.d_model))
        return out, S, P


# ================================================================== #
#  TRANSFORMER ENCODER BLOCK
# ================================================================== #

class AnomalyTransformerBlock(nn.Module):
    """Pre-norm: x = x + Attn(LN(x)); x = x + FFN(LN(x))."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.attn = AnomalyAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        residual = x
        attn_out, S, P = self.attn(self.norm1(x))
        x = residual + attn_out
        x = x + self.ff(self.norm2(x))
        return x, S, P


# ================================================================== #
#  CONV STEM / DECODER
# ================================================================== #

class ConvStem(nn.Module):
    """
    Three-layer Conv1D encoder stem. BatchNorm in the convolutional domain,
    LayerNorm in the transformer domain after transpose.
    Input (B, L) -> output (B, seq_len, d_model).
    """

    def __init__(self, channels=STEM_CHANNELS, kernel=STEM_KERNEL, strides=STEM_STRIDES, padding=STEM_PADDING):
        super().__init__()
        layers = []
        for i, s in enumerate(strides):
            c_in, c_out = channels[i], channels[i + 1]
            layers += [
                nn.Conv1d(c_in, c_out, kernel, stride=s, padding=padding, bias=False),
                nn.BatchNorm1d(c_out),
                nn.GELU(),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)          # (B, L) -> (B, 1, L)
        x = self.net(x)              # (B, d_model, seq_len)
        return x.transpose(1, 2)     # (B, seq_len, d_model)


class ConvDecoder(nn.Module):
    """
    Three-layer ConvTranspose1D decoder that exactly inverts ConvStem.
    output_padding values are computed at init time from EXPECTED_LEN so
    the decoder always reproduces the input length exactly -- no cropping.
    A final Conv1d(1->1, k=7) pass removes transposed-conv artefacts.
    """

    def __init__(self, output_paddings, channels=STEM_CHANNELS, kernel=STEM_KERNEL,
                 strides=STEM_STRIDES, padding=STEM_PADDING):
        super().__init__()
        ch = list(reversed(channels))
        st = list(reversed(strides))
        op = list(output_paddings)

        layers = []
        for i in range(len(st)):
            c_in, c_out = ch[i], ch[i + 1]
            layers.append(nn.ConvTranspose1d(
                c_in, c_out, kernel, stride=st[i], padding=padding, output_padding=op[i], bias=False,
            ))
            if i < len(st) - 1:
                layers += [nn.BatchNorm1d(c_out), nn.GELU()]
        layers.append(nn.Conv1d(1, 1, kernel_size=7, padding=3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)        # (B, seq_len, d_model) -> (B, d_model, seq_len)
        x = self.net(x)
        return x.squeeze(1)          # (B, L)


# ================================================================== #
#  FULL MODEL
# ================================================================== #

class AnomalyTransformer(nn.Module):
    """ConvStem -> PositionalEncoding -> N blocks -> ConvDecoder. in_len drives all stem arithmetic."""

    def __init__(self, in_len: int = EXPECTED_LEN, d_model: int = D_MODEL, n_heads: int = N_HEADS,
                 n_layers: int = N_LAYERS, d_ff: int = D_FF, dropout: float = DROPOUT):
        super().__init__()
        self.in_len = in_len
        self.seq_len = _compute_seq_len(in_len)
        op = _compute_output_paddings(in_len)

        self.stem = ConvStem()
        self.pos_enc = PositionalEncoding(d_model, max_len=self.seq_len + 10, dropout=dropout)
        self.blocks = nn.ModuleList([
            AnomalyTransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.decoder = ConvDecoder(output_paddings=op)

    def forward(self, x: torch.Tensor):
        z = self.pos_enc(self.stem(x))
        S_list, P_list = [], []
        for block in self.blocks:
            z, S, P = block(z)
            S_list.append(S)
            P_list.append(P)
        xhat = self.decoder(z)
        return xhat, S_list, P_list


def print_model_summary(model: AnomalyTransformer):
    x = torch.randn(2, model.in_len).to(DEVICE)
    print(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params: {total:,}  |  Trainable: {trainable:,}")
    model.eval()
    with torch.no_grad():
        xhat, S_list, _ = model(x)
    print(f"Input  (B, L) : (2, {model.in_len})")
    print(f"Tokens (B, L'): (2, {model.seq_len})")
    print(f"xhat   (B, L) : {tuple(xhat.shape)}")
    print(f"S shape       : {tuple(S_list[0].shape)}  (B, H, L', L')")


# ================================================================== #
#  ASSOCIATION DISCREPANCY
# ================================================================== #

def _discrepancy_per_sample(S_list: list, P_list: list) -> torch.Tensor:
    """Per-sample association discrepancy averaged over layers. Fully vectorised. Returns (B,)."""
    eps = 1e-8
    layer_discs = []
    for S, P in zip(S_list, P_list):
        P_exp = P.unsqueeze(0).expand_as(S)
        S_c = S.clamp(min=eps)
        P_c = P_exp.clamp(min=eps)
        kl_ps = (P_c * (P_c.log() - S_c.log())).sum(dim=-1)
        kl_sp = (S_c * (S_c.log() - P_c.log())).sum(dim=-1)
        layer_discs.append((kl_ps + kl_sp).mean(dim=(1, 2)))
    return torch.stack(layer_discs).mean(dim=0)


# ================================================================== #
#  TRAINING
# ================================================================== #

def train_one_run(Xtr: np.ndarray, seed: int) -> AnomalyTransformer:
    """
    Two-phase minimax training per batch, mirroring USAD's two-optimiser structure.
    Phase 1 updates only log_sigma (prior scale), with the reconstruction term
    detached so sigma responds to discrepancy pressure alone. Phase 2 updates
    everything else. Early stopping monitors validation reconstruction loss.
    """
    set_seed(seed)
    Xfit = subsample_rows(Xtr, MAX_SAMPLES, seed)
    n = len(Xfit)
    n_val = max(1, int(VAL_SPLIT * n))
    idx = np.random.permutation(n)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    Xtr_t = torch.from_numpy(Xfit[tr_idx]).float()
    Xval_t = torch.from_numpy(Xfit[val_idx]).float().to(DEVICE)

    dl_tr = DataLoader(
        TensorDataset(Xtr_t), batch_size=BATCH_SIZE, shuffle=True,
        drop_last=False, pin_memory=torch.cuda.is_available(), num_workers=0,
    )

    model = AnomalyTransformer(in_len=EXPECTED_LEN).to(DEVICE)

    sigma_params = [p for nm, p in model.named_parameters() if "log_sigma" in nm]
    other_params = [p for nm, p in model.named_parameters() if "log_sigma" not in nm]

    opt_sigma = torch.optim.Adam(sigma_params, lr=LR, weight_decay=WEIGHT_DECAY)
    opt_other = torch.optim.Adam(other_params, lr=LR, weight_decay=WEIGHT_DECAY)
    mse_crit = nn.MSELoss(reduction="mean")

    best_val, best_state, bad = float("inf"), None, 0

    for ep in range(1, EPOCHS + 1):
        model.train()

        for (xb,) in dl_tr:
            xb = xb.to(DEVICE, non_blocking=True)

            # ---- phase 1: maximise discrepancy, update sigma only ----
            xhat, S_list, P_list = model(xb)
            disc = _discrepancy_per_sample(S_list, P_list).mean()
            lrec = mse_crit(xb, xhat)
            loss1 = lrec.detach() - LAMBDA_DISC * disc
            opt_sigma.zero_grad(set_to_none=True)
            loss1.backward(retain_graph=True)
            if GRAD_CLIP:
                nn.utils.clip_grad_norm_(sigma_params, GRAD_CLIP)
            opt_sigma.step()

            # ---- phase 2: minimise reconstruction + disc, update everything else ----
            xhat, S_list, P_list = model(xb)
            disc = _discrepancy_per_sample(S_list, P_list).mean()
            lrec = mse_crit(xb, xhat)
            loss2 = lrec + LAMBDA_DISC * disc
            opt_other.zero_grad(set_to_none=True)
            loss2.backward()
            if GRAD_CLIP:
                nn.utils.clip_grad_norm_(other_params, GRAD_CLIP)
            opt_other.step()

        model.eval()
        with torch.no_grad():
            xhat_val, _, _ = model(Xval_t)
            val_loss = mse_crit(Xval_t, xhat_val).item()

        if val_loss < best_val - 1e-6:
            best_val, best_state, bad = val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= PATIENCE:
            break

    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()
    return model


# ================================================================== #
#  ANOMALY SCORING
# ================================================================== #

@torch.no_grad()
def reconstruction_scores(model: AnomalyTransformer, X: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """
    A(x) = alpha * recon_mse(x) + (1-alpha) * disc_per_sample(x).
    Batched at 64 (conservative, due to attention memory).
    """
    model.eval()
    mse = nn.MSELoss(reduction="none")
    scores = []

    for i in range(0, len(X), 64):
        xb = torch.from_numpy(X[i:i + 64]).float().to(DEVICE)
        xhat, S_list, P_list = model(xb)
        recon_err = mse(xb, xhat).mean(dim=1)
        disc = _discrepancy_per_sample(S_list, P_list)
        score = alpha * recon_err.cpu() + (1.0 - alpha) * disc.cpu()
        scores.append(score.numpy().astype(np.float32))

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
    tau["mean_minus_3std"] = mu - 3 * sig

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
    tr_path = os.path.join(chan_dir, f"train_scores_runs_ATF_{tag}.npy")
    te_path = os.path.join(chan_dir, f"test_scores_runs_ATF_{tag}.npy")

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
    seq_len_check = _compute_seq_len(EXPECTED_LEN)
    op_check = _compute_output_paddings(EXPECTED_LEN)
    print(f"\nStem arithmetic for EXPECTED_LEN={EXPECTED_LEN}:")
    print(f"  Token sequence length   : {seq_len_check}")
    print(f"  Decoder output_paddings : (stride-5 layer={op_check[0]}, "
          f"stride-3 layer={op_check[1]}, stride-3 layer={op_check[2]})")

    set_seed(SEED_GLOBAL)
    manifest = []

    for train_dir, test_dir, tag, mat_key in DATASET_PAIRS:
        stamp = tag.replace("_", "").replace("-", "")
        print(f"\n{'=' * 60}\n  Dataset: {tag}  |  mat_key='{mat_key}'\n{'=' * 60}")

        Xtr_raw, Ftr = load_folder_mat_1d(train_dir, mat_key, EXPECTED_LEN)
        Xte_raw, Fte = load_folder_mat_1d(test_dir, mat_key, EXPECTED_LEN)
        assert Xtr_raw.shape[1] == EXPECTED_LEN and Xte_raw.shape[1] == EXPECTED_LEN

        Xtr, Xte, scaler_mean, scaler_scale = standardize_train_test(Xtr_raw, Xte_raw, mode="train")
        chan_dir = ensure_channel_dir(tag, stamp)

        meta = dict(
            tag=tag, stamp=stamp, mat_key=mat_key,
            train_dir=train_dir, test_dir=test_dir,
            n_train=int(Xtr.shape[0]), n_test=int(Xte.shape[0]),
            seeds=SEEDS, expected_len=EXPECTED_LEN,
            remove_dc=REMOVE_DC, use_scaler=USE_SCALER,
            scaler_mean=scaler_mean.tolist(), scaler_scale=scaler_scale.tolist(),
            atf_params=dict(
                stem_channels=list(STEM_CHANNELS), stem_kernel=STEM_KERNEL,
                stem_strides=list(STEM_STRIDES), stem_padding=STEM_PADDING,
                seq_len=seq_len_check, decoder_output_paddings=list(op_check),
                d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                d_ff=D_FF, dropout=DROPOUT, lambda_disc=LAMBDA_DISC, alpha=ALPHA,
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
            tmp = AnomalyTransformer(in_len=EXPECTED_LEN).to(DEVICE)
            print_model_summary(tmp)
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

            model = train_one_run(Xtr, seed=seed)

            if SAVE_MODELS:
                ckpt = os.path.join(chan_dir, f"atf_{tag}_run{r:02d}_best.pt")
                torch.save(model.state_dict(), ckpt)

            mm_tr[r] = reconstruction_scores(model, Xtr, alpha=ALPHA)
            mm_te[r] = reconstruction_scores(model, Xte, alpha=ALPHA)
            mm_tr.flush()
            mm_te.flush()

            print(f"  Done in {time.time() - t0:.1f}s  |  "
                  f"train_max={mm_tr[r].max():.4f}  test_max={mm_te[r].max():.4f}")

            write_json_safely(prog_path, dict(last_completed_run=r, seeds=SEEDS))

        tr_ens, te_ens = ensemble_scores(np.array(mm_tr), np.array(mm_te), zscore_per_run=True)

        final_npz = os.path.join(chan_dir, f"atf_runs_{tag}_{stamp}.npz")
        np.savez_compressed(
            final_npz,
            train_scores_runs=np.array(mm_tr), test_scores_runs=np.array(mm_te),
            train_scores_ens=tr_ens.astype(np.float32), test_scores_ens=te_ens.astype(np.float32),
            train_files=Ftr, test_files=Fte,
            seeds=np.array(SEEDS, dtype=int),
            scaler_mean=scaler_mean, scaler_scale=scaler_scale,
            params=np.array(meta["atf_params"], dtype=object),
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
