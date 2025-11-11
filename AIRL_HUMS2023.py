"""
AIRL for Early Fault Detection on HUMS2023 (RF2 senosr)

This script implements an Adversarial Inverse Reinforcement Learning (AIRL) workflow
for early fault detection on the HUMS2023 dataset on sensor RF2 (timeseries vibration data).
It uses already-preprocessed 1D sequences saved as MATLAB `.mat` files (matfiles variable is 'xr')
split into expert (healthy up to Day 20) and test (from Day 21 onward) folders.

The pipeline:
  1) Load train/test sequences (natural filename order because they are timeseries and the filename are as per the time of data collection) from `.mat` files.
  2) Build scalar state–action–next-state transitions from consecutive samples.
  3) Define and train AIRL components, like Reward network, Value network etc.
  4) Score train/test samples with 1 − σ(f). Higher = “more anomalous”.
  5) Aggregate per-file scores (default: max) in natural order.
  6) Build multiple thresholds from train (quantiles, mean±kσ, MAD, KMeans midpoint,
     Otsu, and optional GPD tail fits) and apply them to TEST.
  7) Save:
       - train/test per-file scores (`.npy`)
       - Threshold sweep outputs & earliest-detection summaries (`.csv`)
       - Optional visualizations (`.png`) and compact bundles (`.npz`)

Data Info:
- Preprocessed HUMS2023 files in two directories. If you want to use the original dataset, the link and the preprocessing techniques are also described in Readme file.
    DATA_DIR_TRAIN = "./data/AIRL/training_until20"
    DATA_DIR_TEST  = "./data/AIRL/testAIRL_from21"
- Each `.mat` file contains a 1D array named **`xr`** (float or convertible to float).


Key Outputs:
- train_anomaly_scores_alreadystndzHUMSRF2.npy
- test_anomaly_scores_alreadystndzHUMSRF2.npy
- airl_threshold_results_HUMS/ALL_THRESHOLDS.csv
- airl_threshold_results_HUMS/earliest_and_after.csv
(plus optional visualizations)

"""
import os, glob, csv, json, time, re
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu
from scipy.stats import genpareto


SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data folder
DATA_DIR_TRAIN = r"./data/AIRL/training_until20"
DATA_DIR_TEST  = r"./data/AIRL/testAIRL_from21"

# Threshold config
USE_GPD       = True
GPD_TAIL_FRAC = 0.05
GPD_PCTS      = [0.95, 0.98, 0.99]
DEFAULT_THRESHOLD = "kmeans_mid"  
EPS = 1e-12                     


_num_re = re.compile(r'(\d+)')

def natural_key(s):
    s = os.path.basename(str(s))
    return [int(t) if t.isdigit() else t.lower() for t in _num_re.split(s)]


def load_data_mat_1d(folder_path, varname='xr'):
    files = sorted(glob.glob(os.path.join(folder_path, '*.mat')), key=natural_key)
    seqs, fnames = [], []
    for file in files:
        mat = loadmat(file)
        if varname in mat:
            x = mat[varname].astype(np.float32).flatten()
            seqs.append(x)
            fnames.append(file)
    return seqs, fnames

def save_scores(npy_fname, file_list, score_list):
    arr = np.empty(len(file_list), dtype=object)
    arr[:] = list(zip(file_list, score_list))
    np.save(npy_fname, arr, allow_pickle=True)


train_seqs, train_files = load_data_mat_1d(DATA_DIR_TRAIN, varname='xr')
test_seqs,  test_files  = load_data_mat_1d(DATA_DIR_TEST,  varname='xr')

if len(train_seqs) == 0 or len(test_seqs) == 0:
    raise RuntimeError("No .mat files found. Check DATA_DIR_TRAIN / DATA_DIR_TEST and 'xr' var name.")


def create_transitions(seqs):
    s, a, sp = [], [], []
    for seq in seqs:
        if len(seq) < 2:
            continue
        # Consecutive scalar (s, a=next value, sp=next value) tuples
        s.extend(seq[:-1])
        a.extend(seq[1:])
        sp.extend(seq[1:])
    s  = torch.tensor(s,  dtype=torch.float32, device=device).unsqueeze(-1)
    a  = torch.tensor(a,  dtype=torch.float32, device=device).unsqueeze(-1)
    sp = torch.tensor(sp, dtype=torch.float32, device=device).unsqueeze(-1)
    return s, a, sp

s_train, a_train, sp_train = create_transitions(train_seqs)

# AIRL Framework 
class RewardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=1))

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, s):
        return self.net(s)

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc      = nn.Sequential(nn.Linear(1, 64), nn.ReLU())
        self.mean    = nn.Linear(64, 1)
        self.log_std = nn.Parameter(torch.ones(1) * -1.0)
    def forward(self, s):
        x = self.fc(s)
        mu  = self.mean(x)
        std = torch.clamp(self.log_std.exp(), min=1e-3)
        return mu, std
    def sample(self, s):
        mu, std = self.forward(s)
        dist = Normal(mu, std)
        a = dist.rsample()
        logp = dist.log_prob(a)
        return a, logp

reward_net = RewardNet().to(device)
value_net  = ValueNet().to(device)
policy_net = PolicyNet().to(device)

opt_disc   = optim.Adam(list(reward_net.parameters()) + list(value_net.parameters()), lr=1e-3)
opt_policy = optim.Adam(policy_net.parameters(), lr=1e-3)


gamma              = 0.99
beta               = 0.01
batch_size         = 128
episodes_per_epoch = 50
steps_per_episode  = 50
num_epochs         = 300


for epoch in range(num_epochs):
    # Discriminator
    idx = torch.randint(0, len(s_train), (batch_size,), device=s_train.device)
    s_exp, a_exp, sp_exp = s_train[idx], a_train[idx], sp_train[idx]
    s_fake = s_train[idx]
    a_fake, _ = policy_net.sample(s_fake)
    sp_fake   = a_fake

    r_exp  = reward_net(s_exp, a_exp)
    r_fake = reward_net(s_fake, a_fake)
    V_se, V_spe  = value_net(s_exp),  value_net(sp_exp)
    V_sf, V_spf  = value_net(s_fake), value_net(sp_fake)

    D_exp  = torch.sigmoid(r_exp  + gamma * V_spe - V_se)
    D_fake = torch.sigmoid(r_fake + gamma * V_spf - V_sf)

    loss_disc = - (torch.log(D_exp + 1e-9).mean() + torch.log(1 - D_fake + 1e-9).mean())
    opt_disc.zero_grad()
    loss_disc.backward()
    opt_disc.step()

    # Policy (REINFORCE + entropy)
    policy_loss_total = 0.0
    for _ in range(episodes_per_epoch):
        idx0 = torch.randint(0, len(s_train), (1,), device=s_train.device)
        s    = s_train[idx0]

        logps, rewards, entropies = [], [], []
        for _ in range(steps_per_episode):
            a, logp = policy_net.sample(s)
            mu, std = policy_net.forward(s)
            dist = Normal(mu, std)
            entropies.append(dist.entropy().mean())

            sp = a
            with torch.no_grad():
                r = reward_net(s, a)

            logps.append(logp.squeeze(-1))
            rewards.append(r.squeeze(-1))

            s = sp.detach()

        # Return normalization
        returns = []
        G = torch.zeros(1, device=device)
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.stack(returns).squeeze(-1)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        logps = torch.stack(logps).squeeze(-1)
        ent   = torch.stack(entropies).mean()

        policy_loss_ep = -(logps * returns.detach()).mean() - beta * ent
        policy_loss_total += policy_loss_ep

    policy_loss_total /= episodes_per_epoch
    opt_policy.zero_grad()
    policy_loss_total.backward()
    opt_policy.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Disc={loss_disc.item():.4f} | log_std={policy_net.log_std.item():+.3f}")
      
# Threshold helper
def build_thresholds(tr: np.ndarray):
    tr = np.asarray(tr, float)
    if tr.size == 0:
        return {"p99": 0.0}

    mu    = float(tr.mean())
    sigma = float(tr.std() + 1e-12)
    n     = max(1, int(len(tr)))
    se    = sigma / np.sqrt(n)

    thresholds = {}
    thresholds['mean_plus_2std']   = mu + 2 * sigma
    thresholds['mean_plus_3std']   = mu + 3 * sigma
    thresholds['mean_minus_3std']  = mu - 3 * sigma

    for p in (90, 95, 97, 98, 99):
        thresholds[f"p{p}"] = float(np.percentile(tr, p))

    thresholds['max']           = float(tr.max())
    thresholds['max_minus_se']  = float(tr.max() - se)
    thresholds['max_minus_2se'] = float(tr.max() - 2 * se)

    med = float(np.median(tr))
    mad = float(np.median(np.abs(tr - med)) + 1e-12)
    thresholds['median_plus_3mad'] = med + 3*mad

    # KMeans midpoint
    try:
        k2 = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit(tr.reshape(-1,1))
        centers = np.sort(k2.cluster_centers_.flatten())
        thresholds['kmeans_mid'] = float(0.5*(centers[0]+centers[1]))
    except Exception:
        pass

    # Otsu
    try:
        thresholds['otsu'] = float(threshold_otsu(tr))
    except Exception:
        pass

    # GPD tail
    if USE_GPD:
        try:
            tail_n    = max(10, int(GPD_TAIL_FRAC * len(tr)))
            tail_vals = np.sort(tr)[-tail_n:]
            shift     = float(tail_vals.min())
            shape, loc, scale = genpareto.fit(tail_vals - shift, floc=0)
            for pct in GPD_PCTS:
                thresholds[f"gpd_tail{int(GPD_TAIL_FRAC*100)}_{int(pct*100)}"] = float(
                    shift + genpareto.ppf(pct, shape, loc, scale)
                )
        except Exception:
            pass
    return thresholds

# Build scores & thresholds (TRAIN) 
with torch.no_grad():
    f_train   = reward_net(s_train, a_train) + gamma * value_net(sp_train) - value_net(s_train)
    D_train   = torch.sigmoid(f_train)
    anom_train = (1.0 - D_train).detach().cpu().numpy().astype(np.float32)

# Aggregate per-file TRAIN scores (max) — in NATURAL ORDER of train_files
train_file_scores = []
ptr = 0
for seq in train_seqs:
    n = len(seq) - 1
    seg = anom_train[ptr:ptr+n]
    score = float(seg.max()) if n > 0 else 0.0
    train_file_scores.append(score)
    ptr += n

save_scores("train_anomaly_scores_alreadystndzHUMSRF2.npy", train_files, train_file_scores)

# Prefer building thresholds on per-file TRAIN scores
thresholds = build_thresholds(np.asarray(train_file_scores, float))

# Score TEST per-file
test_scores = []
for seq, fname in zip(test_seqs, test_files):
    s, a, sp = create_transitions([seq])
    with torch.no_grad():
        f_test = reward_net(s, a) + gamma * value_net(sp) - value_net(s)
        D_test = torch.sigmoid(f_test)
        anom_seq = (1.0 - D_test).detach().cpu().numpy().astype(np.float32)
        anom_score = float(np.max(anom_seq)) if len(anom_seq) else 0.0
    test_scores.append((fname, anom_score))

save_scores("test_anomaly_scores_alreadystndzHUMSRF2.npy",
            [f for f, _ in test_scores],
            [s for _, s in test_scores])

THR_OUT_ROOT = "./airl_threshold_results_HUMS"
os.makedirs(THR_OUT_ROOT, exist_ok=True)

# Natural-order arrays for convenience
te_files  = np.array([f for f, _ in test_scores], dtype=object)   
te_scores = np.array([s for _, s in test_scores], dtype=float)   

# Master CSV (all thresholds)
thr_names = sorted(thresholds.keys())
master_csv = os.path.join(THR_OUT_ROOT, "ALL_THRESHOLDS.csv")
with open(master_csv, "w", newline="") as fm:
    wm = csv.writer(fm)
    wm.writerow(["filename", "score", "order_index", "day"] + [f"flagged_{t}" for t in thr_names])
    # 'day' left blank (no Day() parsing here since we use filename natural order)
    for i in range(len(te_files)):
        row = [
            os.path.basename(str(te_files[i])),
            float(te_scores[i]),
            int(i),
            ""  # day not used; keeping column for compatibility
        ]
        for t in thr_names:
            tau = thresholds[t]
            row.append(int(te_scores[i] >= (tau - EPS)))
        wm.writerow(row)
print(f"[saved] {master_csv}")

# counts + earliest-and-after + per-threshold CSVs (NATURAL ORDER)
counts_csv   = os.path.join(THR_OUT_ROOT, "threshold_counts.csv")
earliest_csv = os.path.join(THR_OUT_ROOT, "earliest_and_after.csv")
with open(counts_csv, "w", newline="") as fc, open(earliest_csv, "w", newline="") as fe:
    wc = csv.writer(fc); we = csv.writer(fe)
    wc.writerow(["threshold_name", "tau", "n_anomalous", "n_normal"])
    we.writerow(["threshold_name", "tau",
                 "earliest_index", "earliest_day", "earliest_filename",
                 "n_anom_after_inclusive", "n_norm_after_inclusive"])

    for t in thr_names:
        tau   = thresholds[t]
        flags = (te_scores >= (tau - EPS))  # NATURAL ORDER flags

        # per-threshold CSV
        thr_csv = os.path.join(THR_OUT_ROOT, f"{t}.csv")
        with open(thr_csv, "w", newline="") as ft:
            wt = csv.writer(ft)
            wt.writerow(["filename", "score", "flagged", "order_index", "day"])
            for i in range(len(te_files)):
                wt.writerow([
                    os.path.basename(str(te_files[i])),
                    float(te_scores[i]),
                    int(flags[i]),
                    int(i),
                    ""  # day empty
                ])
        print(f"[saved] {thr_csv}")

        # totals
        n_anom = int(flags.sum())
        n_norm = int((~flags).sum())
        wc.writerow([t, f"{tau:.6f}", n_anom, n_norm])
        print(f"[{t}] τ={tau:.6f} | anomalies={n_anom} | normals={n_norm}")

        # earliest + after (by NATURAL ORDER)
        if n_anom == 0:
            we.writerow([t, f"{tau:.6f}", "", "", "", 0, len(flags)])
            continue
        first_hit_idx = int(np.argmax(flags))  # first True in natural order
        earliest_file = os.path.basename(str(te_files[first_hit_idx]))
        anom_after = int(flags[first_hit_idx:].sum())
        norm_after = int((~flags[first_hit_idx:]).sum())
        we.writerow([t, f"{tau:.6f}",
                     first_hit_idx,
                     "",  # no day
                     earliest_file,
                     anom_after,
                     norm_after])

print(f"[saved] {counts_csv}")
print(f"[saved] {earliest_csv}")

# Apply thresholds & print 
for method, thresh in thresholds.items():
    print(f"Threshold ({method}): {thresh:.6f}")
    flags = (te_scores >= (thresh - EPS))
    if flags.any():
        first_idx = int(np.argmax(flags))
        print("Earliest Anomalous File:", os.path.basename(str(te_files[first_idx])))
    else:
        print("No anomalies detected.")

#  Top-ranked TEST files by score (descending by score)
print("\nTop-ranked TEST files by score:")
for fname, score in sorted(test_scores, key=lambda x: x[1], reverse=True)[:10]:
    print(f"{os.path.basename(str(fname))}: {score:.4f}")

# 
OUT_ROOT  = "./airl_alreadystndzscores_"
os.makedirs(OUT_ROOT, exist_ok=True)
STAMP     = 'HUMSRF2'

# Visualization aggregation knobs (for alt outputs)
FILE_AGG_METHOD = 'topk_mean'   # 'max' | 'topk_mean' | 'topk_median'
TOPK_FRAC       = 0.05
TOPK_KMIN       = 5
TOPK_KMAX       = None
USE_ROLLING     = False
ROLL_WIN        = 256
ROLL_TAKE       = 'max'         # 'max' | 'mean'

def topk_mean(arr, k):
    arr = np.asarray(arr)
    if len(arr) == 0: return 0.0
    if k <= 0: return float(np.max(arr))
    return float(np.mean(np.sort(arr)[-k:]))

def topk_median(arr, k):
    arr = np.asarray(arr)
    if len(arr) == 0: return 0.0
    if k <= 0: return float(np.max(arr))
    return float(np.median(np.sort(arr)[-k:]))

def rolling_topk_mean(arr, win=256, kfrac=0.05, kmin=5):
    arr = np.asarray(arr)
    out = []
    for i in range(win, len(arr)+1):
        w = arr[i-win:i]
        k = max(kmin, int(kfrac * len(w)))
        out.append(np.mean(np.sort(w)[-k:]))
    return np.array(out)

def aggregate_file_scores(scores_1d, method='topk_mean', kfrac=0.05, kmin=5, kmax=None,
                          use_rolling=False, roll_win=256, roll_take='max'):
    scores_1d = np.asarray(scores_1d)
    if len(scores_1d) == 0: return 0.0
    if use_rolling:
        roll = rolling_topk_mean(scores_1d, win=roll_win, kfrac=kfrac, kmin=kmin)
        if len(roll) == 0:
            use_rolling = False
        else:
            return float(np.max(roll) if roll_take == 'max' else np.mean(roll))
    if method == 'max':
        return float(np.max(scores_1d))
    k = max(kmin, int(kfrac * len(scores_1d)))
    if kmax is not None:
        k = min(k, kmax)
    if method == 'topk_mean':
        return topk_mean(scores_1d, k)
    elif method == 'topk_median':
        return topk_median(scores_1d, k)
    else:
        raise ValueError(f"Unknown FILE_AGG_METHOD: {method}")

# Transition-level train scores already computed: anom_train
anom_train_1d = np.asarray(anom_train, dtype=np.float32)

# Per-file TRAIN (max) and alt aggregation (NATURAL ORDER)
train_file_scores_max = []
train_file_scores_alt = []
ptr = 0
for seq in train_seqs:
    n = len(seq) - 1
    seg = anom_train_1d[ptr:ptr+n]
    train_file_scores_max.append(float(np.max(seg)) if n > 0 else 0.0)
    train_file_scores_alt.append(aggregate_file_scores(
        seg, method=FILE_AGG_METHOD, kfrac=TOPK_FRAC, kmin=TOPK_KMIN, kmax=TOPK_KMAX,
        use_rolling=USE_ROLLING, roll_win=ROLL_WIN, roll_take=ROLL_TAKE
    ) if n > 0 else 0.0)
    ptr += n

# TEST per-file (recompute sequences for alt aggregation) — NATURAL ORDER
test_seq_scores = []
test_file_scores_max = []
test_file_scores_alt = []
with torch.no_grad():
    for seq in test_seqs:
        s, a, sp = create_transitions([seq])
        f_test = reward_net(s, a) + gamma * value_net(sp) - value_net(s)
        D_test = torch.sigmoid(f_test)
        anom_seq = (1.0 - D_test).detach().cpu().numpy().astype(np.float32)
        test_seq_scores.append(anom_seq)
        test_file_scores_max.append(float(np.max(anom_seq)) if len(anom_seq) else 0.0)
        test_file_scores_alt.append(aggregate_file_scores(
            anom_seq, method=FILE_AGG_METHOD, kfrac=TOPK_FRAC, kmin=TOPK_KMIN, kmax=TOPK_KMAX,
            use_rolling=USE_ROLLING, roll_win=ROLL_WIN, roll_take=ROLL_TAKE
        ) if len(anom_seq) else 0.0)

train_file_scores_max = np.array(train_file_scores_max, dtype=np.float32)
test_file_scores_max  = np.array(test_file_scores_max,  dtype=np.float32)
train_file_scores_alt = np.array(train_file_scores_alt, dtype=np.float32)
test_file_scores_alt  = np.array(test_file_scores_alt,  dtype=np.float32)

# Thresholds from TRAIN transitions (or switch to file-level by passing train_file_scores_max)
thresholds_ = build_thresholds(anom_train_1d)
tau = thresholds_.get(DEFAULT_THRESHOLD, thresholds_.get("p99", np.percentile(anom_train_1d, 99.0)))
tau_name = DEFAULT_THRESHOLD if DEFAULT_THRESHOLD in thresholds_ else ("p99" if "p99" in thresholds_ else "custom")

# Save compact bundle
OUT_ROOT = "./airl_alreadystndzscores_"
chan_dir = os.path.join(OUT_ROOT, f"_HUMSRF2{STAMP}")
os.makedirs(chan_dir, exist_ok=True)

np.savez_compressed(
    os.path.join(chan_dir, f"airl_scores__HUMSRF2{STAMP}.npz"),
    train_file_scores_max=train_file_scores_max,
    test_file_scores_max=test_file_scores_max,
    train_file_scores_alt=train_file_scores_alt,
    test_file_scores_alt=test_file_scores_alt,
    train_transition_scores=anom_train_1d,
    test_seq_scores=np.array(test_seq_scores, dtype=object),
    train_files=np.array(train_files, dtype=object),
    test_files=np.array(test_files, dtype=object),
    params=dict(
        file_agg_method=FILE_AGG_METHOD, topk_frac=TOPK_FRAC, topk_kmin=TOPK_KMIN, topk_kmax=TOPK_KMAX,
        use_rolling=USE_ROLLING, roll_win=ROLL_WIN, roll_take=ROLL_TAKE
    ),
    standardization=dict(mean=None, std=None),
)
with open(os.path.join(chan_dir, "meta.json"), "w") as f:
    json.dump(dict(stamp=STAMP, default_threshold=tau_name, tau=float(tau)), f, indent=2)

# Visualizations (per-file max)
plt.figure(figsize=(8,4))
plt.hist(train_file_scores_max, bins=50, alpha=0.6, label="TRAIN files (max)", density=True)
plt.hist(test_file_scores_max,  bins=50, alpha=0.6, label="TEST files (max)",  density=True)
plt.axvline(tau, linestyle="--", label=f"τ ({tau_name}) = {tau:.3f}")
plt.title(" | Per-file scores (max)")
plt.xlabel("score (higher = more anomalous)"); plt.ylabel("density"); plt.legend()
plt.tight_layout()
out_hist = os.path.join(chan_dir, "_HUMSRF2hist_max.png")
plt.savefig(out_hist, dpi=150); plt.close()

xs_tr = np.sort(train_file_scores_max); ys_tr = np.linspace(0,1,len(xs_tr),endpoint=True)
xs_te = np.sort(test_file_scores_max);  ys_te = np.linspace(0,1,len(xs_te),endpoint=True)
plt.figure(figsize=(8,4))
plt.plot(xs_tr, ys_tr, label="TRAIN (per-file, max) CDF")
plt.plot(xs_te, ys_te, label="TEST (per-file, max) CDF")
plt.axvline(tau, linestyle="--", label=f"τ ({tau_name}) = {tau:.3f}")
plt.title(" | CDFs (per-file, max)"); plt.xlabel("score"); plt.ylabel("F(x)"); plt.legend()
plt.tight_layout()
out_cdf = os.path.join(chan_dir, "_HUMSRF2cdf_max.png")
plt.savefig(out_cdf, dpi=150); plt.close()

print(f"[saved] {out_hist}")
print(f"[saved] {out_cdf}")

# Threshold sweep table on TEST (per-file max) — NATURAL ORDER not needed for counts
out_csv = os.path.join(chan_dir, "_alreadystndzHUMSRF2threshold_sweep_max.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["threshold_name","tau","n_anomalous","n_normal"])
    for name, t in thresholds_.items():
        flags = (test_file_scores_max >= (t - EPS))
        w.writerow([name, f"{t:.6f}", int(flags.sum()), int((~flags).sum())])
print(f"[saved] {out_csv}")

# Apply thresholds & print
for method, thresh in thresholds.items():
    flags = (te_scores >= (thresh - EPS))
    n_anom = int(flags.sum())
    n_norm = int((~flags).sum())
    print(f"Threshold ({method}): {thresh:.6f} | Anomalous: {n_anom} | Normal: {n_norm}")
    if n_anom > 0:
        earliest_idx = int(np.argmax(flags))
        print("  Earliest Anomalous File:", os.path.basename(str(te_files[earliest_idx])))
    else:
        print("  No anomalies detected.")

