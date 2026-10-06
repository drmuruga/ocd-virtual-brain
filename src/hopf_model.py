"""
Whole-brain Hopf model utilities for the OCD virtual-brain project.

Model: Deco-style Stuart-Landau (Hopf) oscillators on a structural connectome,
94 AAL2 cerebral regions (cortical + subcortical) from the HCP data in neurolib.
Healthy reference: G = 2.0, a = -0.02, f = 0.05 Hz, noise beta = 0.02.

Usage in Colab:
    import sys; sys.path.append("/content/ocd-virtual-brain/src")
    from hopf_model import *
"""
import numpy as np

# ---------------------------------------------------------------------------
# Atlas: AAL2 regions 1-94 (cerebellum excluded), ordered Left, Right, Left, ...
# ---------------------------------------------------------------------------
AAL2_BASE = [
    "Precentral", "Frontal_Sup_2", "Frontal_Mid_2", "Frontal_Inf_Oper", "Frontal_Inf_Tri",
    "Frontal_Inf_Orb_2", "Rolandic_Oper", "Supp_Motor_Area", "Olfactory", "Frontal_Sup_Medial",
    "Frontal_Med_Orb", "Rectus", "OFCmed", "OFCant", "OFCpost", "OFClat", "Insula",
    "Cingulate_Ant", "Cingulate_Mid", "Cingulate_Post", "Hippocampus", "ParaHippocampal",
    "Amygdala", "Calcarine", "Cuneus", "Lingual", "Occipital_Sup", "Occipital_Mid",
    "Occipital_Inf", "Fusiform", "Postcentral", "Parietal_Sup", "Parietal_Inf",
    "SupraMarginal", "Angular", "Precuneus", "Paracentral_Lobule", "Caudate", "Putamen",
    "Pallidum", "Thalamus", "Heschl", "Temporal_Sup", "Temporal_Pole_Sup", "Temporal_Mid",
    "Temporal_Pole_Mid", "Temporal_Inf",
]
LABELS = [f"{b}_{h}" for b in AAL2_BASE for h in "LR"]
N_REGIONS = len(LABELS)  # 94


def region_pair(base_name):
    """Indices [left, right] of an AAL2 region, e.g. region_pair('Caudate') -> [74, 75]."""
    return [LABELS.index(f"{base_name}_L"), LABELS.index(f"{base_name}_R")]


OFC = sum([region_pair(r) for r in ["OFCmed", "OFCant", "OFCpost", "OFClat"]], [])
ACC = region_pair("Cingulate_Ant")
CORTEX = OFC + ACC                      # cortical part of the OCD loop
STRIATUM = region_pair("Caudate")
THALAMUS = region_pair("Thalamus")
LOOP_NODES = sorted(CORTEX + STRIATUM + THALAMUS)
SEGMENTS = {
    "OFC-Caud": (CORTEX, STRIATUM),
    "Caud-Thal": (STRIATUM, THALAMUS),
    "Thal-OFC": (THALAMUS, CORTEX),
}
IU = np.triu_indices(N_REGIONS, k=1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_brain(max_weight=0.2):
    """Return (C, real_fc): normalised 94-region connectome and group-mean empirical FC."""
    from neurolib.utils.loadData import Dataset
    ds = Dataset("hcp", subcortical=True)
    C = np.array(ds.Cmat, dtype=float)
    np.fill_diagonal(C, 0)
    C = C / C.max() * max_weight
    real_fc = np.mean(ds.FCs, axis=0)
    return C, real_fc


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def simulate_hopf(C, G, a=-0.02, f=0.05, beta=0.02, dt=0.1,
                  T=900, TR=2.0, transient=60, seed=0):
    """Simulate the whole-brain Hopf model. `a` may be a scalar or a per-region array.
    Returns the simulated BOLD-like signal, shape (regions, time points)."""
    rng = np.random.default_rng(seed)
    n = C.shape[0]
    omega = 2 * np.pi * f
    strength = C.sum(axis=1)
    x = 0.1 * rng.standard_normal(n)
    y = 0.1 * rng.standard_normal(n)
    save_every = int(round(TR / dt))
    n_steps = int((T + transient) / dt)
    out = []
    for t in range(n_steps):
        r2 = x * x + y * y
        dx = (a - r2) * x - omega * y + G * (C @ x - strength * x)
        dy = (a - r2) * y + omega * x + G * (C @ y - strength * y)
        x = x + dt * dx + np.sqrt(dt) * beta * rng.standard_normal(n)
        y = y + dt * dy + np.sqrt(dt) * beta * rng.standard_normal(n)
        if t * dt >= transient and t % save_every == 0:
            out.append(x.copy())
    return np.array(out).T


def fc_per_seed(C, G=2.0, a=-0.02, seeds=range(10), **kw):
    """Simulated FC for each seed, shape (n_seeds, regions, regions)."""
    return np.array([np.corrcoef(simulate_hopf(C, G, a=a, seed=s, **kw)) for s in seeds])


def fc_similarity(A, B):
    """Pearson correlation of the upper triangles of two FC matrices."""
    return np.corrcoef(A[IU], B[IU])[0, 1]


# ---------------------------------------------------------------------------
# OCD perturbation and treatments
# ---------------------------------------------------------------------------
def pair_mask(A, B, n=N_REGIONS):
    m = np.zeros((n, n), bool)
    m[np.ix_(A, B)] = True
    m[np.ix_(B, A)] = True
    return m


def make_ocd(C, factor=2.0):
    """Strengthen all existing OFC/ACC-caudate-thalamus connections. Returns (C_ocd, loop_mask)."""
    loop_mask = np.zeros_like(C, dtype=bool)
    for A, B in SEGMENTS.values():
        loop_mask |= pair_mask(A, B)
    loop_mask &= C > 0
    C_ocd = C.copy()
    C_ocd[loop_mask] *= factor
    return C_ocd, loop_mask


def treatment(C_ocd, targets=(), a_treat=-0.02, a_base=-0.02, cut=None, cut_frac=1.0):
    """Build (C, a_vector) for a treatment: damp `targets` (DBS/TMS-like) and/or cut fibres."""
    a_vec = np.full(C_ocd.shape[0], a_base)
    a_vec[list(targets)] = a_treat
    Cm = C_ocd.copy()
    if cut is not None:
        A, B = cut
        Cm[np.ix_(A, B)] *= (1 - cut_frac)
        Cm[np.ix_(B, A)] *= (1 - cut_frac)
    return Cm, a_vec


def strength_matched_sham(C, target, exclude=(), n_options=1):
    """Find left/right region pair(s) outside the loop whose total wiring strength
    best matches `target` (a left/right pair). Returns a list of [L, R] index pairs."""
    strength = C.sum(axis=1)
    goal = strength[target].sum()
    banned = set(LOOP_NODES) | set(target) | set(exclude)
    candidates = []
    for k in range(0, N_REGIONS, 2):
        pair = [k, k + 1]
        if banned & set(pair):
            continue
        candidates.append((abs(strength[pair].sum() - goal), pair))
    candidates.sort(key=lambda c: c[0])
    return [p for _, p in candidates[:n_options]]


# ---------------------------------------------------------------------------
# Outcome measures
# ---------------------------------------------------------------------------
def seg_mean(fc, A, B):
    """Mean FC between region sets A and B (works on (N,N) or (seeds,N,N))."""
    return fc[..., A, :][..., :, B].mean(axis=(-2, -1))


def restoration(fc_t, fc_o, fc_h, A, B):
    """Per-seed signed restoration (%) of segment A-B.
    100 = back to healthy, 0 = unchanged OCD, >100 = over-correction, <0 = worse.
    Inputs are per-seed FC arrays (seeds, N, N) with matched seeds."""
    shift = seg_mean(fc_o, A, B).mean() - seg_mean(fc_h, A, B).mean()
    return 100 * (seg_mean(fc_o, A, B) - seg_mean(fc_t, A, B)) / shift


def off_target(fc_t, fc_h, treated=()):
    """Per-seed mean |FC change vs healthy| on connections outside the loop,
    excluding the treated regions' own connections."""
    keep = np.ones((N_REGIONS, N_REGIONS), bool)
    keep[np.ix_(LOOP_NODES, LOOP_NODES)] = False
    keep[list(treated), :] = False
    keep[:, list(treated)] = False
    keep = np.triu(keep, 1)
    return np.abs(fc_t - fc_h)[:, keep].mean(axis=1)


def ci95(x):
    """Mean and 95% confidence half-width across seeds."""
    x = np.asarray(x, float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))
