"""
Publication figures for the OCD virtual-brain study (600 DPI).

Every function takes the `data` dictionary from `compute_figure_data()` and an
output path, and saves a 600 DPI PNG (plus a PDF vector copy where useful).

Usage in Colab:
    import sys; sys.path.insert(0, "/content/ocd-virtual-brain/src")
    from paper_figures import *
    data = compute_figure_data(cache="/content/drive/MyDrive/ocd_figure_data.npz")
    make_all_figures(data, out_dir="/content/drive/MyDrive/ocd_figures_600dpi")
"""
import os
import urllib.request

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Wedge
from matplotlib.lines import Line2D
from nilearn import plotting, datasets, surface
import nibabel as nib

from hopf_model import (LABELS, AAL2_BASE, N_REGIONS, LOOP_NODES, CORTEX, STRIATUM, THALAMUS,
                        OFC, ACC, SEGMENTS, region_pair, load_brain, make_ocd, treatment,
                        fc_per_seed, simulate_hopf, strength_matched_sham)

DPI = 600
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 8,
    "savefig.dpi": DPI, "figure.dpi": 150, "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# Atlas image (AAL2, MNI 2 mm) and region centroids
# ---------------------------------------------------------------------------
AAL2_URL = ("https://raw.githubusercontent.com/spunt/bspmview/master/"
            "supportfiles/AAL2_Atlas_Map.nii")


def get_aal2(path="/content/aal2_atlas.nii"):
    """Download (once) and load the AAL2 atlas image. Labels 1-94 = hopf_model.LABELS."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve(AAL2_URL, path)
    return nib.load(path)


def atlas_centroids(img):
    """MNI centroid (mm) of each of the 94 regions."""
    d = np.asarray(img.dataobj)
    return np.array([nib.affines.apply_affine(img.affine, np.argwhere(d == k).mean(0))
                     for k in range(1, N_REGIONS + 1)])


# ---------------------------------------------------------------------------
# Lobe groups and colours (validated categorical palette, fixed order)
# ---------------------------------------------------------------------------
LOBES = {
    "Frontal": ["Precentral", "Frontal_Sup_2", "Frontal_Mid_2", "Frontal_Inf_Oper",
                "Frontal_Inf_Tri", "Frontal_Inf_Orb_2", "Rolandic_Oper", "Supp_Motor_Area",
                "Olfactory", "Frontal_Sup_Medial", "Frontal_Med_Orb", "Rectus",
                "Paracentral_Lobule"],
    "Orbitofrontal": ["OFCmed", "OFCant", "OFCpost", "OFClat"],
    "Limbic": ["Insula", "Cingulate_Ant", "Cingulate_Mid", "Cingulate_Post", "Hippocampus",
               "ParaHippocampal", "Amygdala"],
    "Subcortical": ["Caudate", "Putamen", "Pallidum", "Thalamus"],
    "Temporal": ["Heschl", "Temporal_Sup", "Temporal_Pole_Sup", "Temporal_Mid",
                 "Temporal_Pole_Mid", "Temporal_Inf"],
    "Parietal": ["Postcentral", "Parietal_Sup", "Parietal_Inf", "SupraMarginal", "Angular",
                 "Precuneus"],
    "Occipital": ["Calcarine", "Cuneus", "Lingual", "Occipital_Sup", "Occipital_Mid",
                  "Occipital_Inf", "Fusiform"],
}
LOBE_COLORS = dict(zip(LOBES, ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#eda100",
                               "#008300", "#4a3aa7"]))
LOBE_OF = {b: lobe for lobe, bases in LOBES.items() for b in bases}
LOOP_PART_COLORS = {"OFC/ACC": "#2a78d6", "Caudate": "#1baf7a", "Thalamus": "#eb6834"}

SHORT = {"Precentral": "PreCG", "Frontal_Sup_2": "SFG", "Frontal_Mid_2": "MFG",
         "Frontal_Inf_Oper": "IFGoper", "Frontal_Inf_Tri": "IFGtri", "Frontal_Inf_Orb_2": "IFGorb",
         "Rolandic_Oper": "ROL", "Supp_Motor_Area": "SMA", "Olfactory": "OLF",
         "Frontal_Sup_Medial": "dmPFC", "Frontal_Med_Orb": "vmPFC", "Rectus": "REC",
         "Paracentral_Lobule": "PCL", "OFCmed": "OFCmed", "OFCant": "OFCant",
         "OFCpost": "OFCpost", "OFClat": "OFClat", "Insula": "INS", "Cingulate_Ant": "ACC",
         "Cingulate_Mid": "MCC", "Cingulate_Post": "PCC", "Hippocampus": "HIP",
         "ParaHippocampal": "PHG", "Amygdala": "AMY", "Caudate": "CAU", "Putamen": "PUT",
         "Pallidum": "PAL", "Thalamus": "THA", "Heschl": "HES", "Temporal_Sup": "STG",
         "Temporal_Pole_Sup": "TPOsup", "Temporal_Mid": "MTG", "Temporal_Pole_Mid": "TPOmid",
         "Temporal_Inf": "ITG", "Postcentral": "PoCG", "Parietal_Sup": "SPG",
         "Parietal_Inf": "IPL", "SupraMarginal": "SMG", "Angular": "ANG", "Precuneus": "PCUN",
         "Calcarine": "CAL", "Cuneus": "CUN", "Lingual": "LING", "Occipital_Sup": "SOG",
         "Occipital_Mid": "MOG", "Occipital_Inf": "IOG", "Fusiform": "FFG"}

DMPFC = region_pair("Frontal_Sup_Medial")
TARGETS = {"DBS-like caudate": STRIATUM, "DBS-like thalamus": THALAMUS, "TMS-like dmPFC": DMPFC}


def _save(fig, path, pdf=True):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor="white")
    if pdf:
        fig.savefig(os.path.splitext(path)[0] + ".pdf", facecolor="white")
    plt.close(fig)
    print("saved", path)


# ---------------------------------------------------------------------------
# Simulation data for all figures (computed once, cached to an .npz file)
# ---------------------------------------------------------------------------
def compute_figure_data(seeds=range(10), dose=-0.15, doses=(-0.04, -0.08, -0.15),
                        ocd_factor=2.0, cache=None, atlas_path="/content/aal2_atlas.nii"):
    """Run every simulation the figures need. Saves/loads `cache` (.npz) if given."""
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        print("loaded cached figure data from", cache)
        return z["data"].item()

    C, real_fc = load_brain()
    C_ocd, loop_mask = make_ocd(C, ocd_factor)
    data = {"C": C, "C_ocd": C_ocd, "real_fc": real_fc, "dose": dose, "doses": list(doses)}
    data["fc_h"] = fc_per_seed(C, seeds=seeds).mean(0)
    data["fc_o"] = fc_per_seed(C_ocd, seeds=seeds).mean(0)
    print("healthy + OCD done")

    exclude = region_pair("Putamen") + region_pair("Pallidum")
    data["sham_regions"] = {n: strength_matched_sham(C, tg, exclude=exclude)[0] for n, tg in TARGETS.items()}
    data["treated"], data["sham_fc"], data["dose_fc"] = {}, {}, {}

    def run(regions=(), a_d=dose, Cbase=None, cut=None):
        Cm, a = treatment(C_ocd if Cbase is None else Cbase, regions, a_treat=a_d, cut=cut)
        return fc_per_seed(Cm, a=a, seeds=seeds).mean(0)

    for name, tg in TARGETS.items():
        for d in doses:
            data["dose_fc"][(name, d)] = run(tg, d)
            data["sham_fc"][(name, d)] = run(data["sham_regions"][name], d)
        data["treated"][name] = data["dose_fc"][(name, dose)]
        print(name, "done")

    # capsulotomy-like cut and a strength-matched sham cut of non-loop fibres
    data["treated"]["Capsulotomy-like"] = run(cut=(THALAMUS, CORTEX))
    cut_edges = [(i, j) for i in THALAMUS for j in CORTEX if C_ocd[i, j] > 0]
    loop_set = set(LOOP_NODES)
    cand = [(i, j) for i in range(N_REGIONS) for j in range(i + 1, N_REGIONS)
            if C_ocd[i, j] > 0 and not ({i, j} & loop_set)]
    used, sham_cut = set(), []
    for i, j in cut_edges:
        best = min((c for c in cand if c not in used), key=lambda c: abs(C_ocd[c] - C_ocd[i, j]))
        used.add(best); sham_cut.append(best)
    C_sham = C_ocd.copy()
    for i, j in sham_cut:
        C_sham[i, j] = C_sham[j, i] = 0
    data["sham_cut_edges"] = sham_cut
    data["sham_fc"][("Capsulotomy-like", dose)] = run(Cbase=C_sham)
    print("capsulotomy + sham cut done")

    # time series for the animation (1 s sampling, 3 minutes)
    data["ts_h"] = simulate_hopf(C, 2.0, T=180, TR=1.0, seed=0)
    data["ts_o"] = simulate_hopf(C_ocd, 2.0, T=180, TR=1.0, seed=0)

    try:
        data["coords"] = atlas_centroids(get_aal2(atlas_path))
    except Exception as e:  # fall back to the coordinates in brain_viz
        from brain_viz import MNI_COORDS
        print("atlas download failed, using fitted coordinates:", e)
        data["coords"] = MNI_COORDS

    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        np.savez_compressed(cache, data=np.array(data, dtype=object))
        print("cached figure data to", cache)
    return data


def _treated_regions(name):
    return TARGETS.get(name, [])


def _direct_mask(data, name):
    """Connections excluded from 'specific effect' views: those touching the target or its
    sham (they are silenced directly), or the cut fibres and the sham-cut fibres."""
    m = np.zeros((N_REGIONS, N_REGIONS), bool)
    if name in TARGETS:
        for r in list(TARGETS[name]) + list(data["sham_regions"][name]):
            m[r, :] = m[:, r] = True
    else:
        m[np.ix_(THALAMUS, CORTEX)] = m[np.ix_(CORTEX, THALAMUS)] = True
        for i, j in data["sham_cut_edges"]:
            m[i, j] = m[j, i] = True
    return m


def specific_change(data, name, dose=None):
    """Treatment minus its strength-matched sham (same dose); direct connections set to NaN.
    Negative = the treatment lowers coupling more than the sham does."""
    dose = data["dose"] if dose is None else dose
    fc_t = data["dose_fc"][(name, dose)] if name in TARGETS else data["treated"][name]
    d = fc_t - data["sham_fc"][(name, dose if name in TARGETS else data["dose"])]
    d[_direct_mask(data, name)] = np.nan
    np.fill_diagonal(d, np.nan)
    return d


def _top_threshold(d, n_edges):
    v = np.abs(d[np.triu_indices(N_REGIONS, 1)])
    v = v[~np.isnan(v)]
    return np.sort(v)[::-1][min(n_edges, len(v)) - 1]


def _loop_colors(highlight_black=(), grey_star=()):
    cols = []
    for i in range(N_REGIONS):
        if i in highlight_black:
            cols.append("#0b0b0b")
        elif i in grey_star:
            cols.append("#898781")
        elif i in CORTEX:
            cols.append(LOOP_PART_COLORS["OFC/ACC"])
        elif i in STRIATUM:
            cols.append(LOOP_PART_COLORS["Caudate"])
        elif i in THALAMUS:
            cols.append(LOOP_PART_COLORS["Thalamus"])
        else:
            cols.append("#c3c2b7")
    return cols


# ---------------------------------------------------------------------------
# Figure: OCD effect + sham-controlled treatment effects (glass brains)
# ---------------------------------------------------------------------------
def fig_treatment_panels(data, path, n_edges=60):
    names = list(data["treated"])
    coords = data["coords"]
    rows = [("OCD − healthy", None)] + [(n, n) for n in names]
    fig = plt.figure(figsize=(7.2, 1.45 * len(rows) + 0.3))
    H = 1.45 * len(rows) + 0.3
    spec = [specific_change(data, n) for n in names]
    vmax_t = np.nanmax([np.nanmax(np.abs(d)) for d in spec])
    for r, (label, name) in enumerate(rows):
        if name is None:
            d = data["fc_o"] - data["fc_h"]; np.fill_diagonal(d, 0)
            vmax = np.abs(d).max(); thr = _top_threshold(d, n_edges)
            colors, sizes, sub = _loop_colors(), [14 if i in LOOP_NODES else 5 for i in range(N_REGIONS)],                 "OCD loop strengthened ×2"
        else:
            d = np.nan_to_num(spec[r - 1]); vmax = vmax_t; thr = _top_threshold(spec[r - 1], n_edges)
            tg = TARGETS.get(name, []); sh = data["sham_regions"].get(name, [])
            colors = _loop_colors(tg, sh)
            sizes = [30 if i in tg else 22 if i in sh else 14 if i in LOOP_NODES else 5 for i in range(N_REGIONS)]
            sub = ("vs strength-matched sham (" + LABELS[sh[0]][:-2] + ")") if sh else "vs strength-matched sham cut"
        top = 1 - (r * 1.45 + 0.3) / H
        ax = fig.add_axes([0.16, top - 1.38 / H, 0.74, 1.35 / H])
        plotting.plot_connectome(d, coords, node_color=colors, node_size=sizes, edge_threshold=thr,
                                 edge_cmap="RdBu_r", edge_vmin=-vmax, edge_vmax=vmax, colorbar=False,
                                 display_mode="lzr", axes=ax, figure=fig, annotate=False,
                                 edge_kwargs={"linewidth": 1.0})
        fig.text(0.15, top - 0.6 / H, label, ha="right", va="center", fontsize=7.5, weight="bold")
        fig.text(0.15, top - 0.85 / H, sub, ha="right", va="center", fontsize=5.8, color="#52514e")
        if r == 0:
            vm_ocd = vmax
    for k, (vm, y0, lab) in enumerate([(vm_ocd, 1 - 1.4 / H, "OCD − healthy"),
                                         (vmax_t, 0.12, "Treatment − sham")]):
        cax = fig.add_axes([0.93, y0, 0.012, 1.0 / H if k == 0 else 0.6])
        cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(-vm, vm), "RdBu_r"), cax=cax)
        cb.set_label(lab + " (ΔFC)", fontsize=6.5); cb.ax.tick_params(labelsize=5.5)
    fig.text(0.5, -0.012, f"Strongest {n_edges} changes per row. Black = treated region, grey = sham region; "
             "their direct connections (and cut fibres) are not shown.", ha="center", fontsize=5.8,
             color="#52514e")
    _save(fig, path)


# ---------------------------------------------------------------------------
# Figure: circular connectogram
# ---------------------------------------------------------------------------
def _circle_layout(gap_deg=4.0):
    """Angles (deg) for each region: left hemisphere on the left half, right mirrored."""
    order_L = [LABELS.index(f"{b}_L") for lobe in LOBES for b in LOBES[lobe]]
    n = len(order_L)
    n_gaps = len(LOBES) - 1
    span = 170.0 - gap_deg * n_gaps
    step = span / (n - 1)
    ang, a = {}, 95.0
    prev_lobe = None
    for i in order_L:
        lobe = LOBE_OF[LABELS[i][:-2]]
        if prev_lobe is not None and lobe != prev_lobe:
            a += gap_deg
        ang[i] = a
        ang[i + 1] = 180.0 - a          # right homologue, mirrored
        a += step
        prev_lobe = lobe
    return ang


def fig_connectogram(matrix, path, title="FC change: OCD − healthy", n_edges=150, vmax=None):
    ang = _circle_layout()
    m = np.array(matrix, float).copy()
    np.fill_diagonal(m, 0)
    iu = np.triu_indices(N_REGIONS, 1)
    vals = m[iu]
    order = np.argsort(np.abs(vals))[::-1][:n_edges][::-1]   # strongest drawn last
    vmax = vmax or np.abs(vals[order]).max()
    norm = mpl.colors.Normalize(-vmax, vmax)
    cmap = mpl.colormaps["RdBu_r"]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_aspect("equal"); ax.axis("off")
    R = 1.0
    pos = {i: R * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]) for i, a in ang.items()}
    for k in order:
        i, j = iu[0][k], iu[1][k]
        p0, p1 = pos[i], pos[j]
        ctrl = (p0 + p1) / 2 * 0.15
        path_ = Path([p0, ctrl, p1], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        w = 0.4 + 2.2 * abs(vals[k]) / vmax
        ax.add_patch(PathPatch(path_, fc="none", ec=cmap(norm(vals[k])), lw=w,
                               alpha=0.85, capstyle="round"))
    # nodes, lobe arcs, labels
    for i, a in ang.items():
        base = LABELS[i][:-2]
        col = LOBE_COLORS[LOBE_OF[base]]
        loop = i in LOOP_NODES
        ax.scatter(*pos[i], s=26 if loop else 12, c=col, edgecolors="#0b0b0b" if loop else "white",
                   linewidths=0.8 if loop else 0.4, zorder=5)
        lab = SHORT[base]
        rot = a if -90 <= ((a + 180) % 360) - 180 <= 90 else a + 180
        ha = "left" if np.cos(np.radians(a)) >= 0 else "right"
        ax.text(*(1.07 * pos[i]), lab, rotation=rot, rotation_mode="anchor", ha=ha, va="center",
                fontsize=4.6, color="#0b0b0b" if loop else "#52514e",
                weight="bold" if loop else "normal")
    for lobe, bases in LOBES.items():
        for h in "LR":
            idx = [LABELS.index(f"{b}_{h}") for b in bases]
            a_ = [ang[i] for i in idx]
            lo, hi = min(a_) - 1.2, max(a_) + 1.2
            ax.add_patch(Wedge((0, 0), 1.035, lo, hi, width=0.022, color=LOBE_COLORS[lobe], lw=0))
            mid = np.radians((lo + hi) / 2)
            ax.text(1.36 * np.cos(mid), 1.36 * np.sin(mid), lobe, ha="center", va="center",
                    fontsize=6.2, color="#0b0b0b",
                    rotation=np.degrees(mid) - 90 if np.sin(mid) >= 0 else np.degrees(mid) + 90)
    ax.text(-1.55, 1.5, "Left hemisphere", fontsize=7.5, weight="bold", ha="left")
    ax.text(1.55, 1.5, "Right hemisphere", fontsize=7.5, weight="bold", ha="right")
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    ax.set_title(title, fontsize=9, weight="bold", pad=2)
    cax = fig.add_axes([0.3, 0.07, 0.4, 0.015])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm, cmap), cax=cax, orientation="horizontal")
    cb.set_label(f"FC change (strongest {n_edges} connections; black-ringed nodes = OCD loop)",
                 fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    _save(fig, path)


# ---------------------------------------------------------------------------
# Figure: network modules (Louvain)
# ---------------------------------------------------------------------------
def network_modules(C, n_runs=200, gamma=1.0, seed=0):
    import bct
    rng = np.random.default_rng(seed)
    best_q, best_ci = -np.inf, None
    for _ in range(n_runs):
        ci, q = bct.community_louvain(C, gamma=gamma, seed=int(rng.integers(1e9)))
        if q > best_q:
            best_q, best_ci = q, ci
    return best_ci, best_q


def fig_modules(data, path):
    ci, q = network_modules(data["C"])
    mods = sorted(set(ci), key=lambda m: -np.sum(ci == m))
    palette = list(LOBE_COLORS.values())
    if len(mods) > len(palette):
        raise ValueError(f"{len(mods)} modules > {len(palette)} colours; raise gamma resolution")
    col_of = {m: palette[k] for k, m in enumerate(mods)}
    colors = [col_of[m] for m in ci]
    sizes = [34 if i in LOOP_NODES else 16 for i in range(N_REGIONS)]
    edgec = ["#0b0b0b" if i in LOOP_NODES else "white" for i in range(N_REGIONS)]
    fig = plt.figure(figsize=(7.2, 2.2))
    ax = fig.add_axes([0, 0.12, 1, 0.85])
    C = data["C"]
    plotting.plot_connectome(C, data["coords"], node_color=colors, node_size=sizes,
                             edge_threshold="97%", edge_cmap="Greys", edge_vmin=0,
                             edge_vmax=float(np.percentile(C[C > 0], 99.5)), colorbar=False,
                             display_mode="lzry", axes=ax, figure=fig, annotate=False,
                             node_kwargs={"edgecolors": edgec, "linewidths": 0.7},
                             edge_kwargs={"linewidth": 0.5, "alpha": 0.5})
    handles = [Line2D([], [], marker="o", ls="", color=col_of[m],
                      label=f"Module {k + 1} ({np.sum(ci == m)} regions)")
               for k, m in enumerate(mods)]
    handles.append(Line2D([], [], marker="o", ls="", mfc="white", mec="#0b0b0b", label="OCD loop region"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=6, frameon=False)
    fig.suptitle(f"Structural network modules (Louvain, Q = {q:.2f})", fontsize=8.5, weight="bold", y=1.02)
    _save(fig, path)
    return ci


# ---------------------------------------------------------------------------
# Region-level treatment effect
# ---------------------------------------------------------------------------
def region_effect(data, name, dose=None):
    """Per region: mean reduction of its coupling with the OCD loop, treatment vs matched sham.
    Positive = the treatment calms that region's loop coupling more than the sham. NaN = target."""
    d = specific_change(data, name, dose)
    eff = -np.nanmean(d[:, LOOP_NODES], axis=1)
    eff[list(TARGETS.get(name, []))] = np.nan
    return eff


def _plot_values(ax, fig, vals, coords, vmax, display_mode, treated=(), cmap="PiYG", size=18, shams=()):
    ok = ~np.isnan(vals)
    disp = plotting.plot_connectome(np.zeros((ok.sum(), ok.sum())), coords[ok],
                                    node_color=[mpl.colormaps[cmap](mpl.colors.Normalize(-vmax, vmax)(v))
                                                for v in vals[ok]],
                                    node_size=size, display_mode=display_mode, axes=ax, figure=fig,
                                    colorbar=False, annotate=False,
                                    node_kwargs={"edgecolors": "#52514e", "linewidths": 0.25})
    if len(treated):
        disp.add_markers(coords[list(treated)], marker_color="#0b0b0b", marker_size=40, marker="*")
    if len(shams):
        disp.add_markers(coords[list(shams)], marker_color="#898781", marker_size=30, marker="*")
    return disp


def fig_region_effects(data, path):
    names = list(data["treated"])
    effs = {n: region_effect(data, n) for n in names}
    vmax = np.nanmax(np.abs(np.concatenate(list(effs.values()))))
    fig = plt.figure(figsize=(7.2, 1.5 * len(names)))
    for r, n in enumerate(names):
        ax = fig.add_axes([0.15, 1 - (r + 1) / len(names), 0.78, 0.95 / len(names)])
        _plot_values(ax, fig, effs[n], data["coords"], vmax, "lzr", _treated_regions(n),
                     shams=data["sham_regions"].get(n, []))
        fig.text(0.14, 1 - (r + 0.5) / len(names), n, ha="right", va="center", fontsize=7.5, weight="bold")
    cax = fig.add_axes([0.95, 0.25, 0.012, 0.5])
    cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(-vmax, vmax), "PiYG"), cax=cax)
    cb.set_label("Loop coupling reduced vs matched sham (ΔFC)\n(green = calmer than sham; ★ black = target, grey = sham)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    _save(fig, path)


def fig_dose_grid(data, path):
    doses = data["doses"]
    effs = {(n, d): region_effect(data, n, d) for n, tg in TARGETS.items() for d in doses}
    vmax = np.nanmax(np.abs(np.concatenate(list(effs.values()))))
    fig = plt.figure(figsize=(7.2, 6.0))
    nr, nc = len(TARGETS), len(doses)
    for r, (n, tg) in enumerate(TARGETS.items()):
        for c, d in enumerate(doses):
            ax = fig.add_axes([0.14 + c * 0.27, 0.93 - (r + 1) * 0.29, 0.26, 0.28])
            _plot_values(ax, fig, effs[(n, d)], data["coords"], vmax, "z", tg, size=14,
                         shams=data["sham_regions"][n])
            if r == 0:
                fig.text(0.14 + c * 0.27 + 0.13, 0.94, f"a = {d}", ha="center", fontsize=8, weight="bold")
        fig.text(0.13, 0.93 - (r + 0.5) * 0.29, n, ha="right", va="center", fontsize=7.5, weight="bold")
    fig.text(0.55, 0.985, "Damping strength (stronger →)", ha="center", fontsize=8)
    cax = fig.add_axes([0.35, 0.03, 0.4, 0.012])
    cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(-vmax, vmax), "PiYG"), cax=cax,
                      orientation="horizontal")
    cb.set_label("Loop coupling reduced vs matched sham (ΔFC; green = calmer than sham; ★ black = target, grey = sham)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    _save(fig, path)


# ---------------------------------------------------------------------------
# Realistic cortical surface figures (fsaverage5, bundled with nilearn)
# ---------------------------------------------------------------------------
SURF_VIEWS = [("left", "lateral"), ("left", "medial"), ("left", "ventral"),
              ("right", "ventral"), ("right", "medial"), ("right", "lateral")]


def _fsaverage():
    return datasets.fetch_surf_fsaverage("fsaverage5")


def _region_texture(atlas_img, fs, hemi):
    """Region label (1-94, 0 = none) for every vertex of the fsaverage5 hemisphere."""
    pial = fs["pial_" + hemi]
    try:
        tex = surface.vol_to_surf(atlas_img, pial, interpolation="nearest_most_frequent", radius=3.0)
    except Exception:
        tex = surface.vol_to_surf(atlas_img, pial, interpolation="nearest", radius=3.0)
    tex = np.rint(np.nan_to_num(tex)).astype(int)
    tex[(tex < 1) | (tex > N_REGIONS)] = 0
    tex = _fill_small_holes(tex, pial)
    # deep grey-matter nuclei sit under the medial wall: never paint them on the cortex
    deep = [r + 1 for b in LOBES["Subcortical"] for r in region_pair(b)]
    tex[np.isin(tex, deep)] = 0
    return tex


def _fill_small_holes(tex, mesh, n_iter=4):
    """Give unlabelled vertices the most common label of their neighbours (fills sampling
    speckles; the large medial wall stays unlabelled)."""
    faces = surface.load_surf_mesh(mesh).faces
    n = len(tex)
    import scipy.sparse as sp
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    tex = tex.copy()
    for _ in range(n_iter):
        empty = np.where(tex == 0)[0]
        if not len(empty):
            break
        new = tex.copy()
        for v in empty:
            nb = tex[A.indices[A.indptr[v]:A.indptr[v + 1]]]
            nb = nb[nb > 0]
            if len(nb) >= 2:
                new[v] = np.bincount(nb).argmax()
        tex = new
    return tex


def _zoom3d(ax, zoom=1.45):
    try:
        ax.set_box_aspect(None, zoom=zoom)
    except TypeError:
        pass


def fig_surface_atlas(atlas_img, path, outline=None):
    """Realistic inflated cortex painted by lobe, with OCD-loop regions outlined in black."""
    fs = _fsaverage()
    lobe_names = list(LOBES)
    cmap = mpl.colors.ListedColormap(["#f0efec"] + [LOBE_COLORS[l] for l in lobe_names])
    outline = outline or {"OCD loop (OFC, ACC)": (CORTEX, "#0b0b0b"),
                          "dmPFC (TMS-like target)": (DMPFC, "#d03b3b")}
    fig = plt.figure(figsize=(7.2, 1.55))
    for k, (hemi, view) in enumerate(SURF_VIEWS):
        tex = _region_texture(atlas_img, fs, hemi)
        cat = np.zeros_like(tex, float)
        for r in range(1, N_REGIONS + 1):
            cat[tex == r] = 1 + lobe_names.index(LOBE_OF[LABELS[r - 1][:-2]])
        ax = fig.add_subplot(1, 6, k + 1, projection="3d")
        plotting.plot_surf_roi(fs["infl_" + hemi], roi_map=cat, hemi=hemi, view=view,
                               bg_map=fs["sulc_" + hemi], bg_on_data=True,
                               cmap=cmap, vmin=0, vmax=len(lobe_names), colorbar=False,
                               axes=ax, figure=fig)
        for name, (regs, col) in outline.items():
            levels = [r + 1 for r in regs if np.any(tex == r + 1)]
            if levels:
                roi = np.where(np.isin(tex, levels), 1, 0)
                try:
                    plotting.plot_surf_contours(fs["infl_" + hemi], roi, levels=[1], colors=[col],
                                                axes=ax, figure=fig)
                except Exception:
                    pass
        _zoom3d(ax)
        ax.set_title(f"{hemi[0].upper()} {view}", fontsize=6.5, pad=-6)
    fig.subplots_adjust(left=0, right=1, top=0.92, bottom=0.08, wspace=0)
    handles = [mpl.patches.Patch(color=LOBE_COLORS[l], label=l) for l in lobe_names if l != "Subcortical"]
    handles += [Line2D([], [], color=c, lw=1.5, label=n) for n, (_, c) in outline.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=5.5, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    _save(fig, path, pdf=False)


def fig_surface_values(atlas_img, rows, path):
    """Paint region values on the inflated cortex.
    rows: list of (title, values[94], colour-bar label, colormap)."""
    fs = _fsaverage()
    texes = {h: _region_texture(atlas_img, fs, h) for h in ("left", "right")}
    fig = plt.figure(figsize=(7.2, 1.25 * len(rows) + 0.2))
    for r, (title, vals, clabel, cmap) in enumerate(rows):
        vals = np.asarray(vals, float)
        vmax = np.nanmax(np.abs(vals))
        for k, (hemi, view) in enumerate(SURF_VIEWS):
            tex = texes[hemi]
            stat = np.zeros(tex.shape)
            ok = tex > 0
            stat[ok] = np.nan_to_num(vals[tex[ok] - 1])
            ax = fig.add_subplot(len(rows), 6, r * 6 + k + 1, projection="3d")
            plotting.plot_surf_stat_map(fs["infl_" + hemi], stat, hemi=hemi, view=view,
                                        bg_map=fs["sulc_" + hemi], bg_on_data=True,
                                        cmap=cmap, vmax=vmax, symmetric_cbar=True, threshold=1e-6,
                                        colorbar=False, axes=ax, figure=fig)
            _zoom3d(ax)
            if r == 0:
                ax.set_title(f"{hemi[0].upper()} {view}", fontsize=6.5, pad=-6)
        y = 1 - (r + 0.5) / len(rows)
        fig.subplots_adjust(left=0.03, right=0.9, top=0.93, bottom=0.02, wspace=0, hspace=0)
        fig.text(0.02, y, title, rotation=90, ha="center", va="center", fontsize=7, weight="bold")
        cax = fig.add_axes([0.92, y - 0.25 / len(rows), 0.008, 0.5 / len(rows)])
        cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(-vmax, vmax), cmap), cax=cax)
        cb.set_label(clabel, fontsize=5.5); cb.ax.tick_params(labelsize=5)
    _save(fig, path, pdf=False)


def ocd_region_coupling(data):
    """Per region: mean FC change (OCD - healthy) with the OCD loop regions."""
    d = data["fc_o"] - data["fc_h"]
    np.fill_diagonal(d, np.nan)
    return np.nanmean(d[:, LOOP_NODES], axis=1)


# ---------------------------------------------------------------------------
# 3D anatomical render of the OCD loop (marching cubes, matplotlib)
# ---------------------------------------------------------------------------
def _mesh(mask, affine, sigma=1.0, step=1, level=0.5):
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes
    vol = gaussian_filter(mask.astype(float), sigma)
    verts, faces, _, _ = marching_cubes(vol, level=level, step_size=step)
    return nib.affines.apply_affine(affine, verts), faces


def _shaded(verts, faces, color, light=(0.3, -0.5, 0.8), ambient=0.38):
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    l = np.asarray(light, float); l /= np.linalg.norm(l)
    shade = ambient + (1 - ambient) * np.abs(n @ l)
    rgb = np.array(mpl.colors.to_rgb(color))
    return tri, np.clip(rgb[None, :] * shade[:, None] + 0.12 * shade[:, None] ** 8, 0, 1)


def fig_3d_loop(data, atlas_img, path, views=((10, 200, "Left lateral"), (90, -90, "Superior"),
                                              (20, 135, "Right anterior oblique"))):
    """Solid 3D OFC/ACC, caudate, thalamus (and dmPFC target) inside a translucent brain,
    with the loop's structural connections as tubes. views = (elevation, azimuth)."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    lab = np.asarray(atlas_img.dataobj)
    aff = atlas_img.affine
    brain_v, brain_f = _mesh(lab > 0, aff, sigma=1.5, step=2)
    parts = [("OFC/ACC", CORTEX, LOOP_PART_COLORS["OFC/ACC"]),
             ("Caudate", STRIATUM, LOOP_PART_COLORS["Caudate"]),
             ("Thalamus", THALAMUS, LOOP_PART_COLORS["Thalamus"]),
             ("dmPFC (TMS-like target)", DMPFC, "#d03b3b")]
    meshes = [(n, *_mesh(np.isin(lab, [r + 1 for r in regs]), aff, sigma=1.1), c) for n, regs, c in parts]
    coords = data["coords"]
    C = data["C_ocd"]
    edges = [(i, j, C[i, j]) for a, b in SEGMENTS.values() for i in a for j in b if C[i, j] > 0]
    wmax = max(w for *_, w in edges)

    fig = plt.figure(figsize=(7.2, 2.6))
    for k, (elev, azim, vtitle) in enumerate(views):
        ax = fig.add_subplot(1, len(views), k + 1, projection="3d")
        tri, fc = _shaded(brain_v, brain_f, "#c3c2b7", ambient=0.6)
        ax.add_collection3d(Poly3DCollection(tri, facecolors=fc, alpha=0.07, linewidths=0))
        for name, v, f, col in meshes:
            tri, fc = _shaded(v, f, col)
            ax.add_collection3d(Poly3DCollection(tri, facecolors=fc, linewidths=0, alpha=0.95))
        for i, j, w in edges:
            p = coords[[i, j]]
            ax.plot(p[:, 0], p[:, 1], p[:, 2], color="#7a1f1f", lw=0.3 + 2.2 * w / wmax,
                    alpha=0.75, solid_capstyle="round")
        ax.set_xlim(-75, 75); ax.set_ylim(-110, 75); ax.set_zlim(-55, 85)
        try:
            ax.set_box_aspect((150, 185, 140), zoom=1.3)
        except TypeError:
            ax.set_box_aspect((150, 185, 140))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(vtitle, fontsize=7, pad=-8)
    handles = [mpl.patches.Patch(color=c, label=n) for n, *_, c in meshes]
    handles.append(Line2D([], [], color="#7a1f1f", lw=1.5, label="Loop connections (width = strength)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=6, frameon=False)
    fig.subplots_adjust(wspace=-0.1, left=0, right=1, top=1, bottom=0.06)
    _save(fig, path, pdf=False)


# ---------------------------------------------------------------------------
# Animated virtual brain (healthy vs OCD) + a 600 DPI frame strip for print
# ---------------------------------------------------------------------------
def _proj(coords, direction):
    return {"x": coords[:, [1, 2]], "y": coords[:, [0, 2]], "z": coords[:, [0, 1]],
            "l": coords[:, [1, 2]], "r": coords[:, [1, 2]]}[direction]


def _activity_axes(fig, rect, coords, title, mode="xz"):
    disp = plotting.plot_glass_brain(None, display_mode=mode, axes=fig.add_axes(rect), figure=fig,
                                     annotate=False, title=None)
    scat = []
    for d, ax in disp.axes.items():
        keep = np.ones(len(coords), bool)
        if d == "l": keep = coords[:, 0] <= 0
        if d == "r": keep = coords[:, 0] >= 0
        xy = _proj(coords[keep], d)
        sc = ax.ax.scatter(xy[:, 0], xy[:, 1], s=20, c=np.zeros(keep.sum()), cmap="magma",
                           vmin=-2.5, vmax=2.5, edgecolors="#0b0b0b", linewidths=0.2, zorder=10)
        scat.append((sc, keep))
    fig.text(rect[0] + rect[2] / 2, rect[1] + rect[3] + 0.01, title, ha="center", fontsize=8, weight="bold")
    return scat


def _zscore(ts):
    return (ts - ts.mean(1, keepdims=True)) / (ts.std(1, keepdims=True) + 1e-12)


def animate_activity(data, path_mp4, fps=10, seconds=None, dpi=200):
    """MP4 (and GIF) of simulated activity: healthy vs OCD virtual brain, glowing nodes."""
    from matplotlib import animation
    zh, zo = _zscore(data["ts_h"]), _zscore(data["ts_o"])
    n = zh.shape[1] if seconds is None else min(zh.shape[1], int(seconds))
    coords = data["coords"]
    fig = plt.figure(figsize=(8, 3.2), facecolor="white")
    sh = _activity_axes(fig, [0.0, 0.08, 0.5, 0.8], coords, "Healthy virtual brain")
    so = _activity_axes(fig, [0.5, 0.08, 0.5, 0.8], coords, "OCD virtual brain (loop ×2)")
    timer = fig.text(0.5, 0.02, "", ha="center", fontsize=7, color="#52514e")
    cax = fig.add_axes([0.42, 0.95, 0.16, 0.02])
    cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(-2.5, 2.5), "magma"), cax=cax,
                      orientation="horizontal")
    cb.ax.tick_params(labelsize=5); cb.set_label("Activity (z)", fontsize=6)

    def update(t):
        for scat, z in ((sh, zh), (so, zo)):
            for sc, keep in scat:
                v = z[keep, t]
                sc.set_array(v)
                sc.set_sizes(8 + 28 * np.clip(np.abs(v), 0, 2.5))
        timer.set_text(f"t = {t:3d} s (simulated BOLD, 1 s sampling)")
        return []

    ani = animation.FuncAnimation(fig, update, frames=n, interval=1000 / fps, blit=False)
    os.makedirs(os.path.dirname(path_mp4) or ".", exist_ok=True)
    try:
        ani.save(path_mp4, writer=animation.FFMpegWriter(fps=fps, bitrate=6000), dpi=dpi)
        print("saved", path_mp4)
    except Exception as e:
        print("MP4 failed (ffmpeg missing?):", e)
    gif = os.path.splitext(path_mp4)[0] + ".gif"
    ani.save(gif, writer=animation.PillowWriter(fps=fps), dpi=90)
    print("saved", gif)
    plt.close(fig)


def fig_activity_strip(data, path, times=(20, 40, 60, 80, 100, 120)):
    """Print version of the animation: activity snapshots, healthy (top) vs OCD (bottom)."""
    zh, zo = _zscore(data["ts_h"]), _zscore(data["ts_o"])
    coords = data["coords"]
    fig = plt.figure(figsize=(7.2, 2.6))
    w = 1 / len(times)
    for k, t in enumerate(times):
        for r, (z, lab) in enumerate(((zh, "Healthy"), (zo, "OCD"))):
            disp = plotting.plot_glass_brain(None, display_mode="z", annotate=False,
                                             axes=fig.add_axes([k * w, 0.52 - r * 0.47, w, 0.42]),
                                             figure=fig)
            ax = disp.axes["z"].ax
            ax.scatter(coords[:, 0], coords[:, 1], s=4 + 16 * np.clip(np.abs(z[:, t]), 0, 2.5),
                       c=z[:, t], cmap="magma", vmin=-2.5, vmax=2.5, edgecolors="#0b0b0b",
                       linewidths=0.15, zorder=10)
            if r == 0:
                fig.text(k * w + w / 2, 0.97, f"t = {t} s", ha="center", fontsize=6.5)
            if k == 0:
                fig.text(-0.005, 0.73 - r * 0.47, lab, rotation=90, ha="right", va="center",
                         fontsize=7, weight="bold")
    cax = fig.add_axes([0.35, 0.0, 0.3, 0.02])
    cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(-2.5, 2.5), "magma"), cax=cax,
                      orientation="horizontal")
    cb.ax.tick_params(labelsize=5); cb.set_label("Simulated activity (z)", fontsize=6)
    _save(fig, path, pdf=False)


# ---------------------------------------------------------------------------
# Everything in one call
# ---------------------------------------------------------------------------
def make_all_figures(data, out_dir, atlas_path="/content/aal2_atlas.nii", animation=True):
    os.makedirs(out_dir, exist_ok=True)
    j = lambda f: os.path.join(out_dir, f)
    atlas = get_aal2(atlas_path)
    fig_surface_atlas(atlas, j("fig01_cortex_atlas_loop.png"))
    fig_3d_loop(data, atlas, j("fig02_3d_ocd_loop.png"))
    fig_modules(data, j("fig03_network_modules.png"))
    fig_connectogram(data["fc_o"] - data["fc_h"], j("fig04_connectogram_ocd.png"))
    fig_surface_values(atlas, [
        ("OCD − healthy", ocd_region_coupling(data), "ΔFC with loop", "RdBu_r"),
        ("TMS-like dmPFC", region_effect(data, "TMS-like dmPFC"), "Loop coupling calmed\nvs sham (green)", "PiYG"),
    ], j("fig05_cortex_ocd_and_dmPFC.png"))
    fig_treatment_panels(data, j("fig06_treatments_glassbrain.png"))
    fig_region_effects(data, j("fig07_treatment_region_effects.png"))
    fig_dose_grid(data, j("fig08_dose_response.png"))
    fig_activity_strip(data, j("figS1_activity_snapshots.png"))
    if animation:
        animate_activity(data, j("movieS1_virtual_brain_activity.mp4"))
