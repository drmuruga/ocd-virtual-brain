"""
Brain visualisation for the OCD virtual-brain project (nilearn based).

Region centres are MNI coordinates (mm) of the 94 AAL2 cerebral regions, in the
same order as hopf_model.LABELS. They were derived from neurolib's AAL2 atlas
centres and mapped to MNI space with an affine fit (~1 mm error).

Usage in Colab:
    import sys; sys.path.insert(0, "/content/ocd-virtual-brain/src")
    from hopf_model import *
    from brain_viz import *
"""
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting

from hopf_model import LABELS, LOOP_NODES, CORTEX, STRIATUM, THALAMUS, region_pair

MNI_COORDS = np.array([
    [ -38.5,   -5.7,   50.3],
    [  40.5,   -8.2,   51.6],
    [ -19.2,   37.3,   35.8],
    [  23.8,   34.3,   37.8],
    [ -34.8,   34.2,   30.8],
    [  40.0,   33.0,   30.4],
    [ -48.4,   12.5,   19.4],
    [  51.2,   13.6,   20.9],
    [ -45.3,   29.9,   14.4],
    [  51.9,   28.8,   14.0],
    [ -42.1,   32.3,   -5.7],
    [  49.1,   33.6,   -6.0],
    [ -47.0,   -8.6,   14.7],
    [  53.4,   -6.4,   14.7],
    [  -5.5,    4.8,   60.8],
    [   8.2,    0.3,   61.5],
    [  -7.8,   15.3,  -11.6],
    [  11.0,   16.3,  -11.5],
    [  -3.7,   48.7,   30.4],
    [  10.7,   49.9,   30.0],
    [  -4.2,   53.9,   -7.0],
    [   9.8,   51.6,   -6.7],
    [  -4.6,   37.4,  -18.3],
    [   9.5,   36.2,  -18.4],
    [ -13.9,   37.4,  -19.4],
    [  19.0,   38.4,  -19.6],
    [ -24.6,   47.8,  -13.2],
    [  31.8,   48.8,  -13.8],
    [ -29.0,   24.4,  -17.9],
    [  36.1,   25.3,  -17.8],
    [ -43.3,   40.8,  -12.4],
    [  52.0,   36.5,  -14.3],
    [ -35.2,    5.9,    4.3],
    [  40.4,    6.4,    2.0],
    [  -3.6,   35.3,   13.5],
    [   9.5,   36.3,   15.4],
    [  -5.8,  -15.0,   42.4],
    [   7.6,   -8.6,   40.3],
    [  -4.7,  -44.4,   25.8],
    [   6.9,  -42.3,   23.3],
    [ -24.5,  -21.2,  -10.2],
    [  29.5,  -19.2,  -10.3],
    [ -20.4,  -16.5,  -21.5],
    [  25.6,  -14.9,  -20.8],
    [ -22.6,   -0.9,  -18.0],
    [  28.3,    0.4,  -17.8],
    [  -6.4,  -79.2,    5.5],
    [  16.7,  -71.7,    9.7],
    [  -5.0,  -80.8,   27.0],
    [  14.0,  -78.2,   29.0],
    [ -14.3,  -68.6,   -5.9],
    [  16.3,  -65.8,   -3.8],
    [ -15.9,  -85.0,   27.4],
    [  24.5,  -79.6,   31.1],
    [ -32.4,  -80.7,   15.1],
    [  37.2,  -78.4,   19.4],
    [ -36.9,  -78.9,   -9.0],
    [  38.4,  -81.5,   -8.1],
    [ -31.1,  -40.0,  -21.2],
    [  34.2,  -39.1,  -20.5],
    [ -42.2,  -22.4,   48.7],
    [  39.8,  -25.7,   52.9],
    [ -23.3,  -59.0,   59.0],
    [  25.4,  -58.8,   62.5],
    [ -41.8,  -46.2,   46.1],
    [  45.4,  -45.4,   48.9],
    [ -54.5,  -33.5,   30.5],
    [  56.2,  -30.8,   34.1],
    [ -43.7,  -60.6,   35.1],
    [  44.8,  -58.8,   38.0],
    [  -7.2,  -56.1,   49.4],
    [   9.8,  -55.7,   45.0],
    [  -8.0,  -25.1,   70.3],
    [   6.9,  -31.3,   68.8],
    [ -12.3,   10.9,    9.8],
    [  15.6,   12.6,    9.1],
    [ -24.3,    3.1,    2.9],
    [  28.8,    5.2,    2.8],
    [ -17.9,   -0.8,    0.2],
    [  21.7,    0.4,    0.4],
    [ -10.5,  -18.4,    8.8],
    [  12.5,  -17.2,    9.0],
    [ -42.0,  -19.8,   10.9],
    [  46.6,  -16.6,   11.3],
    [ -52.7,  -19.9,    7.3],
    [  58.6,  -21.9,    6.8],
    [ -40.6,   15.1,  -19.4],
    [  51.0,   13.5,  -16.8],
    [ -55.8,  -33.6,   -1.3],
    [  58.0,  -37.3,   -1.6],
    [ -37.3,   14.5,  -34.1],
    [  47.8,   12.7,  -32.6],
    [ -50.0,  -27.4,  -22.3],
    [  54.7,  -31.8,  -22.4]])

LOOP_COLOR, OTHER_COLOR = "#d62728", "#9e9e9e"


def node_colors(highlight=LOOP_NODES, color=LOOP_COLOR, other=OTHER_COLOR):
    return [color if i in set(highlight) else other for i in range(len(LABELS))]


def node_sizes(C, min_size=10, max_size=80):
    s = C.sum(axis=1)
    return min_size + (max_size - min_size) * (s - s.min()) / (s.max() - s.min())


def plot_network(C, title="Structural connectome", edge_threshold="97%",
                 highlight=LOOP_NODES, display_mode="lzry", output_file=None):
    """Glass-brain view of the whole network; OCD loop regions in red, node size = wiring strength."""
    disp = plotting.plot_connectome(
        C, MNI_COORDS, node_color=node_colors(highlight), node_size=node_sizes(C),
        edge_threshold=edge_threshold, edge_cmap="viridis", edge_vmin=0, edge_vmax=float(np.percentile(C[C > 0], 99)),
        edge_kwargs={"linewidth": 1.2, "alpha": 0.85},
        display_mode=display_mode, title=title, output_file=output_file)
    return disp


def plot_loop(C, title="OCD loop: OFC/ACC - caudate - thalamus", display_mode="lzry", output_file=None):
    """Only the loop's connections, coloured by strength."""
    loop = np.zeros_like(C)
    idx = np.ix_(LOOP_NODES, LOOP_NODES)
    loop[idx] = C[idx]
    colors = []
    for i in range(len(LABELS)):
        colors.append("#1f77b4" if i in CORTEX else "#2ca02c" if i in STRIATUM
                      else "#ff7f0e" if i in THALAMUS else "none")
    sizes = [60 if i in LOOP_NODES else 0 for i in range(len(LABELS))]
    return plotting.plot_connectome(
        loop, MNI_COORDS, node_color=colors, node_size=sizes, edge_threshold=0,
        edge_cmap="Reds", edge_vmin=0, edge_vmax=float(loop.max()), colorbar=True, display_mode=display_mode,
        title=title, output_file=output_file)


def plot_fc_change(delta_fc, title="FC change (OCD - healthy)", edge_threshold="99%",
                   display_mode="lzry", output_file=None):
    """Glass-brain view of the strongest FC changes: red = increase, blue = decrease."""
    d = np.array(delta_fc, float).copy()
    np.fill_diagonal(d, 0)
    lim = np.abs(d).max()
    return plotting.plot_connectome(
        d, MNI_COORDS, node_color=node_colors(), node_size=20,
        edge_threshold=edge_threshold, edge_cmap="RdBu_r", edge_vmin=-lim, edge_vmax=lim,
        colorbar=True, display_mode=display_mode, title=title, output_file=output_file)


def plot_targets(targets, shams=None, title="Virtual treatment targets", display_mode="lzry",
                 output_file=None):
    """Mark treatment targets (colours) and their shams (grey) on a glass brain.
    `targets` / `shams`: dict name -> list of region indices."""
    palette = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    coords, colors, sizes = [], [], []
    for k, regs in enumerate(targets.values()):
        for r in regs:
            coords.append(MNI_COORDS[r]); colors.append(palette[k % len(palette)]); sizes.append(150)
    for regs in (shams or {}).values():
        for r in regs:
            coords.append(MNI_COORDS[r]); colors.append("#7f7f7f"); sizes.append(70)
    n = len(coords)
    disp = plotting.plot_connectome(np.zeros((n, n)), np.array(coords), node_color=colors,
                                    node_size=sizes, display_mode=display_mode, title=title, colorbar=False,
                                    output_file=output_file)
    return disp


def plot_node_values(values, title="Region values", cmap="viridis", display_mode="lzry",
                     output_file=None, node_size=60):
    """Colour every region by a value (e.g. wiring strength, FC change, treatment effect)."""
    return plotting.plot_markers(np.asarray(values, float), MNI_COORDS, node_cmap=cmap,
                                 node_size=node_size, display_mode=display_mode, title=title,
                                 output_file=output_file)


def view_3d(matrix, edge_threshold="97%", title="Interactive 3D brain network", html_file=None):
    """Interactive, rotatable 3D brain (displays inline in Colab). Optionally save as HTML."""
    view = plotting.view_connectome(np.array(matrix, float), MNI_COORDS,
                                    edge_threshold=edge_threshold, node_size=4,
                                    title=title, colorbar=True)
    if html_file:
        view.save_as_html(html_file)
    return view
