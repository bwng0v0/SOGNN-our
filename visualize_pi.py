"""
Publication-quality PI figures — each saved as an independent file.

Outputs:
  ./result/fig_topomap.pdf      Main figure: scalp topography
  ./result/fig_regions.pdf      Main figure: region-level comparison
  ./result/fig_ranking.pdf      Supplementary: full 62-electrode ranking
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy.interpolate import CloughTocher2DInterpolator

# ── Global style ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.4,
    'ytick.major.width': 0.4,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'mathtext.fontset': 'dejavuserif',
})

# ── Channel / region definitions ─────────────────────────────
CHANNEL_NAMES = [
    'FP1','FPZ','FP2','AF3','AF4',
    'F7','F5','F3','F1','FZ','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8',
    'T7','C5','C3','C1','CZ','C2','C4','C6','T8',
    'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
    'P7','P5','P3','P1','PZ','P2','P4','P6','P8',
    'PO7','PO5','PO3','POZ','PO4','PO6','PO8',
    'CB1','O1','OZ','O2','CB2',
]

CHANNEL_POS = {
    'FP1': (-0.15, 0.90), 'FPZ': (0.0, 0.95), 'FP2': (0.15, 0.90),
    'AF3': (-0.20, 0.80), 'AF4': (0.20, 0.80),
    'F7':  (-0.65, 0.65), 'F5':  (-0.45, 0.65), 'F3':  (-0.30, 0.65),
    'F1':  (-0.15, 0.65), 'FZ':  (0.0,  0.65),  'F2':  (0.15, 0.65),
    'F4':  (0.30, 0.65),  'F6':  (0.45, 0.65),  'F8':  (0.65, 0.65),
    'FT7': (-0.75, 0.45), 'FC5': (-0.55, 0.45), 'FC3': (-0.35, 0.45),
    'FC1': (-0.15, 0.45), 'FCZ': (0.0,  0.45),  'FC2': (0.15, 0.45),
    'FC4': (0.35, 0.45),  'FC6': (0.55, 0.45),  'FT8': (0.75, 0.45),
    'T7':  (-0.85, 0.20), 'C5':  (-0.60, 0.20), 'C3':  (-0.35, 0.20),
    'C1':  (-0.15, 0.20), 'CZ':  (0.0,  0.20),  'C2':  (0.15, 0.20),
    'C4':  (0.35, 0.20),  'C6':  (0.60, 0.20),  'T8':  (0.85, 0.20),
    'TP7': (-0.80, -0.05),'CP5': (-0.55, -0.05),'CP3': (-0.35, -0.05),
    'CP1': (-0.15, -0.05),'CPZ': (0.0, -0.05),  'CP2': (0.15, -0.05),
    'CP4': (0.35, -0.05), 'CP6': (0.55, -0.05), 'TP8': (0.80, -0.05),
    'P7':  (-0.70, -0.30),'P5':  (-0.50, -0.30),'P3':  (-0.30, -0.30),
    'P1':  (-0.15, -0.30),'PZ':  (0.0, -0.30),  'P2':  (0.15, -0.30),
    'P4':  (0.30, -0.30), 'P6':  (0.50, -0.30), 'P8':  (0.70, -0.30),
    'PO7': (-0.45, -0.55),'PO5': (-0.30, -0.50),'PO3': (-0.15, -0.50),
    'POZ': (0.0, -0.50),  'PO4': (0.15, -0.50), 'PO6': (0.30, -0.50),
    'PO8': (0.45, -0.55),
    'CB1': (-0.35, -0.75),'O1':  (-0.15, -0.70),'OZ':  (0.0, -0.70),
    'O2':  (0.15, -0.70), 'CB2': (0.35, -0.75),
}

REGION_DEF = {
    'Pre-frontal': ['FP1','FPZ','FP2','AF3','AF4'],
    'Frontal':     ['F7','F5','F3','F1','FZ','F2','F4','F6','F8',
                    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8'],
    'Central':     ['T7','C5','C3','C1','CZ','C2','C4','C6','T8'],
    'Centro-parietal': ['TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8'],
    'Parietal':    ['P7','P5','P3','P1','PZ','P2','P4','P6','P8'],
    'Occipital':   ['PO7','PO5','PO3','POZ','PO4','PO6','PO8',
                    'CB1','O1','OZ','O2','CB2'],
}

REGION_MAP = {}
for region, channels in REGION_DEF.items():
    for ch in channels:
        REGION_MAP[ch] = region

REGION_COLORS = {
    'Pre-frontal':     '#3B7DD8',
    'Frontal':         '#4DAF4A',
    'Central':         '#E41A1C',
    'Centro-parietal': '#984EA3',
    'Parietal':        '#E6AB02',
    'Occipital':       '#1B9E77',
}

REGION_ORDER = ['Pre-frontal', 'Frontal', 'Central',
                'Centro-parietal', 'Parietal', 'Occipital']


def _build_df(summary):
    """summary csv → DataFrame with name, region, pi(%), sem(%)."""
    rows = []
    for _, r in summary.iterrows():
        e = int(r['electrode'])
        name = CHANNEL_NAMES[e]
        rows.append({
            'electrode': e, 'name': name,
            'region': REGION_MAP[name],
            'pi': r['pi_acc_mean'] * 100,
            'sem': r['pi_acc_std'] / np.sqrt(r['n_observations']) * 100,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# Figure 1 — Topographic scalp map  (single-column, ~3.5 in)
# ═══════════════════════════════════════════════════════════════
def make_topomap(summary):
    electrodes = summary['electrode'].astype(int).values
    values = summary['pi_acc_mean'].values

    xs, ys, vs, names = [], [], [], []
    for e, v in zip(electrodes, values):
        name = CHANNEL_NAMES[e]
        if name in CHANNEL_POS:
            x, y = CHANNEL_POS[name]
            xs.append(x); ys.append(y); vs.append(v); names.append(name)
    xs, ys, vs = np.array(xs), np.array(ys), np.array(vs)

    fig, ax = plt.subplots(figsize=(3.8, 3.8))

    # Interpolate
    gx, gy = np.meshgrid(np.linspace(-1.15, 1.15, 400),
                          np.linspace(-1.0, 1.20, 400))
    gz = CloughTocher2DInterpolator(list(zip(xs, ys)), vs)(gx, gy)

    head_r, hc = 1.02, (0.0, 0.10)
    gz[np.sqrt((gx - hc[0])**2 + (gy - hc[1])**2) > head_r] = np.nan

    im = ax.contourf(gx, gy, gz, levels=60, cmap='RdYlBu_r',
                     vmin=vs.min(), vmax=vs.max(), extend='both')

    # Head / nose / ears
    t = np.linspace(0, 2*np.pi, 200)
    ax.plot(hc[0]+head_r*np.cos(t), hc[1]+head_r*np.sin(t), 'k-', lw=1.0)
    ax.plot([-0.06, 0, 0.06], [1.12, 1.19, 1.12], 'k-', lw=1.0)
    ear = np.linspace(-np.pi/2, np.pi/2, 30)
    ax.plot(-1.02-0.05*np.cos(ear), 0.10+0.17*np.sin(ear), 'k-', lw=1.0)
    ax.plot( 1.02+0.05*np.cos(ear), 0.10+0.17*np.sin(ear), 'k-', lw=1.0)

    # Electrode dots
    ax.scatter(xs, ys, c='k', s=6, zorder=5, linewidths=0)

    # Top-5 labels
    for i in np.argsort(vs)[-5:]:
        ax.annotate(
            names[i], (xs[i], ys[i]), fontsize=6, fontweight='bold',
            ha='center', va='bottom', color='k',
            xytext=(0, 3.5), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.75))

    ax.set_xlim(-1.28, 1.28); ax.set_ylim(-1.08, 1.28)
    ax.set_aspect('equal'); ax.axis('off')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.82, aspect=22)
    cbar.set_label(r'$\Delta$ Accuracy', fontsize=8, labelpad=4)
    cbar.ax.tick_params(labelsize=6.5, width=0.3, length=1.5)
    cbar.outline.set_linewidth(0.4)

    fig.tight_layout(pad=0.3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'./result/fig_topomap.{ext}', dpi=300,
                    bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print('Saved: fig_topomap.pdf / .png')


# ═══════════════════════════════════════════════════════════════
# Figure 2 — Region-level comparison  (single-column, ~3.5 in)
# ═══════════════════════════════════════════════════════════════
def make_region_fig(summary):
    df = _build_df(summary)

    # Stats per region
    stats = []
    for region in REGION_ORDER:
        g = df[df['region'] == region]['pi']
        stats.append({'region': region, 'mean': g.mean(),
                      'sem': g.std() / np.sqrt(len(g)), 'n': len(g)})
    stats = pd.DataFrame(stats)

    # Sort by mean descending
    stats = stats.sort_values('mean', ascending=True).reset_index(drop=True)
    sorted_regions = stats['region'].tolist()

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    y_pos = np.arange(len(sorted_regions))

    # Horizontal bars
    ax.barh(y_pos, stats['mean'], height=0.52,
            color=[REGION_COLORS[r] for r in sorted_regions],
            edgecolor='white', linewidth=0.3, alpha=0.45, zorder=2)
    ax.errorbar(stats['mean'], y_pos, xerr=stats['sem'],
                fmt='none', ecolor='0.3', elinewidth=0.6,
                capsize=2.5, capthick=0.6, zorder=3)

    # Overlay individual electrode dots
    rng = np.random.default_rng(42)
    for i, region in enumerate(sorted_regions):
        vals = df[df['region'] == region]['pi'].values
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(vals, np.full(len(vals), y_pos[i]) + jitter,
                   s=16, color=REGION_COLORS[region], edgecolor='white',
                   linewidth=0.3, zorder=5, alpha=0.8)

    # Mean labels
    for i, row in stats.iterrows():
        ax.text(row['mean'] + row['sem'] + 0.015, y_pos[i],
                f'{row["mean"]:.2f}%',
                va='center', ha='left', fontsize=6.5, fontweight='bold',
                color=REGION_COLORS[row['region']])

    # n labels
    for i, row in stats.iterrows():
        ax.text(0.02, y_pos[i], f'n={row["n"]:.0f}',
                va='center', ha='left', fontsize=5, color='0.55', style='italic')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_regions, fontsize=7.5)
    ax.set_xlabel(r'$\Delta$ Accuracy (%)', fontsize=9)
    ax.set_xlim(left=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout(pad=0.4)
    for ext in ('pdf', 'png'):
        fig.savefig(f'./result/fig_regions.{ext}', dpi=300,
                    bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print('Saved: fig_regions.pdf / .png')


# ═══════════════════════════════════════════════════════════════
# Figure 3 (Suppl.) — Full 62-electrode ranking  (double-col)
# ═══════════════════════════════════════════════════════════════
def make_ranking_fig(summary):
    df = _build_df(summary)

    # Sort: region order, then PI descending within region
    region_rank = {r: i for i, r in enumerate(REGION_ORDER)}
    df['rr'] = df['region'].map(region_rank)
    df = df.sort_values(['rr', 'pi'], ascending=[True, False]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.0, 7.5))

    y_positions = []
    y_labels = []
    y_colors = []
    y_vals = []
    group_bounds = []
    y = 0

    for region in REGION_ORDER:
        grp = df[df['region'] == region]
        start = y
        for _, row in grp.iterrows():
            y_positions.append(y)
            y_labels.append(row['name'])
            y_colors.append(REGION_COLORS[region])
            y_vals.append(row['pi'])
            y += 1
        group_bounds.append((start, y - 1, region))
        y += 1.0

    y_positions = np.array(y_positions)
    y_vals = np.array(y_vals)
    global_mean = y_vals.mean()

    # Lollipop stems from global mean
    for i in range(len(y_positions)):
        ax.plot([global_mean, y_vals[i]], [y_positions[i], y_positions[i]],
                color=y_colors[i], linewidth=0.5, alpha=0.5, zorder=2)
    ax.scatter(y_vals, y_positions, c=y_colors, s=18, zorder=4,
              edgecolors='white', linewidths=0.3)

    # Mean line
    ax.axvline(x=global_mean, color='0.5', linewidth=0.5, linestyle='--', zorder=1)
    ax.text(global_mean, y_positions[-1] + 3,
            f'Mean = {global_mean:.2f}%',
            ha='center', va='bottom', fontsize=6, color='0.45', style='italic')

    # X range
    xmin = max(0, y_vals.min() - 0.12)
    xmax = y_vals.max() + 0.12
    ax.set_xlim(xmin, xmax)

    # Region shading + labels
    for start, end, region in group_bounds:
        ax.axhspan(start - 0.45, end + 0.45,
                    color=REGION_COLORS[region], alpha=0.05, zorder=0)
        ax.text(xmin + 0.005, (start + end) / 2, region,
                fontsize=6, fontweight='bold',
                color=REGION_COLORS[region], alpha=0.8,
                ha='left', va='center')

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=5.5)
    ax.set_xlabel(r'$\Delta$ Accuracy (%)', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.invert_yaxis()

    fig.tight_layout(pad=0.4)
    for ext in ('pdf', 'png'):
        fig.savefig(f'./result/fig_ranking.{ext}', dpi=300,
                    bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print('Saved: fig_ranking.pdf / .png')


# ═══════════════════════════════════════════════════════════════
# Figure 4 — Pure PI ranking, two-column layout  (double-col)
# ═══════════════════════════════════════════════════════════════
def make_ranking_flat_fig(summary):
    df = _build_df(summary)
    df = df.sort_values('pi', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1

    n = len(df)
    half = (n + 1) // 2  # 31 left, 31 right

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.2, 5.0), sharey=False)

    global_mean = df['pi'].mean()
    pi_min = df['pi'].min()
    pi_max = df['pi'].max()
    xmin = pi_min - 0.12
    xmax = pi_max + 0.12

    # Colormap: PI intensity → color
    cmap = plt.cm.YlOrRd
    norm = plt.Normalize(vmin=pi_min, vmax=pi_max)

    for panel_idx, ax in enumerate([ax_l, ax_r]):
        start = panel_idx * half
        end = min(start + half, n)
        sub = df.iloc[start:end].reset_index(drop=True)
        n_rows = len(sub)
        y_pos = np.arange(n_rows)

        colors = [cmap(norm(v)) for v in sub['pi'].values]

        # Horizontal bars
        ax.barh(y_pos, sub['pi'].values, height=0.65,
                color=colors, edgecolor='white', linewidth=0.3, zorder=2)

        # Value labels at bar end
        for i in range(n_rows):
            ax.text(sub['pi'].iloc[i] + 0.015, y_pos[i],
                    f'{sub["pi"].iloc[i]:.2f}',
                    va='center', ha='left', fontsize=5, color='0.35')

        # Electrode name + rank as y-tick
        labels = [f'#{sub["rank"].iloc[i]:<3d} {sub["name"].iloc[i]}'
                  for i in range(n_rows)]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=6, fontfamily='monospace')

        # Mean line
        ax.axvline(x=global_mean, color='0.5', linewidth=0.5,
                   linestyle='--', zorder=1)

        ax.set_xlim(xmin, xmax)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)

    ax_l.set_xlabel(r'$\Delta$ Accuracy (%)', fontsize=8)
    ax_r.set_xlabel(r'$\Delta$ Accuracy (%)', fontsize=8)

    # Mean annotation (left panel only)
    ax_l.text(global_mean, -1.2, f'Mean\n{global_mean:.2f}%',
              ha='center', va='bottom', fontsize=5.5, color='0.45', style='italic')

    fig.subplots_adjust(bottom=0.06, wspace=0.42, top=0.97, left=0.10, right=0.97)

    for ext in ('pdf', 'png'):
        fig.savefig(f'./result/fig_ranking_flat.{ext}', dpi=300,
                    bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print('Saved: fig_ranking_flat.pdf / .png')


# ── Main ──────────────────────────────────────────────────────
def main():
    summary = pd.read_csv('./result/pi_summary.csv')
    make_topomap(summary)
    make_region_fig(summary)
    make_ranking_fig(summary)
    make_ranking_flat_fig(summary)


if __name__ == '__main__':
    main()
