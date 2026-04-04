"""
Stage 5 — Evaluation & Paper Figures
=====================================
Produces four publication-ready figures:

  Fig 1 — Cumulative Regret Curve
  Fig 2 — Rolling Accuracy (Click-Through Rate proxy)
  Fig 3 — Final Accuracy Bar Chart with confidence annotation
  Fig 4 — LinUCB-Semantic Weight Heatmap (what the bandit learned)

All figures saved to results/ as high-res PNGs (300 dpi) + one combined
PDF suitable for direct inclusion in a paper.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
    "legend.framealpha":0.9,
    "legend.edgecolor": "#cccccc",
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

import os
os.makedirs("results", exist_ok=True)

# ── Palette (colour-blind friendly) ──────────────────────────────────────────
PALETTE = {
    "ShowAll":    "#9e9e9e",   # grey
    "MuteAll":    "#bdbdbd",   # light grey
    "RuleBased":  "#FF8C00",   # amber
    "LinUCB_Tab": "#1976D2",   # blue
    "LinUCB_Sem": "#2E7D32",   # green
}
STYLES = {
    "ShowAll":    dict(lw=1.4, ls=":",  alpha=0.85),
    "MuteAll":    dict(lw=1.4, ls="-.", alpha=0.85),
    "RuleBased":  dict(lw=2.0, ls="--", alpha=0.95),
    "LinUCB_Tab": dict(lw=2.2, ls="-",  alpha=0.95),
    "LinUCB_Sem": dict(lw=2.8, ls="-",  alpha=1.00),
}
LABELS = {
    "ShowAll":    "Show-All (baseline)",
    "MuteAll":    "Mute-All (baseline)",
    "RuleBased":  "Rule-Based",
    "LinUCB_Tab": "LinUCB-Tabular",
    "LinUCB_Sem": "LinUCB-Semantic ★",
}
AGENTS     = list(PALETTE.keys())
N_TOTAL    = 10_000

# ── Load data ─────────────────────────────────────────────────────────────────
log = pd.read_csv("simulation_log.csv")
summary = pd.read_csv("agent_summary.csv")

steps = log["step"].values

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Cumulative Regret
# ═══════════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(9, 5))

for agent in AGENTS:
    ax1.plot(steps, log[f"{agent}_cum_reg"],
             color=PALETTE[agent], label=LABELS[agent],
             **STYLES[agent])

# Annotate final regret values
for agent in AGENTS:
    final = int(log[f"{agent}_cum_reg"].iloc[-1])
    ax1.annotate(
        f"{final:,}",
        xy=(N_TOTAL, final),
        xytext=(N_TOTAL - 100, final),
        ha="right", va="center",
        fontsize=9.5, color=PALETTE[agent], fontweight="bold",
    )

ax1.set_title("Cumulative Regret over 10,000 Notifications")
ax1.set_xlabel("Notification (t)")
ax1.set_ylabel("Cumulative Regret  (# wrong decisions)")
ax1.set_xlim(0, N_TOTAL * 1.02)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

# Add shaded "learning phase" region
ax1.axvspan(0, 1500, color="#e3f2fd", alpha=0.45, zorder=0)
ax1.text(750, ax1.get_ylim()[1] * 0.97, "Exploration\nphase",
         ha="center", va="top", fontsize=8.5, color="#1565C0", style="italic")

ax1.legend(loc="upper left", fontsize=9)
fig1.tight_layout()
fig1.savefig("results/fig1_cumulative_regret.png")
plt.close(fig1)
print("✓ fig1_cumulative_regret.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Rolling Accuracy (500-step window)
# ═══════════════════════════════════════════════════════════════════════════════
WINDOW = 500
fig2, ax2 = plt.subplots(figsize=(9, 5))

for agent in AGENTS:
    # Derive per-step reward from cumulative
    cum = log[f"{agent}_cum_rew"].values
    per_step = np.diff(cum, prepend=0).astype(float)
    rolling  = pd.Series(per_step).rolling(WINDOW, min_periods=WINDOW // 4).mean()
    ax2.plot(steps, rolling * 100,
             color=PALETTE[agent], label=LABELS[agent],
             **STYLES[agent])

# Final accuracy horizontal references
for agent in ["LinUCB_Sem", "LinUCB_Tab"]:
    final_acc = summary.loc[summary["agent"] == agent, "accuracy_%"].values[0]
    ax2.axhline(final_acc, color=PALETTE[agent], lw=0.8, ls=":", alpha=0.5)

ax2.set_title(f"Rolling Accuracy  (window = {WINDOW} notifications)")
ax2.set_xlabel("Notification (t)")
ax2.set_ylabel("Accuracy  (%)")
ax2.set_xlim(0, N_TOTAL)
ax2.set_ylim(0, 100)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

ax2.axvspan(0, WINDOW, color="#fff8e1", alpha=0.5, zorder=0)
ax2.text(WINDOW // 2, 4, "Warm-up", ha="center", va="bottom",
         fontsize=8, color="#f57f17", style="italic")

ax2.legend(loc="lower right", fontsize=9)
fig2.tight_layout()
fig2.savefig("results/fig2_rolling_accuracy.png")
plt.close(fig2)
print("✓ fig2_rolling_accuracy.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Final Accuracy Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(8, 4.5))

accs   = [summary.loc[summary["agent"] == a, "accuracy_%"].values[0] for a in AGENTS]
colors = [PALETTE[a] for a in AGENTS]
xlabs  = [LABELS[a] for a in AGENTS]

bars = ax3.barh(xlabs[::-1], accs[::-1], color=colors[::-1],
                height=0.55, edgecolor="white", linewidth=1.2)

# Value labels
for bar, acc in zip(bars, accs[::-1]):
    ax3.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
             f"{acc:.1f}%", va="center", ha="left",
             fontsize=10.5, fontweight="bold",
             color=colors[::-1][list(bars).index(bar)])

# Chance level reference
chance = 100 / 3  # 3 arms
ax3.axvline(chance, color="#757575", lw=1.2, ls="--", alpha=0.7)
ax3.text(chance + 0.2, -0.5, f"Random\n({chance:.1f}%)",
         fontsize=8, color="#757575", va="bottom")

ax3.set_xlabel("Final Accuracy  (%)")
ax3.set_title("Agent Accuracy over 10,000 Notifications")
ax3.set_xlim(0, 100)
ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax3.grid(axis="x", alpha=0.25, linestyle="--")
ax3.grid(axis="y", alpha=0)

# Delta annotation (LinUCB_Sem vs RuleBased)
sem_acc  = summary.loc[summary["agent"] == "LinUCB_Sem",  "accuracy_%"].values[0]
rule_acc = summary.loc[summary["agent"] == "RuleBased",   "accuracy_%"].values[0]
delta    = sem_acc - rule_acc
ax3.annotate(
    f"+{delta:.1f}pp over Rule-Based",
    xy=(sem_acc, 0), xytext=(sem_acc - 12, 0.35),
    fontsize=9, color=PALETTE["LinUCB_Sem"], fontweight="bold",
    arrowprops=dict(arrowstyle="-|>", color=PALETTE["LinUCB_Sem"],
                    lw=1.5, mutation_scale=12),
)

fig3.tight_layout()
fig3.savefig("results/fig3_final_accuracy.png")
plt.close(fig3)
print("✓ fig3_final_accuracy.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Weight Heatmap  (what LinUCB-Semantic learned)
# ═══════════════════════════════════════════════════════════════════════════════
# Reconstruct feature-level theta from the simulation log.
# We don't have the matrices saved, so we re-derive *effective* arm preference
# scores from the log itself: per-feature accuracy breakdown per arm.

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Weight Heatmap  (what LinUCB-Semantic learned)
# ═══════════════════════════════════════════════════════════════════════════════

ARMS = ["NOW", "MUTE", "BATCH"]
CAT_COLS = ["app_source", "sender_type", "user_context", "time_slot"]

orig = pd.read_csv("notifications_10k_with_embeddings.csv")
merged = pd.concat([orig.reset_index(drop=True),
                    log[["LinUCB_Sem_choice"]].reset_index(drop=True)], axis=1)
merged["correct_sem"] = (merged["LinUCB_Sem_choice"] == merged["true_optimal_action"]).astype(int)

global_arm_probs = log["LinUCB_Sem_choice"].value_counts(normalize=True)

rows_labels = []
heatmap_data = []

for col in CAT_COLS:
    for val in sorted(merged[col].unique()):
        mask = merged[col] == val
        arm_probs = merged.loc[mask, "LinUCB_Sem_choice"].value_counts(normalize=True)
        affinities = []
        for arm in ARMS:
            p     = arm_probs.get(arm, 0)
            p_glo = global_arm_probs.get(arm, 0)
            affinities.append(p - p_glo)
        # FIX 1: Removed newline to save vertical space
        rows_labels.append(f"{col}: {val}")
        heatmap_data.append(affinities)

heatmap_arr = np.array(heatmap_data)

# FIX 2: Wider figure (8 instead of 6) and slightly taller rows (0.45 instead of 0.38)
fig4, ax4 = plt.subplots(figsize=(8, len(rows_labels) * 0.45))

vmax = np.abs(heatmap_arr).max()
im   = ax4.imshow(heatmap_arr, aspect="auto", cmap="RdYlGn",
                  vmin=-vmax, vmax=vmax)

ax4.set_xticks(range(len(ARMS)))
ax4.set_xticklabels(ARMS, fontweight="bold", fontsize=11)
ax4.set_yticks(range(len(rows_labels)))
ax4.set_yticklabels(rows_labels, fontsize=8.5)
ax4.set_title("LinUCB-Semantic: Arm Affinity by Feature\n"
              "(green = over-represented, red = under-represented vs. global rate)",
              fontsize=11, pad=15)

for i in range(len(rows_labels)):
    for j in range(len(ARMS)):
        v = heatmap_arr[i, j]
        ax4.text(j, i, f"{v:+.2f}", ha="center", va="center",
                 fontsize=7.5,
                 color="white" if abs(v) > 0.15 else "#333333",
                 fontweight="bold" if abs(v) > 0.15 else "normal")

cbar = fig4.colorbar(im, ax=ax4, fraction=0.03, pad=0.02)
cbar.set_label("Affinity  (Δ from global arm rate)", fontsize=9)

group_sizes = [len(merged[col].unique()) for col in CAT_COLS]
dividers = np.cumsum(group_sizes)[:-1] - 0.5
for d in dividers:
    ax4.axhline(d, color="white", lw=2.0)

start = 0
group_labels = ["App Source", "Sender Type", "User Context", "Time Slot"]
for glabel, gsize in zip(group_labels, group_sizes):
    mid = start + gsize / 2 - 0.5
    # FIX 3: Adjusted X coordinate for the side labels so they don't clip
    ax4.text(-0.55, mid, glabel, ha="right", va="center",
             fontsize=9, fontweight="bold", color="#444444",
             transform=ax4.get_yaxis_transform())
    start += gsize

# FIX 4: Explicitly force a left margin so tight_layout doesn't crush the heatmap
fig4.subplots_adjust(left=0.3)
fig4.tight_layout()
fig4.savefig("results/fig4_weight_heatmap.png")
plt.close(fig4)
print("✓ fig4_weight_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED PDF (all 4 figures on one page)
# ═══════════════════════════════════════════════════════════════════════════════
fig_all = plt.figure(figsize=(18, 20))
gs = gridspec.GridSpec(2, 2, figure=fig_all,
                       hspace=0.38, wspace=0.32)

def _load_and_draw(png_path, ax):
    from matplotlib.image import imread
    img = imread(png_path)
    ax.imshow(img)
    ax.axis("off")

imgs = [
    "results/fig1_cumulative_regret.png",
    "results/fig2_rolling_accuracy.png",
    "results/fig3_final_accuracy.png",
    "results/fig4_weight_heatmap.png",
]
titles = [
    "Fig 1 — Cumulative Regret",
    "Fig 2 — Rolling Accuracy",
    "Fig 3 — Final Accuracy",
    "Fig 4 — Arm Affinity Heatmap",
]
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

for (r, c), img_path, title in zip(positions, imgs, titles):
    ax = fig_all.add_subplot(gs[r, c])
    _load_and_draw(img_path, ax)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=6)

fig_all.suptitle(
    "Context-Aware Notification Management via LinUCB Bandits\n"
    "Evaluation Results — 10,000 Notification Simulation",
    fontsize=15, fontweight="bold", y=0.995
)
fig_all.savefig("results/all_figures.pdf", format="pdf")
plt.close(fig_all)
print("✓ all_figures.pdf  (combined 4-panel)")

# ── Console summary ───────────────────────────────────────────────────────────
print("\n── Final Summary ────────────────────────────────────────────────────────")
print(summary.to_string(index=False))
print("\nFiles written to results/")
