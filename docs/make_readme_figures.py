import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#6b6b6b", "#e6e6e3"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#a8a8a5"

def frame(ax, title, subtitle, xlabel):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=1, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9.5, length=0)
    ax.set_xlabel(xlabel, fontsize=10, color=MUTED, labelpad=9)
    ax.set_title(title, fontsize=13.5, color=INK, fontweight="bold", loc="left", pad=34)
    ax.text(0, 1.055, subtitle, transform=ax.transAxes, fontsize=10.5, color=MUTED, va="bottom")

# ---------------------------------------------------------------- ROGII
ROWS = [("the public route\n(third-party artifacts)", 6.470, 9.471, "bad"),
        ("v11 (mine) — selected",                     8.893, 9.422, None),
        ("v8 (mine) — most robust",                   9.079, 9.167, "good")]
fig, ax = plt.subplots(figsize=(10.4, 3.9), dpi=200); fig.patch.set_facecolor(SURFACE)
ys = list(range(len(ROWS)))[::-1]
for y, (lab, pub, pri, hl) in zip(ys, ROWS):
    col = {"bad": ORANGE, "good": BLUE}.get(hl, NEUTRAL)
    ax.plot([pub, pri], [y, y], color=col, lw=2.2 if hl else 1.6, alpha=0.95 if hl else 0.55,
            solid_capstyle="round", zorder=2)
    ax.scatter([pub], [y], s=78, facecolor=SURFACE, edgecolor=col, lw=2.2, zorder=3)
    ax.scatter([pri], [y], s=86, facecolor=col, edgecolor=SURFACE, lw=1.6, zorder=3)
    ax.text(pub - 0.09, y, f"{pub:.2f}", ha="right", va="center", fontsize=10, color=MUTED)
    ax.text(pri + 0.09, y, f"{pri:.2f}", ha="left", va="center", fontsize=10,
            color=INK if hl else MUTED, fontweight="bold" if hl else "normal")
    ax.text((pub + pri) / 2, y + 0.26, f"+{pri - pub:.2f}", ha="center", va="bottom",
            fontsize=9.5, color=col, fontweight="bold")
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in ROWS], fontsize=10.5)
for t, r in zip(ax.get_yticklabels(), ROWS):
    t.set_color(INK if r[3] else MUTED); t.set_fontweight("bold" if r[3] else "normal")
ax.tick_params(axis="y", pad=10)
ax.set_xlim(5.9, 10.1); ax.set_ylim(-0.75, len(ROWS) - 0.3)
frame(ax, "The route that wins on the public board collapses on the private one",
      "RMSE of dTVT in feet — lower is better. The number above each pair is how much it degrades.",
      "RMSE dTVT (ft)")
ax.legend(handles=[
    Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=MUTED,
           markeredgewidth=2.2, markersize=8.5, label="Public leaderboard"),
    Line2D([], [], marker="o", ls="", markerfacecolor=MUTED, markeredgecolor=SURFACE,
           markersize=9.5, label="Private  (decides the prize)")],
    loc="upper left", bbox_to_anchor=(0, -0.21), frameon=False, fontsize=10,
    labelcolor=MUTED, handletextpad=0.6, ncol=2, columnspacing=2.4)
fig.savefig("rogii_en.png", facecolor=SURFACE, bbox_inches="tight", pad_inches=0.35)

# ---------------------------------------------------------------- RED TEAM
days = ["Aug 27", "Aug 28", "Aug 29", "Aug 31", "Sep 1\n(close)"]
pub  = [41.265, 81.3, 90.1, 91.005, 91.005]
fig, ax = plt.subplots(figsize=(10.4, 4.3), dpi=200); fig.patch.set_facecolor(SURFACE)
x = range(len(days))
ax.axhline(90.99, color=NEUTRAL, lw=1.6, ls=(0, (5, 4)), zorder=1)
ax.text(0.03, 92.4, "public bronze cut (90.99)", fontsize=9.5, color=MUTED)
ax.plot(x, pub, color=BLUE, lw=2.4, marker="o", markersize=8,
        markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)
ax.scatter([len(days) - 1], [0], s=150, facecolor=ORANGE, edgecolor=SURFACE, lw=2, zorder=4)
ax.annotate("final private score = 0\non every submission",
            xy=(len(days) - 1.06, 2), xytext=(2.55, 26), fontsize=10, color=ORANGE,
            ha="center", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6,
                            connectionstyle="arc3,rad=-0.22"))
ax.set_xticks(list(x)); ax.set_xticklabels(days, fontsize=10)
ax.set_ylim(-7, 103); ax.set_xlim(-0.35, len(days) - 0.5)
ax.yaxis.grid(True, color=GRID, lw=1, zorder=0); ax.xaxis.grid(False)
frame(ax, "Near bronze on the public board, exactly zero on the private one",
      "Public score peaked at 91.0 — around the top 10% — on a documented guardrail the hidden one blocked.",
      "")
ax.set_ylabel("Normalised attack score  (0–1000, clipped to ≤100 here)",
              fontsize=10, color=MUTED, labelpad=9)
fig.savefig("redteam_en.png", facecolor=SURFACE, bbox_inches="tight", pad_inches=0.35)

# ---------------------------------------------------------------- POKEMON
labels = ["Search alone", "Network alone", "Network + search"]
vals   = [0.709, 0.724, 0.859]
fig, ax = plt.subplots(figsize=(9.2, 3.9), dpi=200); fig.patch.set_facecolor(SURFACE)
ys = [2, 1, 0]
for y, lab, v in zip(ys, labels, vals):
    col = BLUE if y == 0 else NEUTRAL
    ax.barh(y, v, height=0.46, color=col, alpha=1 if y == 0 else 0.55, zorder=2)
    ax.text(v + 0.012, y, f"{v:.3f}", va="center", fontsize=10.5,
            color=INK if y == 0 else MUTED, fontweight="bold" if y == 0 else "normal")
ax.scatter([0.865], [0], s=150, marker="|", color=ORANGE, lw=2.6, zorder=4)
ax.annotate("0.865 predicted if the two\nadd independently in log-odds",
            xy=(0.869, 0.28), xytext=(0.545, 1.62), fontsize=9.5, color=ORANGE,
            ha="center", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4,
                            connectionstyle="arc3,rad=-0.18"))
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=10.5)
for t, y in zip(ax.get_yticklabels(), ys):
    t.set_color(INK if y == 0 else MUTED); t.set_fontweight("bold" if y == 0 else "normal")
ax.tick_params(axis="y", pad=10)
ax.set_xlim(0, 1.0); ax.set_ylim(-0.55, 2.95)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
frame(ax, "Policy network and search add up — they do not interfere",
      "Win rate against the ladder agent. The observed 0.859 lands on the independent prediction.",
      "Win rate vs the ladder agent")
fig.savefig("pokemon_en.png", facecolor=SURFACE, bbox_inches="tight", pad_inches=0.35)
print("3 charts ok")
