#!/usr/bin/env python3
"""Generate all plots for OpenMP bucket sort report."""

import csv
import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "wykresy")
os.makedirs(OUT, exist_ok=True)

PHASES = ["generate_seconds", "distribute_seconds", "sort_seconds", "rewrite_seconds", "total_seconds"]

_OLD_COLS = [
    "timestamp","slurm_job_id","slurm_nodelist","slurm_cpus_per_task",
    "algorithm","vector_size","base_seed","threads",
    "generate_seconds","distribute_seconds","sort_seconds",
    "rewrite_seconds","total_seconds","test_mode",
]
_NEW_COLS = [
    "timestamp","slurm_job_id","slurm_nodelist","slurm_cpus_per_task",
    "algorithm","bucket_multiplier","vector_size","base_seed","threads",
    "generate_seconds","distribute_seconds","sort_seconds",
    "rewrite_seconds","total_seconds","test_mode",
]
PHASE_LABELS = {
    "generate_seconds":   "Generowanie",
    "distribute_seconds": "Rozdzielanie",
    "sort_seconds":       "Sortowanie",
    "rewrite_seconds":    "Przepisywanie",
    "total_seconds":      "Łącznie",
}
PHASE_COLORS = {
    "generate_seconds":   "#4C72B0",
    "distribute_seconds": "#DD8452",
    "sort_seconds":       "#55A868",
    "rewrite_seconds":    "#C44E52",
    "total_seconds":      "#8172B2",
}
THREADS = [1, 2, 4, 8, 16, 24, 32, 40, 48]

# ── helpers ──────────────────────────────────────────────────────────────────

def load_dir(directory):
    """Load all CSV files from a result directory; return averaged DataFrame.

    Handles both the old 14-column format (no bucket_multiplier) and the new
    15-column format.  When both formats are present in the same file (Ares
    appended new rows to old files), only the new-format rows are kept so that
    results always reflect the most recent experiment run.
    """
    parsed = []
    for path in glob.glob(os.path.join(directory, "run_*.csv")):
        with open(path, newline="") as fh:
            reader = csv.reader(fh)
            next(reader)  # skip header line (may be old or new format)
            for fields in reader:
                n = len(fields)
                if n == len(_NEW_COLS):
                    parsed.append(dict(zip(_NEW_COLS, fields)))
                elif n == len(_OLD_COLS):
                    row = dict(zip(_OLD_COLS, fields))
                    row["bucket_multiplier"] = None
                    parsed.append(row)
    if not parsed:
        raise FileNotFoundError(f"No data rows in {directory}")

    df = pd.DataFrame(parsed)
    for col in PHASES + ["threads", "vector_size"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Prefer new-format rows (bucket_multiplier present) if any exist
    new_rows = df[df["bucket_multiplier"].notna()]
    if not new_rows.empty:
        df = new_rows

    return (
        df.groupby(["threads", "vector_size"])[PHASES]
        .mean()
        .reset_index()
        .sort_values("threads")
    )


def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def style_ax(ax, xlabel=None, ylabel=None, title=None):
    ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(THREADS)
    ax.set_xticklabels(THREADS, fontsize=8)


# ── load data ────────────────────────────────────────────────────────────────
print("Loading data …")
strong1 = load_dir(os.path.join(ROOT, "results_strong_alg1"))
strong2 = load_dir(os.path.join(ROOT, "results_strong_alg2"))
weak1   = load_dir(os.path.join(ROOT, "results_weak_alg1"))
weak2   = load_dir(os.path.join(ROOT, "results_weak_alg2"))

# strong scaling uses fixed vector_size → pick that row per thread count
strong1 = strong1.groupby("threads")[PHASES].mean().reset_index().sort_values("threads")
strong2 = strong2.groupby("threads")[PHASES].mean().reset_index().sort_values("threads")

# weak scaling: one row per thread (vector_size grows proportionally)
weak1 = weak1.sort_values("threads").reset_index(drop=True)
weak2 = weak2.sort_values("threads").reset_index(drop=True)

threads = strong1["threads"].values

# ── (1) Strong scaling: execution time per phase, both algorithms ─────────────
print("Plot 1 – strong scaling time …")
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
for ax, df, alg_label in zip(axes, [strong1, strong2], ["Algorytm 1", "Algorytm 2"]):
    for phase in PHASES:
        ax.plot(df["threads"], df[phase], marker="o", markersize=5,
                label=PHASE_LABELS[phase], color=PHASE_COLORS[phase],
                linewidth=1.6)
    style_ax(ax, xlabel="Liczba wątków", ylabel="Czas [s]", title=alg_label)
    ax.legend(fontsize=8, loc="upper right")
fig.suptitle("Skalowanie silne – czas wykonania poszczególnych faz", fontsize=12, fontweight="bold")
fig.tight_layout()
savefig(fig, "strong_time_phases.png")

# ── (2) Strong scaling: total time comparison ─────────────────────────────────
print("Plot 2 – strong scaling total time comparison …")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(strong1["threads"], strong1["total_seconds"], marker="o", markersize=6,
        label="Algorytm 1", color="#4C72B0", linewidth=2)
ax.plot(strong2["threads"], strong2["total_seconds"], marker="s", markersize=6,
        label="Algorytm 2", color="#DD8452", linewidth=2)
style_ax(ax, xlabel="Liczba wątków", ylabel="Czas [s]",
         title="Skalowanie silne – całkowity czas wykonania")
ax.legend(fontsize=10)
fig.tight_layout()
savefig(fig, "strong_total_comparison.png")

# ── (3) Strong scaling: speedup per phase ────────────────────────────────────
print("Plot 3 – strong scaling speedup …")
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
t1_s1 = strong1.set_index("threads")
t1_s2 = strong2.set_index("threads")

for ax, df, alg_label in zip(axes, [t1_s1, t1_s2], ["Algorytm 1", "Algorytm 2"]):
    for phase in PHASES:
        speedup = df.loc[1, phase] / df[phase]
        ax.plot(df.index, speedup, marker="o", markersize=5,
                label=PHASE_LABELS[phase], color=PHASE_COLORS[phase],
                linewidth=1.6)
    # ideal speedup line
    ax.plot(THREADS, THREADS, "k--", linewidth=1, label="Idealne")
    style_ax(ax, xlabel="Liczba wątków", ylabel="Przyspieszenie S(p)",
             title=alg_label)
    ax.legend(fontsize=8)
fig.suptitle("Skalowanie silne – przyspieszenie faz algorytmu", fontsize=12, fontweight="bold")
fig.tight_layout()
savefig(fig, "strong_speedup_phases.png")

# ── (4) Strong scaling: speedup total comparison ──────────────────────────────
print("Plot 4 – strong scaling speedup total …")
fig, ax = plt.subplots(figsize=(8, 5))
su1 = strong1.set_index("threads")["total_seconds"]
su2 = strong2.set_index("threads")["total_seconds"]
ax.plot(THREADS, su1.loc[1] / su1, marker="o", markersize=6,
        label="Algorytm 1", color="#4C72B0", linewidth=2)
ax.plot(THREADS, su2.loc[1] / su2, marker="s", markersize=6,
        label="Algorytm 2", color="#DD8452", linewidth=2)
ax.plot(THREADS, THREADS, "k--", linewidth=1.2, label="Idealne")
style_ax(ax, xlabel="Liczba wątków", ylabel="Przyspieszenie S(p)",
         title="Skalowanie silne – całkowite przyspieszenie")
ax.legend(fontsize=10)
fig.tight_layout()
savefig(fig, "strong_speedup_total.png")

# ── (5) Strong scaling: efficiency ───────────────────────────────────────────
print("Plot 5 – strong scaling efficiency …")
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, df_idx, alg_label in zip(axes, [t1_s1, t1_s2], ["Algorytm 1", "Algorytm 2"]):
    for phase in ["distribute_seconds", "sort_seconds", "rewrite_seconds", "total_seconds"]:
        speedup = df_idx.loc[1, phase] / df_idx[phase]
        eff = speedup / df_idx.index
        ax.plot(df_idx.index, eff * 100, marker="o", markersize=5,
                label=PHASE_LABELS[phase], color=PHASE_COLORS[phase],
                linewidth=1.6)
    ax.axhline(100, color="k", linestyle="--", linewidth=1, label="Idealna")
    style_ax(ax, xlabel="Liczba wątków", ylabel="Efektywność [%]", title=alg_label)
    ax.legend(fontsize=8)
fig.suptitle("Skalowanie silne – efektywność", fontsize=12, fontweight="bold")
fig.tight_layout()
savefig(fig, "strong_efficiency.png")

# ── (6) Weak scaling: execution time ─────────────────────────────────────────
print("Plot 6 – weak scaling time …")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(weak1["threads"], weak1["total_seconds"], marker="o", markersize=6,
        label="Algorytm 1", color="#4C72B0", linewidth=2)
ax.plot(weak2["threads"], weak2["total_seconds"], marker="s", markersize=6,
        label="Algorytm 2", color="#DD8452", linewidth=2)
style_ax(ax, xlabel="Liczba wątków (rozmiar ∝ wątki × 1M)", ylabel="Czas [s]",
         title="Skalowanie słabe – całkowity czas wykonania")
ax.legend(fontsize=10)
fig.tight_layout()
savefig(fig, "weak_total_comparison.png")

# ── (7) Weak scaling: time per phase for each algorithm ──────────────────────
print("Plot 7 – weak scaling time phases …")
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
for ax, df, alg_label in zip(axes, [weak1, weak2], ["Algorytm 1", "Algorytm 2"]):
    for phase in PHASES:
        ax.plot(df["threads"], df[phase], marker="o", markersize=5,
                label=PHASE_LABELS[phase], color=PHASE_COLORS[phase],
                linewidth=1.6)
    style_ax(ax, xlabel="Liczba wątków", ylabel="Czas [s]", title=alg_label)
    ax.legend(fontsize=8)
fig.suptitle("Skalowanie słabe – czas wykonania poszczególnych faz", fontsize=12, fontweight="bold")
fig.tight_layout()
savefig(fig, "weak_time_phases.png")

# ── (8) Weak scaling: efficiency (ideal = constant time) ─────────────────────
print("Plot 8 – weak scaling efficiency …")
fig, ax = plt.subplots(figsize=(8, 5))
w1_t = weak1.set_index("threads")["total_seconds"]
w2_t = weak2.set_index("threads")["total_seconds"]
eff1 = w1_t.iloc[0] / w1_t * 100
eff2 = w2_t.iloc[0] / w2_t * 100
ax.plot(THREADS, eff1, marker="o", markersize=6, label="Algorytm 1", color="#4C72B0", linewidth=2)
ax.plot(THREADS, eff2, marker="s", markersize=6, label="Algorytm 2", color="#DD8452", linewidth=2)
ax.axhline(100, color="k", linestyle="--", linewidth=1.2, label="Idealna")
style_ax(ax, xlabel="Liczba wątków", ylabel="Efektywność skalowania słabego [%]",
         title="Skalowanie słabe – efektywność")
ax.legend(fontsize=10)
fig.tight_layout()
savefig(fig, "weak_efficiency.png")

# ── (9) Stacked bar: phase breakdown for strong scaling ──────────────────────
print("Plot 9 – stacked bar phase breakdown …")
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
phase_stack = ["generate_seconds", "distribute_seconds", "sort_seconds", "rewrite_seconds"]
stack_labels = [PHASE_LABELS[p] for p in phase_stack]
stack_colors = [PHASE_COLORS[p] for p in phase_stack]

for ax, df, alg_label in zip(axes, [strong1, strong2], ["Algorytm 1", "Algorytm 2"]):
    bottoms = np.zeros(len(df))
    x = np.arange(len(df))
    for phase, label, color in zip(phase_stack, stack_labels, stack_colors):
        vals = df[phase].values
        ax.bar(x, vals, bottom=bottoms, label=label, color=color, alpha=0.9, width=0.7)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(df["threads"].astype(int), fontsize=8)
    ax.set_xlabel("Liczba wątków", fontsize=10)
    ax.set_ylabel("Czas [s]", fontsize=10)
    ax.set_title(alg_label, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Skalowanie silne – struktura czasu wykonania", fontsize=12, fontweight="bold")
fig.tight_layout()
savefig(fig, "strong_stacked_phases.png")

# ── (10) Single-thread phase breakdown pie (alg1 vs alg2) ─────────────────────
print("Plot 10 – pie charts phase breakdown at 1 thread …")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, df, alg_label in zip(axes, [strong1, strong2], ["Algorytm 1", "Algorytm 2"]):
    row = df[df["threads"] == 1].iloc[0]
    vals = [row[p] for p in phase_stack]
    explode = [0.03] * len(phase_stack)
    ax.pie(vals, labels=stack_labels, colors=stack_colors, explode=explode,
           autopct="%1.1f%%", startangle=140, textprops={"fontsize": 9})
    ax.set_title(f"{alg_label} – 1 wątek", fontsize=11, fontweight="bold")
fig.suptitle("Udział faz w łącznym czasie (1 wątek)", fontsize=12, fontweight="bold")
fig.tight_layout()
savefig(fig, "phase_pie_1thread.png")

# ── (11) Bucket multiplier sweep ─────────────────────────────────────────────
print("Plot 11 – bucket multiplier sweep …")
sweep_csvs = glob.glob(os.path.join(ROOT, "results_bucket_sweep", "*.csv"))
if sweep_csvs:
    rows = []
    for path in sweep_csvs:
        df = pd.read_csv(path, quotechar='"')
        df.columns = df.columns.str.strip()
        rows.append(df)
    sweep = pd.concat(rows, ignore_index=True)
    sweep_avg = (
        sweep.groupby(["bucket_multiplier", "threads"])[
            ["distribute_seconds", "sort_seconds", "total_seconds"]
        ]
        .mean()
        .reset_index()
    )

    mults = sorted(sweep_avg["bucket_multiplier"].unique())
    highlight_threads = [1, 8, 16, 32, 48]
    cmap = plt.cm.viridis
    colors = {t: cmap(i / (len(highlight_threads) - 1))
              for i, t in enumerate(highlight_threads)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # left: total time vs multiplier
    ax = axes[0]
    for t in highlight_threads:
        sub = sweep_avg[sweep_avg["threads"] == t].sort_values("bucket_multiplier")
        ax.plot(sub["bucket_multiplier"], sub["total_seconds"],
                marker="o", markersize=5, linewidth=1.8,
                label=f"T={t}", color=colors[t])
    ax.set_xscale("log", base=2)
    ax.set_xticks(mults)
    ax.set_xticklabels([str(m) for m in mults], fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Mnożnik liczby kubełków", fontsize=10)
    ax.set_ylabel("Czas łączny [s]", fontsize=10)
    ax.set_title("Czas łączny vs mnożnik kubełków", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    # right: distribute time vs multiplier (log-log)
    ax = axes[1]
    for t in highlight_threads:
        sub = sweep_avg[sweep_avg["threads"] == t].sort_values("bucket_multiplier")
        ax.plot(sub["bucket_multiplier"], sub["distribute_seconds"],
                marker="o", markersize=5, linewidth=1.8,
                label=f"T={t}", color=colors[t])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(mults)
    ax.set_xticklabels([str(m) for m in mults], fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Mnożnik liczby kubełków", fontsize=10)
    ax.set_ylabel("Czas rozdzielania [s] (skala log)", fontsize=10)
    ax.set_title("Faza rozdzielania vs mnożnik kubełków", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle("Wpływ mnożnika liczby kubełków na wydajność Algorytmu 2",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "bucket_sweep.png")
else:
    print("  (no sweep data – skipping)")

print("\nDone — all plots saved to", OUT)
