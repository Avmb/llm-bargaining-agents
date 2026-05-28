"""Build the honesty-vs-utility scatter plots used in the paper.

Reads the pre-computed result dataframes under ``results/`` and writes three
PDFs into ``analysis/figures/``:

- ``honesty_utility_untrained.pdf`` -- the five zero-shot models in self-play.
- ``honesty_utility_trained.pdf``   -- base Qwen3-8B vs the six joint-trained
  variants (absolute axes).
- ``honesty_utility_trained_delta.pdf`` -- each trained variant's change vs the
  base self-play baseline, restricted to its matched-aware transparency cell.

For each model/variant we compute:
- avg_honesty: mean of all honesty observations (buyer + seller) across cells
  where they are defined (seller_honesty in buyer_unaware/both_unaware,
  buyer_honesty in seller_unaware/both_unaware).
- avg_utility: mean of normalised per-agent utility across all trials, averaging
  buyer and seller sides; utility is normalised by (v_B - v_S); no-deal -> 0.

Run from anywhere: ``python analysis/make_honesty_utility_scatter.py``.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = REPO / "analysis" / "figures"

# display label -> result directory (relative to results/)
UNTRAINED = [
    ("claude-sonnet-4-6", "crossmodel/sonnet46"),
    ("claude-opus-4-7",   "crossmodel/opus47"),
    ("GPT-5.2",           "crossmodel/gpt52"),
    ("GPT-5.5",           "crossmodel/gpt55"),
    ("Qwen3.5-9B",        "crossmodel/qwen35"),
]

TRAINED = [
    ("Qwen3-8B base",   "rl_evals/joint_base"),
    ("GRPO as B",       "rl_evals/joint_grpo_as_buyer"),
    ("GRPO as S",       "rl_evals/joint_grpo_as_seller"),
    ("GRPO self-play",  "rl_evals/joint_grpo_selfplay"),
    ("CISPO as B",      "rl_evals/joint_cispo_as_buyer"),
    ("CISPO as S",      "rl_evals/joint_cispo_as_seller"),
    ("CISPO self-play", "rl_evals/joint_cispo_selfplay"),
]

BASE_TRAINED = "rl_evals/joint_base"

# For each variant: (matched transparency cell, which side is trained).
# self-play uses both_unaware (only cell with all four honesty ratings defined)
# and averages buyer+seller (both sides are trained).
VARIANT_MATCH = {
    "GRPO as B":       ("seller_unaware", "buyer"),
    "GRPO as S":       ("buyer_unaware",  "seller"),
    "GRPO self-play":  ("both_unaware",   "both"),
    "CISPO as B":      ("seller_unaware", "buyer"),
    "CISPO as S":      ("buyer_unaware",  "seller"),
    "CISPO self-play": ("both_unaware",   "both"),
}


def load_trials(rel_dir: str) -> pd.DataFrame:
    with open(RESULTS / rel_dir / "data" / "trials_with_honesty.pkl", "rb") as f:
        return pickle.load(f)


def metrics(df: pd.DataFrame) -> dict:
    surplus = df["buyer_res_price"].astype(float) - df["seller_res_price"].astype(float)
    nb = np.where(df["deal"] == "deal", df["buyer_utility"].astype(float) / surplus, 0.0)
    ns = np.where(df["deal"] == "deal", df["seller_utility"].astype(float) / surplus, 0.0)
    avg_util = float(np.mean(np.concatenate([nb, ns])))
    sh = pd.to_numeric(df["seller_honesty"], errors="coerce").dropna().to_numpy()
    bh = pd.to_numeric(df["buyer_honesty"], errors="coerce").dropna().to_numpy()
    avg_hon = float(np.mean(np.concatenate([sh, bh])))
    return {"honesty": avg_hon, "utility": avg_util}


def collect(rows):
    out = []
    for label, rel in rows:
        m = metrics(load_trials(rel))
        out.append({"label": label, **m})
        print(f"{label:25s}  honesty={m['honesty']:.3f}  utility={m['utility']:.3f}")
    return out


def scatter(rows, outpath, title, *, label_offsets=None,
            x_axis_label="Average honesty (0--4, judge-rated)",
            y_axis_label="Average normalised utility per agent",
            zero_lines=False):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    xs = [r["honesty"] for r in rows]
    ys = [r["utility"] for r in rows]
    ax.scatter(xs, ys, s=52, c="#1f77b4", zorder=3)
    label_offsets = label_offsets or {}
    for r in rows:
        dx, dy, ha = label_offsets.get(r["label"], (6, 4, "left"))
        ax.annotate(r["label"], (r["honesty"], r["utility"]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=10.2, ha=ha)
    ax.set_xlabel(x_axis_label, fontsize=12)
    ax.set_ylabel(y_axis_label, fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=12)
    xpad = (max(xs) - min(xs)) * 0.12 or 0.02
    ypad = (max(ys) - min(ys)) * 0.18 or 0.005
    ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
    ax.set_ylim(min(ys) - ypad, max(ys) + ypad)
    if zero_lines:
        ax.axhline(0, color="#888", linewidth=0.7, zorder=2)
        ax.axvline(0, color="#888", linewidth=0.7, zorder=2)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")


def trained_role_metrics(df: pd.DataFrame, cell: str, side: str) -> dict:
    sub = df[df["transparency"] == cell].copy()
    surplus = sub["buyer_res_price"].astype(float) - sub["seller_res_price"].astype(float)
    nb = np.where(sub["deal"] == "deal", sub["buyer_utility"].astype(float) / surplus, 0.0)
    ns = np.where(sub["deal"] == "deal", sub["seller_utility"].astype(float) / surplus, 0.0)
    bh = pd.to_numeric(sub["buyer_honesty"], errors="coerce").dropna().to_numpy()
    sh = pd.to_numeric(sub["seller_honesty"], errors="coerce").dropna().to_numpy()
    if side == "buyer":
        return {"honesty": float(np.mean(bh)), "utility": float(np.mean(nb))}
    if side == "seller":
        return {"honesty": float(np.mean(sh)), "utility": float(np.mean(ns))}
    return {"honesty": float(np.mean(np.concatenate([bh, sh]))),
            "utility": float(np.mean(np.concatenate([nb, ns])))}


def collect_trained_deltas():
    base_df = load_trials(BASE_TRAINED)
    rows = []
    for label, rel in TRAINED:
        if label == "Qwen3-8B base":
            rows.append({"label": label, "honesty": 0.0, "utility": 0.0})
            print(f"{label:25s}  d_honesty=0.000  d_utility=0.000  (origin)")
            continue
        cell, side = VARIANT_MATCH[label]
        var = trained_role_metrics(load_trials(rel), cell, side)
        base = trained_role_metrics(base_df, cell, side)
        d = {"label": label, "honesty": var["honesty"] - base["honesty"],
             "utility": var["utility"] - base["utility"]}
        rows.append(d)
        print(f"{label:25s}  cell={cell:14s} side={side:6s}  "
              f"d_honesty={d['honesty']:+.3f}  d_utility={d['utility']:+.3f}")
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("== untrained ==")
    untrained = collect(UNTRAINED)
    print("\n== trained ==")
    trained = collect(TRAINED)
    print("\n== trained deltas (matched cell, trained-side only) ==")
    trained_delta = collect_trained_deltas()

    untrained_off = {
        "claude-sonnet-4-6": (6, 4, "left"), "claude-opus-4-7": (6, 4, "left"),
        "GPT-5.2": (-6, -12, "right"), "GPT-5.5": (6, 4, "left"),
        "Qwen3.5-9B": (6, 4, "left"),
    }
    trained_off = {
        "Qwen3-8B base": (-6, 6, "right"), "GRPO as B": (-6, 6, "right"),
        "GRPO as S": (6, 2, "left"), "GRPO self-play": (-6, 4, "right"),
        "CISPO as B": (-6, 4, "right"), "CISPO as S": (-6, -12, "right"),
        "CISPO self-play": (6, 4, "left"),
    }
    trained_delta_off = {
        "Qwen3-8B base": (-6, 6, "right"), "GRPO as B": (-6, 6, "right"),
        "GRPO as S": (6, 4, "left"), "GRPO self-play": (-6, 4, "right"),
        "CISPO as B": (-6, 4, "right"), "CISPO as S": (-6, -12, "right"),
        "CISPO self-play": (6, 4, "left"),
    }
    scatter(untrained, OUT / "honesty_utility_untrained.pdf",
            "Untrained models (self-play, $T=1.0$)", label_offsets=untrained_off)
    scatter(trained, OUT / "honesty_utility_trained.pdf",
            "Qwen3-8B base vs joint-trained variants ($T=0.7$)", label_offsets=trained_off)
    scatter(trained_delta, OUT / "honesty_utility_trained_delta.pdf",
            "Trained-agent change vs base in matched cell", label_offsets=trained_delta_off,
            x_axis_label="$\\Delta$ honesty (trained $-$ base)",
            y_axis_label="$\\Delta$ normalised utility (trained $-$ base)", zero_lines=True)


if __name__ == "__main__":
    main()
