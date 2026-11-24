"""Offer dynamics of informed and uninformed agents (paper Appendix H.1, Table 16).

For the five zero-shot models in self-play, compares each agent's opening offer,
total concession and final offer in an asymmetric transparency condition with
the same role under full information.

Offers are expressed in surplus units from the offering agent's side:
    x = (offer - v_S) / (v_B - v_S)
    a_seller = x,   a_buyer = 1 - x
so a = 0.5 is the midpoint, higher is more self-favourable, and a > 1 lies
beyond the opponent's true reservation price. Values are clipped to [-3, 3].
Per agent: opening a (first offer), final a (last offer), concession
(opening - final). Differences vs full information use Welch t-tests.

Inputs, per model under ``results/zeroshot/<model>/data/``:
  - ``trials_processed.pkl``  reservation prices and transparency per trial
  - ``offers.jsonl``          one line per OFFER action: trial_id, round, actor, offer_price
    (numbers only; the dialogue logs are not shipped with the repository)

Usage:
    python analysis/offer_dynamics.py
        Print the tables (per model and pooled) used in the paper.
    python analysis/offer_dynamics.py --extract MODEL EXPERIMENT_DIR
        Rebuild results/zeroshot/MODEL/data/offers.jsonl from the full per-trial
        logs of a re-run (EXPERIMENT_DIR/logs/trial_*.jsonl, as written by the
        zero-shot notebooks under ./experiments/).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "zeroshot"
MODELS = [("claude-sonnet-4-6", "sonnet46"), ("claude-opus-4-7", "opus47"),
          ("GPT-5.2", "gpt52"), ("GPT-5.5", "gpt55"), ("Qwen3.5-9B", "qwen35")]
CLIP = 3.0


def extract(model: str, exp_dir: Path) -> None:
    trials = pd.read_pickle(exp_dir / "data" / "trials_processed.pkl")
    out = RESULTS / model / "data" / "offers.jsonl"
    n = 0
    with open(out, "w") as f:
        for _, t in trials.iterrows():
            for line in open(exp_dir / "logs" / Path(t.jsonl_path).name):
                e = json.loads(line)
                if e.get("type") == "summary" or e.get("action") != "OFFER":
                    continue
                f.write(json.dumps({"trial_id": str(t.trial_id), "round": int(e["round"]),
                                    "actor": e["actor"], "offer_price": e["offer_price"]}) + "\n")
                n += 1
    print(f"wrote {n} offers to {out}")


def load() -> pd.DataFrame:
    rows = []
    for label, model in MODELS:
        d = RESULTS / model / "data"
        trials = pd.read_pickle(d / "trials_processed.pkl")
        offers: dict[tuple[str, str], list[tuple[int, float]]] = {}
        for line in open(d / "offers.jsonl"):
            e = json.loads(line)
            try:
                offers.setdefault((e["trial_id"], e["actor"]), []).append(
                    (int(e["round"]), float(e["offer_price"])))
            except (TypeError, ValueError):
                pass
        for _, t in trials.iterrows():
            vb, vs = float(t.buyer_res_price), float(t.seller_res_price)
            rec = {"model": label, "transparency": t.transparency}
            for role in ("buyer", "seller"):
                o = sorted(offers.get((str(t.trial_id), role), []))
                if not o:
                    continue
                x = [(p - vs) / (vb - vs) for _, p in o]
                a = np.clip([xi if role == "seller" else 1 - xi for xi in x], -CLIP, CLIP)
                rec[f"{role}_open"], rec[f"{role}_final"] = a[0], a[-1]
                rec[f"{role}_conc"] = a[0] - a[-1]
            rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    D = load()
    blocks = [("Uninformed buyer (buyer-unaware)", "buyer", "buyer_unaware"),
              ("Uninformed seller (seller-unaware)", "seller", "seller_unaware"),
              ("Informed seller (buyer-unaware)", "seller", "buyer_unaware"),
              ("Informed buyer (seller-unaware)", "buyer", "seller_unaware")]
    for title, role, cond in blocks:
        print(f"\n{title}: difference vs the same role under full information")
        print(f"  {'model':18s} {'open':>14s} {'concession':>14s} {'final':>14s}")
        for label in [m for m, _ in MODELS] + ["Pooled"]:
            sub = D if label == "Pooled" else D[D.model == label]
            cells = []
            for metric in ("open", "conc", "final"):
                a = sub[sub.transparency == cond][f"{role}_{metric}"].dropna()
                b = sub[sub.transparency == "full"][f"{role}_{metric}"].dropna()
                p = stats.ttest_ind(a, b, equal_var=False).pvalue
                cells.append(f"{a.mean() - b.mean():+.3f} (p={p:.1g})")
            print(f"  {label:18s} " + " ".join(f"{c:>14s}" for c in cells))
    print("\nPooled full-information baselines:")
    full = D[D.transparency == "full"]
    for role in ("buyer", "seller"):
        print(f"  {role}: open {full[f'{role}_open'].mean():.2f}, "
              f"concession {full[f'{role}_conc'].mean():.2f}, final {full[f'{role}_final'].mean():.2f}")
    print("\nShare of opening offers beyond the opponent's true reservation (a > 1):")
    for title, role, cond in blocks[:2] + [("Buyer, full information", "buyer", "full"),
                                           ("Seller, full information", "seller", "full")]:
        v = D[D.transparency == cond][f"{role}_open"].dropna()
        print(f"  {title:36s} {(v > 1).mean():.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", nargs=2, metavar=("MODEL", "EXPERIMENT_DIR"))
    args = ap.parse_args()
    if args.extract:
        extract(args.extract[0], Path(args.extract[1]))
    else:
        main()
