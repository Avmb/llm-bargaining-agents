"""Val-curve + sample-trajectory driver for post-run analysis.

Given a running vLLM server (started with --enable-lora), for each configured
(config, LoRA checkpoint) condition this script:
  1. Hot-loads the LoRA into vLLM under the adapter name "trained".
  2. Runs full val-split evaluation, saves results DataFrame as
     {label}_val.pkl under --out-dir.
  3. Generates N full-episode sample trajectories on the first N val
     specs. Per-turn raw events are written by BargainingSimulator to
     {label}_trial_{i}.jsonl; a header summary per trial is concatenated
     into {label}_trajectories.jsonl.

Usage:
    python eval_checkpoints.py \
        --vllm-url http://localhost:8000/v1 \
        --out-dir  ./post_run_analysis \
        --n-trajectories 8
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import bargaining_rl as br
import requests
from urllib.parse import urlparse

logger = logging.getLogger("eval_checkpoints")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def swap_lora(vllm_url: str, lora_name: str, lora_path: str):
    """Load or replace a LoRA adapter in vLLM.

    Tries unload (ignore 4xx) then load. Avoids the /pause endpoint used by
    bargaining_rl._sync_lora_to_vllm, which doesn't exist on vanilla vLLM.
    """
    parsed = urlparse(vllm_url)
    base = f"{parsed.scheme}://{parsed.hostname}:{parsed.port or 8000}"
    abs_path = str(Path(lora_path).resolve())

    # Best-effort unload; fine if it's the first call and there's nothing to unload
    try:
        r = requests.post(f"{base}/v1/unload_lora_adapter",
                          json={"lora_name": lora_name}, timeout=30)
        if r.status_code >= 400:
            logger.info(f"  unload {lora_name}: {r.status_code} {r.text[:120]}")
    except Exception as e:
        logger.info(f"  unload {lora_name} failed (ignored): {e}")

    r = requests.post(f"{base}/v1/load_lora_adapter",
                      json={"lora_name": lora_name, "lora_path": abs_path},
                      timeout=120)
    r.raise_for_status()
    logger.info(f"  loaded LoRA '{lora_name}' from {abs_path}")


BASE = Path("experiments")  # <-- directory holding your training-run output dirs
CONDITIONS = {
    "thinking_step0": {
        "config": BASE / "configs" / "train_buyer_with_seller_qwen3.yaml",
        "lora":   BASE / "experiments" / "rl_train_2026-04-21_20-09-53" / "checkpoints" / "step_0",
    },
    "thinking_step17": {
        "config": BASE / "configs" / "train_buyer_with_seller_qwen3.yaml",
        "lora":   BASE / "experiments" / "rl_train_2026-04-21_20-09-53" / "checkpoints" / "step_17",
    },
    "nothink_step0": {
        "config": BASE / "configs" / "train_buyer_with_seller_qwen3_no_think.yaml",
        "lora":   BASE / "experiments" / "rl_train_2026-04-22_00-14-05" / "checkpoints" / "step_0",
    },
    "nothink_step44": {
        "config": BASE / "configs" / "train_buyer_with_seller_qwen3_no_think.yaml",
        "lora":   BASE / "experiments" / "rl_train_2026-04-22_00-14-05" / "checkpoints" / "step_44",
    },
}


def prepare_config(config_path: Path, vllm_url: str) -> br.Config:
    """Load a training config, retarget it for eval against the LoRA in vLLM."""
    cfg = br.load_config(str(config_path), [])
    cfg.buyer_model.base_url = vllm_url
    cfg.seller_model.base_url = vllm_url
    # The trained buyer is the "trained" LoRA adapter
    cfg.buyer_model.model_name = "trained"
    cfg.wandb.enabled = False
    cfg.mode = "eval"
    return cfg


async def sample_trajectories(task_specs, buyer_model, seller_model,
                               traj_header_path: Path):
    """Run N full episodes, writing per-turn raw logs + a header-summary file."""
    with open(traj_header_path, "w") as out:
        for i, spec in enumerate(task_specs):
            task = br.BargainingTask(
                item_name=spec["scenario"]["product_name"],
                item_description=spec["scenario"]["product_description"],
                buyer_persona=spec["scenario"]["buyer_persona"],
                seller_persona=spec["scenario"]["seller_persona"],
                buyer_res_price=spec["buyer_res_price"],
                seller_res_price=spec["seller_res_price"],
                buyer_res_price_range=spec["buyer_range"],
                seller_res_price_range=spec["seller_range"],
                transparency=spec["transparency"],
                mode=spec["mode"],
                max_rounds=spec["max_rounds"],
                first_actor=spec["first_actor"],
            )
            per_trial_log = traj_header_path.with_name(
                f"{traj_header_path.stem}_trial_{i+1}.jsonl"
            )
            sim = br.BargainingSimulator(
                task, buyer_model, seller_model, jsonl_path=str(per_trial_log)
            )
            summary = await sim.run()
            rewards = br.compute_normalized_rewards(summary, task)
            out.write(json.dumps({
                "trial_id": i + 1,
                "task": {
                    "item_name": task.item_name,
                    "buyer_res_price": task.buyer_res_price,
                    "seller_res_price": task.seller_res_price,
                    "transparency": task.transparency,
                    "max_rounds": task.max_rounds,
                    "first_actor": task.first_actor,
                },
                "summary": summary,
                "rewards": rewards,
                "per_turn_log": str(per_trial_log),
            }) + "\n")


def run_for_condition(name: str, cond: dict, vllm_url: str, out_dir: Path,
                      n_trajectories: int):
    logger.info(f"=== Condition {name} ===")
    logger.info(f"  LoRA:   {cond['lora']}")
    logger.info(f"  Config: {cond['config']}")

    swap_lora(vllm_url, "trained", str(cond["lora"]))

    config = prepare_config(cond["config"], vllm_url)
    splits = br.load_scenarios(config.data)
    val_scenarios = splits["val"]

    buyer_model = br.make_model_config(config.buyer_model)
    seller_model = br.make_model_config(config.seller_model)

    # 1. Full val eval
    val_df = asyncio.run(
        br.run_evaluation("val", val_scenarios, config, buyer_model, seller_model)
    )
    val_path = out_dir / f"{name}_val.pkl"
    val_df.to_pickle(val_path)
    logger.info(f"  val -> {val_path} (n={len(val_df)})")

    # 2. Sample trajectories on the first n_trajectories val trial specs
    specs = br.build_trial_specs(val_scenarios, config.bargaining, config.data.seed)
    specs = specs[:n_trajectories]
    traj_path = out_dir / f"{name}_trajectories.jsonl"
    asyncio.run(sample_trajectories(specs, buyer_model, seller_model, traj_path))
    logger.info(f"  trajectories -> {traj_path} (n={len(specs)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-url", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-trajectories", type=int, default=8)
    ap.add_argument("--conditions", default=",".join(CONDITIONS.keys()))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.conditions.split(","):
        if name not in CONDITIONS:
            logger.error(f"Unknown condition: {name}")
            sys.exit(1)
        run_for_condition(name, CONDITIONS[name], args.vllm_url, out_dir,
                          args.n_trajectories)

    logger.info("Done.")


if __name__ == "__main__":
    main()
