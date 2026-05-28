#!/usr/bin/env python3
"""
bargaining_rl.py — RL fine-tuning of LLM bargaining agents.

Phase 1: Evaluation of fixed models (no training).
Phase 2: GRPO training with Unsloth (custom training loop).

Usage:
    python bargaining_rl.py --config configs/eval_default.yaml --mode eval
    python bargaining_rl.py --config configs/eval_default.yaml --mode eval --buyer_model.model_name Qwen/Qwen3.5-32B
"""

import argparse
import asyncio
import json
import logging
import math
import random
import re
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import openai
import pandas as pd
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


# Code version label, stamped into each run's output_dir so results can be
# tied to the script that produced them. Bump this manually before any
# meaningful code change. Treat as free-text, not strict semver.
CODE_VERSION = "v8_opponent_temp"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelArgs:
    model_name: str = "Qwen/Qwen3-8B"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "unused"
    temperature: float = 1.0
    top_p: float = 0.95
    presence_penalty: float = 1.5
    max_completion_tokens: int = 4096
    top_k: int = 20
    enable_thinking: Optional[bool] = None


@dataclass
class SplitDataSource:
    """Data source for a single split (train/val/test)."""
    scenario_file: str = "../data/scenarios_by_reservation_ranges.jsonl"
    price_tier: str = "low"
    start_index: int = 0
    n_scenarios: int = 10


@dataclass
class DataConfig:
    train: SplitDataSource = field(default_factory=lambda: SplitDataSource(
        start_index=0, n_scenarios=28,
    ))
    val: SplitDataSource = field(default_factory=lambda: SplitDataSource(
        start_index=28, n_scenarios=6,
    ))
    test: SplitDataSource = field(default_factory=lambda: SplitDataSource(
        start_index=34, n_scenarios=6,
    ))
    seed: int = 42


@dataclass
class BargainingConfig:
    modes: List[str] = field(default_factory=lambda: ["simultaneous"])
    transparencies: List[str] = field(default_factory=lambda: ["both_unaware"])
    max_rounds: List[int] = field(default_factory=lambda: [6])
    first_actors: List[str] = field(default_factory=lambda: ["buyer"])
    n_trials_per_scenario: int = 4


@dataclass
class EvalConfig:
    batch_size: int = 16
    # When set, overrides BargainingConfig.n_trials_per_scenario for eval splits
    # (val/test) only. Lets us run far more eval trials than training rollouts.
    n_trials_per_scenario: Optional[int] = None


@dataclass
class TrainConfig:
    """On-policy GRPO training configuration."""
    train_role: str = "buyer"              # which role to train ("buyer" or "seller"); legacy single-role field
    # v7 joint self-play: if set, every rollout per scenario draws one role
    # from this list and the player is the drawn role. Both buyer and seller
    # share the same adapter (tc.lora_name). When unset, the legacy
    # train_role field is used and the model trains only the named role.
    train_roles: Optional[List[str]] = None
    # v8: optional overrides for the OPPONENT's sampling parameters. When set
    # in joint mode the opponent (whichever role is not the player in a given
    # rollout) is sampled with these params; the player still uses its own
    # role's buyer_model / seller_model settings. When unset, joint mode
    # falls back to the v7 behaviour (opponent uses the other role's
    # buyer_model / seller_model settings).
    opponent_temperature: Optional[float] = None
    opponent_top_p: Optional[float] = None
    learning_rate: float = 5e-6
    num_steps: int = 300                   # total training steps
    scenarios_per_step: int = 4            # scenarios per training step
    rl_group_size: int = 4                 # GRPO group size (episodes per scenario)
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    epochs_per_rollout: int = 1            # outer optimizer epochs per rollout batch
    optim_log_interval: int = 20           # log running pg/kl/grad_norm every N optimizer steps
    max_prompt_length: int = 2048
    max_completion_length: int = 2048
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    load_in_4bit: bool = True
    lora_name: str = "trained"             # LoRA adapter name in vLLM
    lora_target_modules: Optional[List[str]] = None  # model-specific, set in YAML
    # GRPO
    clip_eps: float = 0.2
    kl_coeff: float = 0.0                  # default off (following TRL convention)
    reward_transform: str = "none"         # within-group reward transform: "none" | "rank"
    # PPO importance-sampling correction (cache cur_lp on first visit so
    # the ratio uses the rollout-time policy rather than ref_lp). Off by
    # default so pre-v5 YAMLs reproduce v3/v4 behaviour exactly.
    is_correction: bool = False
    loss_type: str = "grpo"                # "grpo" | "cispo"
    epsilon_high: float = 5.0              # CISPO upper IS-weight clip
    # KL estimator. "weighted_legacy" = exp(ref_lp) * (ref_lp - cur_lp) (the
    # historical buggy form, kept as default for back-compat). "k3" =
    # Schulman's unbiased non-negative estimator exp(ref_lp - cur_lp) -
    # (ref_lp - cur_lp) - 1.
    kl_estimator: str = "weighted_legacy"
    # If True, the training reward is signed util/max_util (matches the eval
    # path). If False, it is clamped at max(0, util/max_util) (legacy
    # behaviour). signed_reward=True paired with reward_transform="rank" is
    # the recommended fix to give the optimizer a gradient away from
    # scale-bug deals.
    signed_reward: bool = False
    # Offer-magnitude sanity check applied during training rollouts. If
    # > 0, an OFFER with offer_price outside
    # [seller_res / mult, buyer_res * mult] is downgraded to INVALID. 0
    # disables the check (legacy).
    offer_sanity_mult: float = 0.0
    # Schedule
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    save_steps: int = 50
    eval_interval: int = 10                # validate every N steps
    logging_steps: int = 10


@dataclass
class WandbConfig:
    project: str = "bargaining-rl"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class Config:
    buyer_model: ModelArgs = field(default_factory=ModelArgs)
    seller_model: ModelArgs = field(default_factory=ModelArgs)
    data: DataConfig = field(default_factory=DataConfig)
    bargaining: BargainingConfig = field(default_factory=BargainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    output_dir: str = "./experiments"
    mode: str = "eval"
    debug: bool = False
    train_model_path: Optional[str] = None  # local path for Unsloth loading (if different from model_name)


def _merge_dict(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _merge_dict(base[k], v)
        else:
            base[k] = v
    return base


def _set_nested(d: dict, dotted_key: str, value: str):
    """Set d['a']['b'] from dotted key 'a.b'."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    # Try to parse as JSON for lists/numbers/bools, fall back to string
    try:
        d[keys[-1]] = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        d[keys[-1]] = value


def _resolve_type(f):
    """Resolve the concrete type for a dataclass field, handling Optional etc."""
    import dataclasses
    tp = f.type
    if hasattr(tp, "__dataclass_fields__"):
        return tp
    # For fields with default_factory producing a dataclass instance, infer from default
    if f.default_factory is not dataclasses.MISSING:
        try:
            sample = f.default_factory()
            if hasattr(type(sample), "__dataclass_fields__"):
                return type(sample)
        except Exception:
            pass
    return None


def _dict_to_dataclass(cls, d: dict):
    """Recursively convert a dict to a nested dataclass."""
    from dataclasses import fields as dc_fields
    kwargs = {}
    for f in dc_fields(cls):
        if f.name in d:
            val = d[f.name]
            nested_cls = _resolve_type(f)
            if nested_cls is not None and isinstance(val, dict):
                kwargs[f.name] = _dict_to_dataclass(nested_cls, val)
            else:
                kwargs[f.name] = val
    return cls(**kwargs)


def load_config(config_path: Optional[str], cli_overrides: List[str]) -> Config:
    """Load config from YAML file, then apply CLI overrides."""
    base = asdict(Config())

    if config_path:
        with open(config_path) as f:
            yaml_cfg = yaml.safe_load(f) or {}
        _merge_dict(base, yaml_cfg)

    # Parse --key.subkey value pairs
    it = iter(cli_overrides)
    for arg in it:
        if arg.startswith("--"):
            key = arg[2:]
            val = next(it)
            _set_nested(base, key, val)

    return _dict_to_dataclass(Config, base)


# ---------------------------------------------------------------------------
# Core bargaining abstractions
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    client: Any
    api_method: Callable
    use_system_message: bool
    args: Optional[Dict[str, Any]] = None


MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0


async def call_llm(model: ModelConfig, messages: List[Dict[str, str]]) -> Tuple[str, dict]:
    """Call LLM with retries and exponential backoff."""
    call_args = {"messages": messages}
    if model.args is not None:
        call_args.update(model.args)

    for attempt in range(MAX_RETRIES):
        try:
            response = await model.api_method(**call_args)
            text = response.choices[0].message.content
            usage = getattr(response, "usage", None) or {}
            return text, {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
            warnings.warn(f"call_llm attempt {attempt+1} failed: {e}. Retrying in {delay:.1f}s...")
            await asyncio.sleep(delay)


@dataclass
class BargainingTask:
    item_name: str
    item_description: str
    buyer_persona: str
    seller_persona: str
    buyer_res_price: float
    seller_res_price: float
    buyer_res_price_range: Optional[Tuple[float, float]] = None
    seller_res_price_range: Optional[Tuple[float, float]] = None
    transparency: str = "full"
    mode: str = "sequential"
    max_rounds: int = 5
    first_actor: str = "buyer"


def build_system_prompt(role: str, task: BargainingTask) -> str:
    prompt = f"""
You are the {role.upper()} in a bargaining negotiation.

Item: {task.item_name}
Description: {task.item_description}

Your persona: {task.buyer_persona if role=='buyer' else task.seller_persona}
Your reservation price: {task.buyer_res_price if role=='buyer' else task.seller_res_price}. You can always {'buy from' if (role=='buyer') else 'sell to'} the market at this price if the bargaining fails.
"""
    if task.transparency == "full":
        opp = task.seller_res_price if role == "buyer" else task.buyer_res_price
        prompt += f"You know the other agent's reservation price is {opp}.\n"
    elif task.transparency == "buyer_unaware":
        lo, hi = task.seller_res_price_range
        if role == "seller":
            prompt += f"You know the buyer's reservation price is {task.buyer_res_price}.\n"
            prompt += f"The buyer does not know your exact reservation price, their prior on your reservation price is ~ Uniform[{lo}, {hi}].\n"
        else:
            prompt += f"Your prior on the seller's reservation price is ~ Uniform[{lo}, {hi}].\n"
    elif task.transparency == "seller_unaware":
        lo, hi = task.buyer_res_price_range
        if role == "buyer":
            prompt += f"You know the seller's reservation price is {task.seller_res_price}.\n"
            prompt += f"The seller does not know your exact reservation price, their prior on your reservation price is ~ Uniform[{lo}, {hi}].\n"
        else:
            prompt += f"Your prior on the buyer's reservation price is ~ Uniform[{lo}, {hi}].\n"
    else:  # both_unaware
        lo_b, hi_b = task.buyer_res_price_range
        lo_s, hi_s = task.seller_res_price_range
        if role == "seller":
            prompt += f"Your prior on the buyer's reservation price is ~ Uniform[{lo_b}, {hi_b}].\n"
            prompt += f"The buyer does not know your exact reservation price, their prior on your reservation price is ~ Uniform[{lo_s}, {hi_s}].\n"
        else:
            prompt += f"Your prior on the seller's reservation price is ~ Uniform[{lo_s}, {hi_s}].\n"
            prompt += f"The seller does not know your exact reservation price, their prior on your reservation price is ~ Uniform[{lo_b}, {hi_b}].\n"

    prompt += f"""
Output format:

- Write 1-3 sentences describing your bargaining strategy. This will not be exposed to the other agent.
- Then output a JSON dict inside a code block using triple backticks:

```json
{{
  "message": "your message to the other agent",
  "action": "OFFER{' / DEAL' if (task.mode == 'sequential') else ''} / NO_DEAL",
  "offer_price": 123.45
}}
```
"""
    return prompt.strip()


def parse_output_json_block(text: str):
    """Parse LLM output: free-form thought + JSON code block."""
    thought = ""
    message = ""
    action = "INVALID"
    offer_price = None

    if not text:
        warnings.warn("Empty LLM output. Marking as INVALID.")
        return thought, message, action, offer_price

    json_block = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not json_block:
        warnings.warn("No JSON block found in LLM output. Marking as INVALID.")
        return text.strip(), message, action, offer_price

    json_text = json_block.group(1)
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        warnings.warn("JSON block could not be parsed. Marking as INVALID.")
        return text.split("```json")[0].strip(), message, action, offer_price

    thought = text.split("```json")[0].strip()
    message = data.get("message", "")
    action = data.get("action", "INVALID")
    offer_price = data.get("offer_price")

    valid_actions = {"OFFER", "DEAL", "NO_DEAL"}
    if action not in valid_actions:
        warnings.warn(f"Invalid action '{action}'. Marking as INVALID.")
        action = "INVALID"
        offer_price = None

    if action == "DEAL":
        offer_price = None

    if action != "OFFER" and offer_price is not None:
        offer_price = None

    if action == "OFFER":
        if offer_price is None or not isinstance(offer_price, (int, float)):
            warnings.warn(f"OFFER with invalid offer_price: {offer_price}. Marking as INVALID.")
            action = "INVALID"
            offer_price = None

    return thought, message, action, offer_price


def compute_game_theory_benchmarks(task: BargainingTask) -> dict:
    rB = task.buyer_res_price
    rS = task.seller_res_price
    true_nbs = (rB + rS) / 2
    if task.transparency == "full":
        expected_nbs = true_nbs
    else:
        BL, BH = task.buyer_res_price_range or (rB, rB)
        SL, SH = task.seller_res_price_range or (rS, rS)
        expected_nbs = (BL + BH + SL + SH) / 4.0
    return {
        "true_nbs_price": true_nbs,
        "expected_nbs_price": expected_nbs,
    }


class BargainingSimulator:
    """Multi-round LLM bargaining simulator."""

    def __init__(self, task, buyer_model, seller_model, jsonl_path=None):
        self.task = task
        self.buyer_model = buyer_model
        self.seller_model = seller_model
        self.jsonl_path = jsonl_path

        self.buyer_msgs = []
        self.seller_msgs = []
        self.log = []

        if self.jsonl_path:
            with open(self.jsonl_path, "w"):
                pass

        self.last_buyer_msg = None
        self.last_seller_msg = None
        self.last_buyer_offer = None
        self.last_seller_offer = None
        self.buyer_sys_sent = False
        self.seller_sys_sent = False
        self.total_invalid_actions = 0

        self.buyer_sys_msg = build_system_prompt("buyer", self.task)
        self.seller_sys_msg = build_system_prompt("seller", self.task)

    def log_event(self, event: dict):
        self.log.append(event)
        if self.jsonl_path:
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(event) + "\n")

    def _build_user_message(self, actor: str, round_num: int, rounds_left: int) -> str:
        content_parts = []
        opp_actor = "Seller" if actor == "buyer" else "Buyer"
        last_opp_msg = self.last_seller_msg if actor == "buyer" else self.last_buyer_msg
        last_opp_offer = self.last_seller_offer if actor == "buyer" else self.last_buyer_offer
        use_sys_msg = (self.buyer_model if actor == "buyer" else self.seller_model).use_system_message
        sys_msg = self.buyer_sys_msg if actor == "buyer" else self.seller_sys_msg
        sys_sent = self.buyer_sys_sent if actor == "buyer" else self.seller_sys_sent

        if not use_sys_msg and not sys_sent:
            content_parts.append(sys_msg + "\n")
            if actor == "buyer":
                self.buyer_sys_sent = True
            else:
                self.seller_sys_sent = True

        if last_opp_msg:
            content_parts.append(f"{opp_actor} said: {last_opp_msg}")
        if last_opp_offer:
            content_parts.append(f"{opp_actor}'s offer: {last_opp_offer}")
        content_parts.append(f"Round {round_num}, rounds left {rounds_left}")
        return "\n".join(content_parts)

    def _update_state(self, actor, thought, message, action, offer_price, usage, round_num):
        if actor == "buyer":
            self.last_buyer_msg = message
            if action == "OFFER":
                self.last_buyer_offer = offer_price
        else:
            self.last_seller_msg = message
            if action == "OFFER":
                self.last_seller_offer = offer_price

        if action == "INVALID":
            self.total_invalid_actions += 1

        self.log_event({
            "round": round_num, "actor": actor,
            "thought": thought, "message": message,
            "action": action, "offer_price": offer_price,
            "tokens": usage,
        })

    async def run(self) -> dict:
        self.buyer_msgs = (
            [{"role": "system", "content": self.buyer_sys_msg}]
            if self.buyer_model.use_system_message else []
        )
        self.seller_msgs = (
            [{"role": "system", "content": self.seller_sys_msg}]
            if self.seller_model.use_system_message else []
        )

        for r in range(1, self.task.max_rounds + 1):
            rounds_left = self.task.max_rounds - r

            if self.task.mode == "sequential":
                actor = (
                    self.task.first_actor if r % 2 == 1
                    else ("seller" if self.task.first_actor == "buyer" else "buyer")
                )
                msgs = self.buyer_msgs if actor == "buyer" else self.seller_msgs
                model = self.buyer_model if actor == "buyer" else self.seller_model

                user_content = self._build_user_message(actor, r, rounds_left)
                msgs.append({"role": "user", "content": user_content})

                raw, usage = await call_llm(model, msgs)
                msgs.append({"role": "assistant", "content": raw})

                thought, message, action, offer_price = parse_output_json_block(raw)
                self._update_state(actor, thought, message, action, offer_price, usage, r)

                if action == "DEAL":
                    deal_price = (
                        self.last_seller_offer if actor == "buyer"
                        else self.last_buyer_offer
                    )
                    return self._finalize(deal_price, r)
                if action == "NO_DEAL":
                    break

            else:  # simultaneous
                for actor in ["buyer", "seller"]:
                    msgs = self.buyer_msgs if actor == "buyer" else self.seller_msgs
                    user_content = self._build_user_message(actor, r, rounds_left)
                    msgs.append({"role": "user", "content": user_content})

                (b_raw, b_usage), (s_raw, s_usage) = await asyncio.gather(
                    call_llm(self.buyer_model, self.buyer_msgs),
                    call_llm(self.seller_model, self.seller_msgs),
                )

                self.buyer_msgs.append({"role": "assistant", "content": b_raw})
                self.seller_msgs.append({"role": "assistant", "content": s_raw})

                b_thought, b_msg, b_action, b_price = parse_output_json_block(b_raw)
                s_thought, s_msg, s_action, s_price = parse_output_json_block(s_raw)

                self._update_state("buyer", b_thought, b_msg, b_action, b_price, b_usage, r)
                self._update_state("seller", s_thought, s_msg, s_action, s_price, s_usage, r)

                if b_action == "OFFER" and s_action == "OFFER" and b_price >= s_price:
                    return self._finalize((b_price + s_price) / 2, r)
                if b_action == "NO_DEAL" or s_action == "NO_DEAL":
                    break

        return self._finalize(None, self.task.max_rounds)

    def _finalize(self, deal_price, rounds) -> dict:
        if deal_price is None:
            buyer_u = seller_u = 0.0
            result = "no_deal"
        else:
            buyer_u = self.task.buyer_res_price - deal_price
            seller_u = deal_price - self.task.seller_res_price
            result = "deal"

        benchmarks = compute_game_theory_benchmarks(self.task)
        summary = {
            "result": result,
            "deal_price": deal_price,
            "rounds": rounds,
            "buyer_utility": buyer_u,
            "seller_utility": seller_u,
            "total_invalid_actions": self.total_invalid_actions,
            **benchmarks,
        }
        self.log_event({"type": "summary", **summary})
        return summary


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def compute_normalized_rewards(summary: dict, task: BargainingTask) -> dict:
    """Compute normalized utility (reward) for each agent.

    Reward = utility / max_possible_utility, where max_utility = buyer_res - seller_res.
    If no zone of agreement (max_utility <= 0), reward is 0.
    If no deal, reward is 0.
    """
    max_utility = task.buyer_res_price - task.seller_res_price
    has_zoa = max_utility > 0

    if not has_zoa or summary["result"] != "deal":
        return {"buyer_reward": 0.0, "seller_reward": 0.0, "has_zoa": has_zoa}

    return {
        "buyer_reward": summary["buyer_utility"] / max_utility,
        "seller_reward": summary["seller_utility"] / max_utility,
        "has_zoa": has_zoa,
    }


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------

def _load_split_scenarios(source: SplitDataSource) -> list:
    """Load scenarios for a single split from its data source config."""
    with open(source.scenario_file) as f:
        all_scenarios = json.load(f)

    tier_scenarios = all_scenarios[source.price_tier]
    end = source.start_index + source.n_scenarios
    selected = tier_scenarios[source.start_index : end]

    for sc in selected:
        sc["price_tier"] = source.price_tier

    return selected


def load_scenarios(config: DataConfig) -> dict:
    """Load scenarios for each split from its configured data source."""
    splits = {}
    for split_name in ["train", "val", "test"]:
        source: SplitDataSource = getattr(config, split_name)
        scenarios = _load_split_scenarios(source)
        splits[split_name] = scenarios
        logger.info(
            f"  {split_name}: {len(scenarios)} scenarios from "
            f"{source.scenario_file} [{source.price_tier}][{source.start_index}:{source.start_index + source.n_scenarios}]"
        )

    return splits


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def build_trial_specs(scenarios: list, bargaining_config: BargainingConfig, seed: int,
                      train_roles: Optional[List[str]] = None) -> list:
    """Build the full grid of trial specifications.

    train_roles: list of player roles to enumerate per (scenario, transparency)
    cell. None or length-1 -> single role assigned to every spec (legacy
    single-side eval). Length-2 (v7 joint) -> each cell is enumerated for
    both roles, so the produced spec list is 2x the legacy count.
    """
    rng = random.Random(seed)
    roles = list(train_roles) if train_roles else [None]
    specs = []

    for sc_idx, scenario in enumerate(scenarios):
        for mode in bargaining_config.modes:
            for transparency in bargaining_config.transparencies:
                for max_rounds in bargaining_config.max_rounds:
                    for first_actor in bargaining_config.first_actors:
                        if mode == "simultaneous" and first_actor != "buyer":
                            continue
                        for role in roles:
                            for _ in range(bargaining_config.n_trials_per_scenario):
                                buyer_range = tuple(scenario["buyer_res_price_range"])
                                seller_range = tuple(scenario["seller_res_price_range"])
                                b_price = round(rng.uniform(*buyer_range), 2)
                                s_price = round(rng.uniform(*seller_range), 2)

                                specs.append({
                                    "scenario_idx": sc_idx,
                                    "scenario": scenario,
                                    "mode": mode,
                                    "transparency": transparency,
                                    "max_rounds": max_rounds,
                                    "first_actor": first_actor,
                                    "buyer_res_price": b_price,
                                    "seller_res_price": s_price,
                                    "buyer_range": buyer_range,
                                    "seller_range": seller_range,
                                    "train_role": role,
                                })
    return specs


async def run_single_trial(
    spec: dict,
    buyer_model: ModelConfig,
    seller_model: ModelConfig,
    trial_id: int,
) -> dict:
    """Run one bargaining trial and return results with rewards."""
    task = BargainingTask(
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

    sim = BargainingSimulator(task, buyer_model, seller_model)
    summary = await sim.run()
    rewards = compute_normalized_rewards(summary, task)

    return {
        "trial_id": trial_id,
        "item_name": task.item_name,
        "price_tier": spec["scenario"].get("price_tier", "unknown"),
        "mode": task.mode,
        "transparency": task.transparency,
        "max_rounds": task.max_rounds,
        "first_actor": task.first_actor,
        "train_role": spec.get("train_role"),
        "buyer_res_price": task.buyer_res_price,
        "seller_res_price": task.seller_res_price,
        "has_zoa": rewards["has_zoa"],
        "result": summary["result"],
        "deal_price": summary["deal_price"],
        "rounds": summary["rounds"],
        "buyer_utility": summary["buyer_utility"],
        "seller_utility": summary["seller_utility"],
        "buyer_reward": rewards["buyer_reward"],
        "seller_reward": rewards["seller_reward"],
        "total_invalid_actions": summary["total_invalid_actions"],
        "true_nbs_price": summary["true_nbs_price"],
        "expected_nbs_price": summary["expected_nbs_price"],
    }


async def run_evaluation(
    split_name: str,
    scenarios: list,
    config: Config,
    buyer_model: ModelConfig,
    seller_model: ModelConfig,
    step: int = 0,
) -> pd.DataFrame:
    """Run evaluation on a data split. Returns DataFrame of trial results."""
    bc = config.bargaining
    if config.eval.n_trials_per_scenario is not None:
        bc = replace(bc, n_trials_per_scenario=config.eval.n_trials_per_scenario)
    # For v7 joint training pass both roles so val enumerates each cell for
    # both player perspectives. For single-side legacy runs the role list is
    # length-1 and behaviour matches v6.
    tc = config.train
    effective_roles = list(tc.train_roles) if tc.train_roles else [tc.train_role]
    specs = build_trial_specs(scenarios, bc, config.data.seed, train_roles=effective_roles)
    logger.info(f"[{split_name}] Running {len(specs)} trials (batch_size={config.eval.batch_size})")

    semaphore = asyncio.Semaphore(config.eval.batch_size)
    results = []
    pbar = tqdm(total=len(specs), desc=f"Eval ({split_name})")

    async def bounded_trial(spec, tid):
        async with semaphore:
            result = await run_single_trial(spec, buyer_model, seller_model, tid)
            pbar.update(1)
            return result

    tasks = [bounded_trial(spec, i + 1) for i, spec in enumerate(specs)]
    results = await asyncio.gather(*tasks)

    pbar.close()
    df = pd.DataFrame(results)

    # Log metrics
    log_eval_metrics(df, split_name, step, config)

    return df


# ---------------------------------------------------------------------------
# Metrics and wandb logging
# ---------------------------------------------------------------------------

def log_eval_metrics(df: pd.DataFrame, split: str, step: int, config: Config):
    """Compute and log evaluation metrics."""
    metrics = {}

    # Overall metrics
    metrics[f"{split}/n_trials"] = len(df)
    metrics[f"{split}/deal_rate"] = (df["result"] == "deal").mean()
    metrics[f"{split}/avg_rounds"] = df["rounds"].mean()
    metrics[f"{split}/invalid_action_rate"] = (
        df["total_invalid_actions"].sum()
        / (df["rounds"].sum() * 2)  # 2 agents per round
        if df["rounds"].sum() > 0 else 0
    )

    # Reward metrics
    metrics[f"{split}/buyer_reward_mean"] = df["buyer_reward"].mean()
    metrics[f"{split}/buyer_reward_std"] = df["buyer_reward"].std()
    metrics[f"{split}/seller_reward_mean"] = df["seller_reward"].mean()
    metrics[f"{split}/seller_reward_std"] = df["seller_reward"].std()
    metrics[f"{split}/total_reward_mean"] = (df["buyer_reward"] + df["seller_reward"]).mean()

    # Utility metrics
    metrics[f"{split}/buyer_utility_mean"] = df["buyer_utility"].mean()
    metrics[f"{split}/seller_utility_mean"] = df["seller_utility"].mean()
    metrics[f"{split}/total_welfare_mean"] = (df["buyer_utility"] + df["seller_utility"]).mean()

    # NBS deviation (deals only)
    deals = df[df["result"] == "deal"]
    if len(deals) > 0:
        metrics[f"{split}/nbs_deviation_mean"] = (
            (deals["deal_price"] - deals["true_nbs_price"]).abs().mean()
        )

    # ZOA breakdown
    zoa = df[df["has_zoa"]]
    no_zoa = df[~df["has_zoa"]]
    if len(zoa) > 0:
        metrics[f"{split}/deal_rate_zoa"] = (zoa["result"] == "deal").mean()
        metrics[f"{split}/buyer_reward_mean_zoa"] = zoa["buyer_reward"].mean()
        metrics[f"{split}/seller_reward_mean_zoa"] = zoa["seller_reward"].mean()
    if len(no_zoa) > 0:
        metrics[f"{split}/deal_rate_no_zoa"] = (no_zoa["result"] == "deal").mean()

    # Per-transparency breakdown
    for transp, group in df.groupby("transparency"):
        metrics[f"{split}/deal_rate/{transp}"] = (group["result"] == "deal").mean()
        metrics[f"{split}/buyer_reward_mean/{transp}"] = group["buyer_reward"].mean()
        metrics[f"{split}/seller_reward_mean/{transp}"] = group["seller_reward"].mean()

    # Per-(transparency, train_role) breakdown for v7 joint self-play.
    # In joint mode train_role is the player-perspective label assigned to
    # the trial; player_reward picks the right side. In legacy single-role
    # mode train_role is constant and these duplicate the per-transp keys
    # with the role suffix, which is fine.
    if "train_role" in df.columns and df["train_role"].notna().any():
        df_role = df[df["train_role"].notna()].copy()
        df_role["player_reward"] = np.where(
            df_role["train_role"] == "buyer",
            df_role["buyer_reward"],
            df_role["seller_reward"],
        )
        for (transp, role), group in df_role.groupby(["transparency", "train_role"]):
            key_prefix = f"{split}/{role}_player"
            metrics[f"{key_prefix}/deal_rate/{transp}"] = (group["result"] == "deal").mean()
            metrics[f"{key_prefix}/player_reward_mean/{transp}"] = group["player_reward"].mean()
            metrics[f"{key_prefix}/buyer_reward_mean/{transp}"] = group["buyer_reward"].mean()
            metrics[f"{key_prefix}/seller_reward_mean/{transp}"] = group["seller_reward"].mean()

    # Per-price-tier breakdown
    for tier, group in df.groupby("price_tier"):
        metrics[f"{split}/deal_rate/{tier}"] = (group["result"] == "deal").mean()
        metrics[f"{split}/buyer_reward_mean/{tier}"] = group["buyer_reward"].mean()
        metrics[f"{split}/seller_reward_mean/{tier}"] = group["seller_reward"].mean()

    # Print summary
    logger.info(
        f"[{split}] deal_rate={metrics[f'{split}/deal_rate']:.3f} "
        f"buyer_r={metrics[f'{split}/buyer_reward_mean']:.3f} "
        f"seller_r={metrics[f'{split}/seller_reward_mean']:.3f} "
        f"total_r={metrics[f'{split}/total_reward_mean']:.3f} "
        f"avg_rounds={metrics[f'{split}/avg_rounds']:.1f} "
        f"invalid_rate={metrics[f'{split}/invalid_action_rate']:.3f}"
    )

    if config.wandb.enabled:
        try:
            import wandb
            metrics["train/step"] = step
            wandb.log(metrics)

            # Log trial-level table
            table_cols = [
                "trial_id", "item_name", "price_tier", "transparency",
                "buyer_res_price", "seller_res_price", "has_zoa",
                "result", "deal_price", "rounds",
                "buyer_reward", "seller_reward",
            ]
            wandb.log({
                f"{split}/trials": wandb.Table(dataframe=df[table_cols]),
                "train/step": step,
            })
        except Exception as e:
            logger.warning(f"Failed to log to wandb: {e}")


# ---------------------------------------------------------------------------
# Training: GRPO with TRL + Unsloth (multi-turn)
#
# Approach: for each training epoch we generate full bargaining episodes via
# vLLM, then unroll every turn of the trained agent into a separate
# (prompt, completion, reward) sample.  All turns in an episode share the
# same episode-level reward (Monte-Carlo return).  GRPOTrainer receives
# these samples with a passthrough reward function that returns the
# pre-computed rewards.
# ---------------------------------------------------------------------------

def _call_vllm_sync(client: "openai.OpenAI", model_args: ModelArgs,
                     messages: list) -> str:
    """Synchronous LLM call to a vLLM endpoint with retries."""
    call_args = {
        "model": model_args.model_name,
        "temperature": model_args.temperature,
        "top_p": model_args.top_p,
        "presence_penalty": model_args.presence_penalty,
        "max_completion_tokens": model_args.max_completion_tokens,
        "messages": messages,
    }
    extra_body: dict = {}
    if model_args.top_k and model_args.top_k > 0:
        extra_body["top_k"] = model_args.top_k
    if model_args.enable_thinking is not None:
        extra_body["chat_template_kwargs"] = {"enable_thinking": bool(model_args.enable_thinking)}
    if extra_body:
        call_args["extra_body"] = extra_body

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(**call_args)
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1))


async def _call_vllm_async(client: "openai.AsyncOpenAI", model_args: ModelArgs,
                           messages: list, return_reasoning: bool = False):
    """Async LLM call to a vLLM endpoint with retries.

    If return_reasoning is True, returns (content, reasoning_content); otherwise
    returns content only. reasoning_content is "" when not present on the response.
    """
    call_args = {
        "model": model_args.model_name,
        "temperature": model_args.temperature,
        "top_p": model_args.top_p,
        "presence_penalty": model_args.presence_penalty,
        "max_completion_tokens": model_args.max_completion_tokens,
        "messages": messages,
    }
    extra_body: dict = {}
    if model_args.top_k and model_args.top_k > 0:
        extra_body["top_k"] = model_args.top_k
    if model_args.enable_thinking is not None:
        extra_body["chat_template_kwargs"] = {"enable_thinking": bool(model_args.enable_thinking)}
    if extra_body:
        call_args["extra_body"] = extra_body

    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.chat.completions.create(**call_args)
            msg = resp.choices[0].message
            content = msg.content or ""
            if return_reasoning:
                reasoning = getattr(msg, "reasoning", None) or ""
                return content, reasoning
            return content
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1))


def _agent_reward(task: BargainingTask, role: str, deal_price: float,
                  signed: bool = False) -> float:
    """Normalised reward for one agent given a deal price.

    By default returns max(0, util/max_util) (legacy training behaviour,
    which clamps deals outside the ZOA to zero). When signed=True returns
    the signed util/max_util and matches the eval-side
    compute_normalized_rewards. The signed form gives the optimizer a
    gradient away from scale-bug deals (deals at price >> buyer_res, where
    buyer_utility is large negative). Pair signed=True with
    reward_transform="rank" to keep advantage magnitudes bounded.
    """
    max_util = task.buyer_res_price - task.seller_res_price
    if max_util <= 0:
        return 0.0
    util = (task.buyer_res_price - deal_price) if role == "buyer" else (deal_price - task.seller_res_price)
    r = util / max_util
    return r if signed else max(0.0, r)


def _is_offer_in_sane_range(offer: Optional[float], task: BargainingTask,
                            mult: float) -> bool:
    """True if an OFFER's offer_price is within sanity bounds.

    Used by the training rollout to downgrade malformed offers (e.g. the
    100x JSON scale-hallucination 150 instead of 1.50) to INVALID before
    they reach the simulator's deal-detection logic.

    mult <= 0 disables the check (returns True). Otherwise the offer must
    satisfy seller_res / mult <= offer <= buyer_res * mult.
    """
    if mult <= 0 or offer is None:
        return True
    if offer <= 0:
        return False
    return (task.seller_res_price / mult) <= offer <= (task.buyer_res_price * mult)


def _rank_transform_rewards(rewards: List[float]) -> List[float]:
    """Average-rank transform within a group. Best=N, worst=1, ties get the mean rank.
    E.g., [0.31, 0.5, 0.31, 0.2] -> [2.5, 4.0, 2.5, 1.0]."""
    arr = np.asarray(rewards, dtype=float)
    n = arr.shape[0]
    if n == 0:
        return []
    order = np.argsort(arr, kind="stable")
    out = np.empty(n, dtype=float)
    out[order] = np.arange(1, n + 1, dtype=float)
    # Average ranks across ties.
    sorted_vals = arr[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[j]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            for k in range(i, j + 1):
                out[order[k]] = avg
        i = j + 1
    return out.tolist()


def generate_episode(task: BargainingTask, train_role: str,
                     trained_call_fn, opponent_call_fn,
                     executor=None) -> dict:
    """Generate a full bargaining episode via vLLM.

    Matches the behaviour of ``BargainingSimulator.run()`` — INVALID actions
    are ignored (the round continues), and in simultaneous mode a deal is only
    detected when *both* agents OFFER in the same round with buyer >= seller.

    Args:
        trained_call_fn: LLM call function for the trained agent (uses LoRA).
        opponent_call_fn: LLM call function for the opponent (uses base model).
        executor: optional ``concurrent.futures.ThreadPoolExecutor`` used to
            parallelise independent LLM calls within a round (simultaneous
            mode).  If *None*, calls are made sequentially.

    Returns a dict with:
      - "reward": float, normalised reward for ``train_role``
      - "turns": list of dicts, one per turn of the trained agent, each with
            "prompt" (list of message dicts up to this turn) and
            "completion" (the agent's response text).
      - "log": list of per-round event dicts (for debug logging).
    """
    opponent_role = "seller" if train_role == "buyer" else "buyer"

    trained_sys = build_system_prompt(train_role, task)
    opp_sys = build_system_prompt(opponent_role, task)

    trained_msgs: list = []
    opp_msgs: list = []
    trained_sys_sent = False
    opp_sys_sent = False

    last_msg = {train_role: None, opponent_role: None}
    last_offer: dict = {train_role: None, opponent_role: None}

    turns = []  # trained agent's (prompt, completion) pairs
    event_log = []  # per-round debug log

    def _build_user_msg(role, round_num, rounds_left):
        nonlocal trained_sys_sent, opp_sys_sent
        other = opponent_role if role == train_role else train_role
        opp_name = "Seller" if role == "buyer" else "Buyer"
        parts = []
        sys_p = trained_sys if role == train_role else opp_sys
        sent = trained_sys_sent if role == train_role else opp_sys_sent
        if not sent:
            parts.append(sys_p + "\n")
            if role == train_role:
                trained_sys_sent = True
            else:
                opp_sys_sent = True
        if last_msg[other]:
            parts.append(f"{opp_name} said: {last_msg[other]}")
        if last_offer[other]:
            parts.append(f"{opp_name}'s offer: {last_offer[other]}")
        parts.append(f"Round {round_num}, rounds left {rounds_left}")
        return "\n".join(parts)

    def _process_response(role, raw):
        _, msg, action, offer = parse_output_json_block(raw)
        last_msg[role] = msg
        if action == "OFFER":
            last_offer[role] = offer
        return msg, action, offer

    def _finalise(deal_price, rounds):
        reward = (
            _agent_reward(task, train_role, deal_price)
            if deal_price is not None else 0.0
        )
        return {"reward": reward, "turns": turns, "log": event_log,
                "deal_price": deal_price, "rounds": rounds}

    if task.mode == "simultaneous":
        for r in range(1, task.max_rounds + 1):
            rl = task.max_rounds - r

            # Build user messages for both agents
            t_user = _build_user_msg(train_role, r, rl)
            o_user = _build_user_msg(opponent_role, r, rl)
            trained_msgs.append({"role": "user", "content": t_user})
            opp_msgs.append({"role": "user", "content": o_user})

            # Snapshot the prompt *before* the assistant reply
            prompt_snapshot = list(trained_msgs)

            # Generate responses via vLLM (parallel within a round)
            if executor is not None:
                t_fut = executor.submit(trained_call_fn, list(trained_msgs))
                o_fut = executor.submit(opponent_call_fn, list(opp_msgs))
                t_raw = t_fut.result()
                o_raw = o_fut.result()
            else:
                t_raw = trained_call_fn(trained_msgs)
                o_raw = opponent_call_fn(opp_msgs)
            trained_msgs.append({"role": "assistant", "content": t_raw})
            opp_msgs.append({"role": "assistant", "content": o_raw})

            t_msg, t_action, t_offer = _process_response(train_role, t_raw)
            o_msg, o_action, o_offer = _process_response(opponent_role, o_raw)

            turns.append({"prompt": prompt_snapshot, "completion": t_raw})
            event_log.append({
                "round": r,
                train_role: {"message": t_msg, "action": t_action, "offer": t_offer,
                             "raw": t_raw},
                opponent_role: {"message": o_msg, "action": o_action, "offer": o_offer,
                                "raw": o_raw},
            })

            # NO_DEAL from either side ends the episode
            if t_action == "NO_DEAL" or o_action == "NO_DEAL":
                return _finalise(None, r)

            # Deal only when BOTH agents OFFER this round and buyer >= seller
            if (t_action == "OFFER" and o_action == "OFFER"
                    and t_offer is not None and o_offer is not None):
                b_offer = t_offer if train_role == "buyer" else o_offer
                s_offer = o_offer if train_role == "buyer" else t_offer
                if b_offer >= s_offer:
                    return _finalise((b_offer + s_offer) / 2, r)

        return _finalise(None, task.max_rounds)

    else:  # sequential
        for r in range(1, task.max_rounds + 1):
            actor = task.first_actor if r % 2 == 1 else (
                "seller" if task.first_actor == "buyer" else "buyer"
            )
            rl = task.max_rounds - r

            if actor == train_role:
                user_content = _build_user_msg(train_role, r, rl)
                trained_msgs.append({"role": "user", "content": user_content})
                prompt_snapshot = list(trained_msgs)

                raw = trained_call_fn(trained_msgs)
                trained_msgs.append({"role": "assistant", "content": raw})
                msg, action, offer = _process_response(train_role, raw)
                turns.append({"prompt": prompt_snapshot, "completion": raw})
            else:
                user_content = _build_user_msg(opponent_role, r, rl)
                opp_msgs.append({"role": "user", "content": user_content})
                raw = opponent_call_fn(opp_msgs)
                opp_msgs.append({"role": "assistant", "content": raw})
                msg, action, offer = _process_response(opponent_role, raw)

            event_log.append({
                "round": r, "actor": actor,
                "message": msg, "action": action, "offer": offer, "raw": raw,
            })

            if action == "DEAL":
                other = opponent_role if actor == train_role else train_role
                dp = last_offer[other]
                return _finalise(dp, r)
            if action == "NO_DEAL":
                return _finalise(None, r)

        return _finalise(None, task.max_rounds)


async def generate_episode_async(
    task: BargainingTask, train_role: str,
    trained_call_fn, opponent_call_fn,
    signed_reward: bool = False,
    offer_sanity_mult: float = 0.0,
) -> dict:
    """Async version of generate_episode for use with asyncio.gather.

    trained_call_fn must return (content, reasoning_content) tuple.
    opponent_call_fn returns content only.

    signed_reward forwards to _agent_reward; see its docstring.
    offer_sanity_mult forwards to _is_offer_in_sane_range; if positive, an
    OFFER from either agent with offer_price outside the sanity range is
    downgraded to INVALID before the deal-detection logic sees it.

    Per-turn records in "turns" contain:
      - "prompt_msgs": snapshot of the public history (no think) up to this turn
      - "content": the text content after </think> (what the opponent sees)
      - "reasoning": the text inside <think>...</think>, or "" if none
    """
    opponent_role = "seller" if train_role == "buyer" else "buyer"

    trained_sys = build_system_prompt(train_role, task)
    opp_sys = build_system_prompt(opponent_role, task)

    trained_msgs: list = []  # public history (no think) — used for next-turn prompt
    opp_msgs: list = []
    trained_sys_sent = False
    opp_sys_sent = False

    last_msg = {train_role: None, opponent_role: None}
    last_offer: dict = {train_role: None, opponent_role: None}

    turns = []
    event_log = []
    first_trained_offer: list = [None]  # [float|None]; set on first trained OFFER

    def _build_user_msg(role, round_num, rounds_left):
        nonlocal trained_sys_sent, opp_sys_sent
        other = opponent_role if role == train_role else train_role
        opp_name = "Seller" if role == "buyer" else "Buyer"
        parts = []
        sys_p = trained_sys if role == train_role else opp_sys
        sent = trained_sys_sent if role == train_role else opp_sys_sent
        if not sent:
            parts.append(sys_p + "\n")
            if role == train_role:
                trained_sys_sent = True
            else:
                opp_sys_sent = True
        if last_msg[other]:
            parts.append(f"{opp_name} said: {last_msg[other]}")
        if last_offer[other]:
            parts.append(f"{opp_name}'s offer: {last_offer[other]}")
        parts.append(f"Round {round_num}, rounds left {rounds_left}")
        return "\n".join(parts)

    def _process_response(role, raw):
        _, msg, action, offer = parse_output_json_block(raw)
        if action == "OFFER" and not _is_offer_in_sane_range(offer, task, offer_sanity_mult):
            action, offer = "INVALID", None
        last_msg[role] = msg
        if action == "OFFER":
            last_offer[role] = offer
            if role == train_role and first_trained_offer[0] is None and offer is not None:
                first_trained_offer[0] = offer
        return msg, action, offer

    def _finalise(deal_price, rounds):
        reward = (
            _agent_reward(task, train_role, deal_price, signed=signed_reward)
            if deal_price is not None else 0.0
        )
        return {"reward": reward, "turns": turns, "log": event_log,
                "deal_price": deal_price, "rounds": rounds,
                "first_trained_offer": first_trained_offer[0],
                "buyer_res_price": task.buyer_res_price,
                "seller_res_price": task.seller_res_price}

    if task.mode == "simultaneous":
        for r in range(1, task.max_rounds + 1):
            rl = task.max_rounds - r

            t_user = _build_user_msg(train_role, r, rl)
            o_user = _build_user_msg(opponent_role, r, rl)
            trained_msgs.append({"role": "user", "content": t_user})
            opp_msgs.append({"role": "user", "content": o_user})

            prompt_snapshot = list(trained_msgs)

            t_result, o_raw = await asyncio.gather(
                trained_call_fn(list(trained_msgs)),
                opponent_call_fn(list(opp_msgs)),
            )
            t_content, t_reasoning = t_result
            trained_msgs.append({"role": "assistant", "content": t_content})
            opp_msgs.append({"role": "assistant", "content": o_raw})

            t_msg, t_action, t_offer = _process_response(train_role, t_content)
            o_msg, o_action, o_offer = _process_response(opponent_role, o_raw)

            turns.append({
                "prompt_msgs": prompt_snapshot,
                "content": t_content,
                "reasoning": t_reasoning,
            })
            event_log.append({
                "round": r,
                train_role: {"message": t_msg, "action": t_action, "offer": t_offer,
                             "raw": t_content, "reasoning": t_reasoning},
                opponent_role: {"message": o_msg, "action": o_action, "offer": o_offer,
                                "raw": o_raw},
            })

            if t_action == "NO_DEAL" or o_action == "NO_DEAL":
                return _finalise(None, r)

            if (t_action == "OFFER" and o_action == "OFFER"
                    and t_offer is not None and o_offer is not None):
                b_offer = t_offer if train_role == "buyer" else o_offer
                s_offer = o_offer if train_role == "buyer" else t_offer
                if b_offer >= s_offer:
                    return _finalise((b_offer + s_offer) / 2, r)

        return _finalise(None, task.max_rounds)

    else:  # sequential
        for r in range(1, task.max_rounds + 1):
            actor = task.first_actor if r % 2 == 1 else (
                "seller" if task.first_actor == "buyer" else "buyer"
            )
            rl = task.max_rounds - r

            if actor == train_role:
                user_content = _build_user_msg(train_role, r, rl)
                trained_msgs.append({"role": "user", "content": user_content})
                prompt_snapshot = list(trained_msgs)

                t_content, t_reasoning = await trained_call_fn(trained_msgs)
                trained_msgs.append({"role": "assistant", "content": t_content})
                msg, action, offer = _process_response(train_role, t_content)
                raw = t_content
                turns.append({
                    "prompt_msgs": prompt_snapshot,
                    "content": t_content,
                    "reasoning": t_reasoning,
                })
            else:
                user_content = _build_user_msg(opponent_role, r, rl)
                opp_msgs.append({"role": "user", "content": user_content})
                raw = await opponent_call_fn(opp_msgs)
                opp_msgs.append({"role": "assistant", "content": raw})
                msg, action, offer = _process_response(opponent_role, raw)

            event_log.append({
                "round": r, "actor": actor,
                "message": msg, "action": action, "offer": offer, "raw": raw,
            })

            if action == "DEAL":
                other = opponent_role if actor == train_role else train_role
                dp = last_offer[other]
                return _finalise(dp, r)
            if action == "NO_DEAL":
                return _finalise(None, r)

        return _finalise(None, task.max_rounds)


def _estimate_rollout_requests(scenarios, bargaining_config, rl_group_size):
    """Estimate the number of LLM requests for a rollout.

    Returns (n_episodes, est_requests, avg_rounds).
    Each simultaneous round needs 2 calls; each sequential round needs 1.
    We assume episodes run to max_rounds (upper bound).
    """
    n_episodes = 0
    est_requests = 0
    total_rounds = 0
    for scenario in scenarios:
        for transparency in bargaining_config.transparencies:
            for max_rounds in bargaining_config.max_rounds:
                for mode in bargaining_config.modes:
                    n_group = bargaining_config.n_trials_per_scenario * rl_group_size
                    n_episodes += n_group
                    calls_per_round = 2 if mode == "simultaneous" else 1
                    est_requests += n_group * max_rounds * calls_per_round
                    total_rounds += n_group * max_rounds
    avg_rounds = total_rounds / max(n_episodes, 1)
    return n_episodes, est_requests, avg_rounds


def rollout_episodes(scenarios, train_role, bargaining_config, call_fn, seed,
                     rl_group_size=4, max_workers=16,
                     wandb_prefix="train", wandb_enabled=False,
                     debug_dir=None, logging_steps=1):
    """Generate episodes and unroll into per-turn training samples.

    Episodes are generated concurrently via a thread pool so that vLLM can
    batch requests on the GPU.  Progress is logged to wandb every 10% of
    episodes.

    For each scenario configuration we generate ``rl_group_size`` episodes
    (the GRPO "group").  Each episode is unrolled so every turn of the
    trained agent becomes a training sample.  All samples within an episode
    share the same reward; all episodes within a group share a ``group_id``
    for GRPO advantage computation.

    Returns a list of dicts ready for training.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Estimate work ahead of time
    n_episodes, est_requests, avg_rounds = _estimate_rollout_requests(
        scenarios, bargaining_config, rl_group_size)
    logger.info(
        f"Rollout plan: {n_episodes} episodes, ~{est_requests} LLM requests "
        f"(upper bound, avg {avg_rounds:.0f} rounds/episode), "
        f"{max_workers} workers"
    )

    rng = random.Random(seed)

    # Pre-build all (task, group_id) pairs so the RNG is consumed
    # deterministically before we fan out to threads.
    jobs = []   # list of (task, group_id)
    group_id = 0

    for scenario in scenarios:
        for transparency in bargaining_config.transparencies:
            for max_rounds in bargaining_config.max_rounds:
                for mode in bargaining_config.modes:
                    for _ in range(bargaining_config.n_trials_per_scenario):
                        buyer_range = tuple(scenario["buyer_res_price_range"])
                        seller_range = tuple(scenario["seller_res_price_range"])
                        b_price = round(rng.uniform(*buyer_range), 2)
                        s_price = round(rng.uniform(*seller_range), 2)

                        task = BargainingTask(
                            item_name=scenario["product_name"],
                            item_description=scenario["product_description"],
                            buyer_persona=scenario["buyer_persona"],
                            seller_persona=scenario["seller_persona"],
                            buyer_res_price=b_price,
                            seller_res_price=s_price,
                            buyer_res_price_range=buyer_range,
                            seller_res_price_range=seller_range,
                            transparency=transparency,
                            mode=mode,
                            max_rounds=max_rounds,
                            first_actor=bargaining_config.first_actors[0],
                        )

                        for g in range(rl_group_size):
                            jobs.append((task, group_id))

                        group_id += 1

    # Debug log file
    debug_fp = None
    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_fp = open(debug_dir / f"rollout_seed{seed}.jsonl", "w")

    # Run episodes concurrently
    samples = []
    rewards_so_far = []
    completed = 0
    failed = 0
    total_llm_calls = 0
    t_start = time.time()

    log_interval = max(1, logging_steps)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generate_episode, task, train_role, call_fn): (task, gid)
            for task, gid in jobs
        }
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Rollout episodes"):
            task, gid = futures[fut]
            try:
                episode = fut.result()
            except Exception as e:
                warnings.warn(f"Episode generation failed: {e}")
                failed += 1
                continue

            completed += 1
            n_turns = len(episode["turns"])
            total_llm_calls += n_turns * 2  # both agents called each round
            rewards_so_far.append(episode["reward"])

            for turn in episode["turns"]:
                samples.append({
                    "prompt": turn["prompt"],
                    "completion": turn["completion"],
                    "reward": episode["reward"],
                    "group_id": gid,
                })

            # Write debug log
            if debug_fp is not None:
                debug_fp.write(json.dumps({
                    "episode_id": completed,
                    "group_id": gid,
                    "item": task.item_name,
                    "mode": task.mode,
                    "transparency": task.transparency,
                    "buyer_res_price": task.buyer_res_price,
                    "seller_res_price": task.seller_res_price,
                    "has_zoa": task.buyer_res_price > task.seller_res_price,
                    "max_rounds": task.max_rounds,
                    "result": "deal" if episode["deal_price"] else "no_deal",
                    "deal_price": episode["deal_price"],
                    "rounds": episode["rounds"],
                    "reward": episode["reward"],
                    "n_turns": n_turns,
                    "conversation": episode["log"],
                }) + "\n")
                debug_fp.flush()

            # Periodic wandb progress update
            if wandb_enabled and completed % log_interval == 0:
                import wandb
                elapsed = time.time() - t_start
                rps = total_llm_calls / elapsed if elapsed > 0 else 0
                wandb.log({
                    f"{wandb_prefix}/rollout_progress": completed / len(jobs),
                    f"{wandb_prefix}/rollout_completed": completed,
                    f"{wandb_prefix}/rollout_failed": failed,
                    f"{wandb_prefix}/rollout_llm_calls": total_llm_calls,
                    f"{wandb_prefix}/rollout_rps": round(rps, 2),
                    f"{wandb_prefix}/rollout_running_mean_reward":
                        round(np.mean(rewards_so_far), 4),
                    f"{wandb_prefix}/rollout_deal_rate":
                        round(np.mean([r > 0 for r in rewards_so_far]), 4),
                    f"{wandb_prefix}/rollout_elapsed_s": round(elapsed, 1),
                })

    if debug_fp is not None:
        debug_fp.close()

    elapsed = time.time() - t_start
    rps = total_llm_calls / elapsed if elapsed > 0 else 0

    random.Random(seed + 1).shuffle(samples)

    mean_reward = np.mean([s["reward"] for s in samples]) if samples else 0.0
    deal_rate = np.mean([r > 0 for r in rewards_so_far]) if rewards_so_far else 0.0

    stats = {
        f"{wandb_prefix}/rollout_episodes": completed,
        f"{wandb_prefix}/rollout_failed": failed,
        f"{wandb_prefix}/rollout_llm_calls": total_llm_calls,
        f"{wandb_prefix}/rollout_time_s": round(elapsed, 1),
        f"{wandb_prefix}/rollout_rps": round(rps, 2),
        f"{wandb_prefix}/rollout_samples": len(samples),
        f"{wandb_prefix}/rollout_mean_reward": round(mean_reward, 4),
        f"{wandb_prefix}/rollout_deal_rate": round(deal_rate, 4),
    }

    logger.info(
        f"Rollout done: {completed} episodes ({failed} failed), "
        f"{len(samples)} turn-level samples, {total_llm_calls} LLM calls "
        f"in {elapsed:.1f}s ({rps:.1f} req/s), "
        f"mean reward {mean_reward:.3f}, deal rate {deal_rate:.3f}"
    )
    return samples, stats


def _get_text_tokenizer(tokenizer):
    """Extract the underlying text tokenizer from a processor or tokenizer."""
    # Unsloth may return a processor (e.g. Qwen3VLProcessor) instead of a
    # tokenizer for multimodal-capable models.  The processor wraps a text
    # tokenizer that has .encode() / .decode().
    if hasattr(tokenizer, "encode"):
        return tokenizer
    if hasattr(tokenizer, "tokenizer"):
        return tokenizer.tokenizer
    raise TypeError(f"Cannot extract text tokenizer from {type(tokenizer).__name__}")


def _tokenize_sample(tokenizer, prompt_msgs: list, completion: str,
                     max_prompt_len: int, max_completion_len: int):
    """Tokenize a (prompt, completion) pair for GRPO training.

    Returns (prompt_ids, completion_ids) as 1-D LongTensors, or None if
    either part is empty after tokenization.
    """
    import torch

    text_tok = _get_text_tokenizer(tokenizer)

    # Format prompt messages using chat template
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = text_tok.encode(prompt_text, add_special_tokens=False,
                                 truncation=True, max_length=max_prompt_len)
    comp_ids = text_tok.encode(completion, add_special_tokens=False,
                               truncation=True, max_length=max_completion_len)
    if not prompt_ids or not comp_ids:
        return None
    return (torch.tensor(prompt_ids, dtype=torch.long),
            torch.tensor(comp_ids, dtype=torch.long))


def _compute_logprobs(model, prompt_ids, completion_ids):
    """Compute per-token log probabilities of completion given prompt.

    Returns a 1-D tensor of shape (len(completion_ids),).
    """
    import torch

    input_ids = torch.cat([prompt_ids, completion_ids]).unsqueeze(0)
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        logits = model(input_ids).logits  # (1, seq_len, vocab)

    # Shift: logits[t] predicts token[t+1]
    # We want log p(completion_ids[i]) for each i
    prompt_len = len(prompt_ids)
    # logits at positions [prompt_len-1 .. prompt_len+comp_len-2] predict
    # tokens [prompt_len .. prompt_len+comp_len-1]
    comp_logits = logits[0, prompt_len - 1 : prompt_len + len(completion_ids) - 1]
    log_probs = torch.log_softmax(comp_logits, dim=-1)
    token_log_probs = log_probs.gather(1, completion_ids.unsqueeze(1).to(model.device)).squeeze(1)
    return token_log_probs


def _tokenize_turn(tokenizer, prompt_msgs: list, content: str, reasoning: str,
                   max_len: int, enable_thinking=None):
    """Tokenize a single per-turn training sample for CoT-RL.

    prompt_msgs is the public history (assistant messages carry content only, no
    <think> blocks) up through the user message that prompted this turn.
    content is the assistant's visible reply; reasoning is the text inside the
    <think> block (empty string if the model did not think). The Qwen3 chat
    template renders the last assistant message with reasoning_content in a
    <think> block; previous turns in prompt_msgs are rendered without think
    (template behaviour + public-history construction).

    enable_thinking is passed to apply_chat_template so the generation-prompt
    prefix matches what vLLM uses at rollout time.

    Returns (input_ids, loss_mask) as torch tensors where loss_mask is True only
    on completion tokens (assistant content + <|im_end|>), or None if mask ends
    up empty.
    """
    import torch

    text_tok = _get_text_tokenizer(tokenizer)

    full_msgs = list(prompt_msgs) + [
        {"role": "assistant", "reasoning_content": reasoning, "content": content}
    ]

    ct_kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True, **ct_kwargs
    )
    full_text = tokenizer.apply_chat_template(
        full_msgs, tokenize=False, add_generation_prompt=False, **ct_kwargs
    )

    prompt_ids = text_tok.encode(prompt_text, add_special_tokens=False)
    full_ids = text_tok.encode(full_text, add_special_tokens=False)
    if not isinstance(full_ids, list):
        full_ids = list(full_ids)
    if not isinstance(prompt_ids, list):
        prompt_ids = list(prompt_ids)

    if len(full_ids) <= len(prompt_ids):
        return None

    # Loss-mask construction below assumes prompt_ids is a token-prefix of
    # full_ids. This holds for Qwen3's chat template (and standard
    # decoder-only templates) but is not asserted by HF; if the template
    # ever renders prompt_msgs differently when followed by an assistant
    # turn, the mask would silently misalign and the assistant tokens
    # would not be the ones being trained on. Cheap defensive check.
    if full_ids[: len(prompt_ids)] != prompt_ids:
        warnings.warn(
            "tokenize_turn: prompt_ids is not a prefix of full_ids; "
            "loss mask would misalign. Returning None."
        )
        return None

    mask = [False] * len(prompt_ids) + [True] * (len(full_ids) - len(prompt_ids))

    if len(full_ids) > max_len:
        # Left-truncate (drop oldest prompt tokens), always preserve completion
        excess = len(full_ids) - max_len
        if excess >= len(prompt_ids):
            # Completion alone exceeds budget — drop earliest completion tokens
            full_ids = full_ids[-max_len:]
            mask = [True] * max_len
        else:
            full_ids = full_ids[excess:]
            mask = mask[excess:]

    if not any(mask):
        return None

    return (torch.tensor(full_ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool))


def _compute_masked_logprobs_nograd(model, input_ids, loss_mask):
    """No-grad forward pass; returns log probs at masked positions only.

    loss_mask[t] is True iff input_ids[t] is a trained-policy assistant token.
    The returned tensor has shape (loss_mask[1:].sum(),).
    """
    import torch

    ids = input_ids.unsqueeze(0).to(model.device)
    with torch.no_grad():
        logits = model(ids).logits[0]  # (T, V)
    shift_logits = logits[:-1]                       # (T-1, V)
    shift_labels = input_ids[1:].to(model.device)    # (T-1,)
    shift_mask = loss_mask[1:].to(model.device)      # (T-1,)
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_lp = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)  # (T-1,)
    return token_lp[shift_mask]


def _fuse_lora_for_vllm(checkpoint_path: str) -> None:
    """Rewrite a PEFT-saved LoRA checkpoint in place so vLLM can load every
    weight without silent drops.

    PEFT trains LoRA on the HuggingFace projection names (q_proj, k_proj,
    v_proj, gate_proj, up_proj). vLLM's Qwen3 implementation packs those
    into qkv_proj and gate_up_proj at load time and only matches LoRA names
    against the fused names; LoRAs on the unfused names are silently
    dropped (warning in vLLM startup).

    We fuse three rank-r LoRAs (q,k,v) into one rank-3r LoRA on qkv_proj
    by stacking A vertically and putting B in a block-diagonal layout —
    mathematically equivalent to applying the three originals
    independently. Same construction for gate_proj+up_proj -> gate_up_proj
    at rank 2r. o_proj and down_proj pass through unchanged.

    No-op if the saved adapter already targets the fused names (e.g. an
    earlier checkpoint that didn't use this fuser).
    """
    import json as _json
    from safetensors import safe_open
    from safetensors.torch import save_file
    import torch as _torch

    ckpt_dir = Path(checkpoint_path)
    cfg_path = ckpt_dir / "adapter_config.json"
    weights_path = ckpt_dir / "adapter_model.safetensors"
    if not cfg_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"adapter files missing under {ckpt_dir}")

    cfg = _json.loads(cfg_path.read_text())
    targets = set(cfg.get("target_modules", []))
    triggers = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}
    if not (targets & triggers):
        return  # nothing to fuse

    with safe_open(str(weights_path), framework="pt") as f:
        sd = {k: f.get_tensor(k) for k in f.keys()}

    # Group LoRA tensors by layer prefix. Keys look like:
    # base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    layers = {}
    pass_through = {}
    for k, v in sd.items():
        for proj in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
            tag = f".{proj}.lora_"
            if tag in k:
                prefix, rest = k.split(tag, 1)
                ab = "lora_A" if rest.startswith("A") else "lora_B"
                layers.setdefault(prefix, {})[(proj, ab)] = (v, rest)
                break
        else:
            pass_through[k] = v

    fused_sd = dict(pass_through)
    rank_pattern = dict(cfg.get("rank_pattern", {}))
    r = int(cfg["r"])

    for prefix, group in layers.items():
        for fused_name, parts in [("qkv_proj", ("q_proj", "k_proj", "v_proj")),
                                  ("gate_up_proj", ("gate_proj", "up_proj"))]:
            if not all((p, "lora_A") in group for p in parts):
                continue
            # Tensors all have suffix ".weight" inside the trailing rest
            # (we captured rest after lora_A/B, so reconstruct full key).
            A_list = [group[(p, "lora_A")][0] for p in parts]
            B_list = [group[(p, "lora_B")][0] for p in parts]
            # Sanity: ranks match
            assert all(t.shape[0] == r for t in A_list), \
                f"rank mismatch: expected {r}, got {[t.shape for t in A_list]}"
            n = len(parts)
            A_fused = _torch.cat(A_list, dim=0)              # (n*r, hidden)
            out_dims = [t.shape[0] for t in B_list]
            total_out = sum(out_dims)
            B_fused = _torch.zeros(total_out, n * r, dtype=B_list[0].dtype)
            row = 0
            # Scale B by n so that vLLM's alpha/rank scaling
            # (alpha/(n*r) for the fused module vs alpha/r for the
            # originals) yields the same effective contribution.
            for i, (B_i, out_i) in enumerate(zip(B_list, out_dims)):
                B_fused[row:row + out_i, i * r:(i + 1) * r] = B_i * n
                row += out_i

            # Suffix from one of the originals (e.g. "A.weight"/"B.weight")
            sample_A_rest = group[(parts[0], "lora_A")][1]   # "A.weight"
            sample_B_rest = group[(parts[0], "lora_B")][1]
            fused_sd[f"{prefix}.{fused_name}.lora_{sample_A_rest}"] = A_fused
            fused_sd[f"{prefix}.{fused_name}.lora_{sample_B_rest}"] = B_fused
            rank_pattern[fused_name] = n * r

            # Also delete originals from any pass_through that snuck in
            for p in parts:
                for ab in ("A", "B"):
                    fused_sd.pop(f"{prefix}.{p}.lora_{ab}.weight", None)

    # Write back atomically: tmp file, then rename
    tmp = weights_path.with_suffix(".safetensors.tmp")
    save_file(fused_sd, str(tmp))
    tmp.replace(weights_path)

    # Update adapter_config.json
    new_targets = sorted({"o_proj", "down_proj", "qkv_proj", "gate_up_proj"} & {
        # Keep modules we actually emitted weights for
        *( ["qkv_proj"] if "qkv_proj" in rank_pattern else [] ),
        *( ["gate_up_proj"] if "gate_up_proj" in rank_pattern else [] ),
        *( ["o_proj"] if any(".o_proj.lora_" in k for k in fused_sd) else [] ),
        *( ["down_proj"] if any(".down_proj.lora_" in k for k in fused_sd) else [] ),
    })
    cfg["target_modules"] = new_targets
    cfg["rank_pattern"] = rank_pattern
    cfg_path.write_text(_json.dumps(cfg, indent=2))


def _sync_lora_to_vllm(checkpoint_path: str, lora_name: str,
                       base_url: str = "http://localhost:8000/v1"):
    """Hot-reload LoRA adapter weights into a running vLLM server.

    Uses vLLM's dynamic LoRA loading API with in-place replacement.
    The server must have been started with --enable-lora and
    VLLM_ALLOW_RUNTIME_LORA_UPDATING=True.
    """
    import requests as req
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.hostname}:{parsed.port or 8000}"
    abs_path = str(Path(checkpoint_path).resolve())

    resp = req.post(f"{base}/pause", params={"mode": "keep"})
    resp.raise_for_status()

    resp = req.post(f"{base}/v1/load_lora_adapter", json={
        "lora_name": lora_name,
        "lora_path": abs_path,
        "load_inplace": True,
    })
    resp.raise_for_status()

    resp = req.post(f"{base}/resume")
    resp.raise_for_status()

    logger.info(f"Synced LoRA '{lora_name}' to vLLM from {abs_path}")


def _verify_lora_active_in_vllm(vllm_base_url, lora_name, base_model_name,
                                enable_thinking=None):
    """One-shot sanity check: does vLLM actually apply the LoRA at inference?

    Forces enable_thinking=False to keep outputs short, deterministic, and to
    avoid the failure mode where the thinking chain consumes all the tokens
    and leaves content empty. Compares the full message (content +
    reasoning_content) at temperature=0 between base and LoRA. If they are
    byte-identical, vLLM is silently dropping the LoRA. Logs WARNING and
    returns False in that case.
    """
    import openai
    client = openai.OpenAI(base_url=vllm_base_url, api_key="unused")
    prompt = ("You are a buyer negotiating for a bag of organic apples. "
              "Your reservation price is $12. The seller's reservation price "
              "is uniform on [$5, $9]. Write a single sentence stating your "
              "opening offer and the reason, then on a new line write "
              "OFFER: $X.")
    kwargs = dict(messages=[{"role": "user", "content": prompt}],
                  temperature=0.0, max_completion_tokens=256, seed=42,
                  extra_body={"chat_template_kwargs": {"enable_thinking": False}})

    def _full_msg(model_name):
        msg = client.chat.completions.create(model=model_name, **kwargs).choices[0].message
        content = msg.content or ""
        reasoning = (getattr(msg, "reasoning_content", None)
                     or getattr(msg, "reasoning", None) or "")
        return content, reasoning, content + "\n---\n" + reasoning

    b_content, b_reasoning, b_full = _full_msg(base_model_name)
    l_content, l_reasoning, l_full = _full_msg(lora_name)

    if b_full == l_full:
        logger.warning(
            "LORA VERIFICATION FAILED: vLLM base and LoRA produced byte-identical "
            "output (content+reasoning, %d chars). The LoRA appears to be silently "
            "dropped at inference. If lora_target_modules includes q_proj/k_proj/"
            "v_proj/gate_proj/up_proj, vLLM may not be merging them into qkv_proj/"
            "gate_up_proj as expected.",
            len(b_full),
        )
        logger.warning("Identical content (first 400): %r", b_content[:400])
        return False
    logger.info(
        "LoRA verification: vLLM base vs LoRA differ. "
        "base content_len=%d reasoning_len=%d, lora content_len=%d reasoning_len=%d",
        len(b_content), len(b_reasoning), len(l_content), len(l_reasoning),
    )
    logger.info("base.content[:300]=%r", b_content[:300])
    logger.info("lora.content[:300]=%r", l_content[:300])
    return True


def _build_scenario_batches(scenarios, bargaining_config, scenarios_per_step,
                            rl_group_size, seed, train_roles=None):
    """Pre-build all (task, group_id, train_role) triples and yield them in batches.

    Each batch contains ``scenarios_per_step`` scenarios, each producing
    ``rl_group_size`` episodes.  Reservation prices are sampled once
    deterministically so that batches are reproducible.

    train_roles: list of player roles to sample from. If None or single-element,
    every group uses that role (single-side training, legacy behaviour). If
    multi-element (v7 joint self-play), each group draws its role uniformly
    from train_roles via the deterministic rng.
    """
    rng = random.Random(seed)
    roles = list(train_roles) if train_roles else ["buyer"]

    all_jobs = []   # list of (task, group_id, train_role)
    group_id = 0

    for scenario in scenarios:
        for transparency in bargaining_config.transparencies:
            for max_rounds in bargaining_config.max_rounds:
                for mode in bargaining_config.modes:
                    for _ in range(bargaining_config.n_trials_per_scenario):
                        buyer_range = tuple(scenario["buyer_res_price_range"])
                        seller_range = tuple(scenario["seller_res_price_range"])
                        b_price = round(rng.uniform(*buyer_range), 2)
                        s_price = round(rng.uniform(*seller_range), 2)

                        task = BargainingTask(
                            item_name=scenario["product_name"],
                            item_description=scenario["product_description"],
                            buyer_persona=scenario["buyer_persona"],
                            seller_persona=scenario["seller_persona"],
                            buyer_res_price=b_price,
                            seller_res_price=s_price,
                            buyer_res_price_range=buyer_range,
                            seller_res_price_range=seller_range,
                            transparency=transparency,
                            mode=mode,
                            max_rounds=max_rounds,
                            first_actor=bargaining_config.first_actors[0],
                        )

                        # One role per group (rl_group_size rollouts share role)
                        role = rng.choice(roles)
                        jobs_for_group = [(task, group_id, role)] * rl_group_size
                        all_jobs.extend(jobs_for_group)
                        group_id += 1

    # Shuffle at scenario-group level so batches are diverse
    group_chunks = []
    i = 0
    while i < len(all_jobs):
        group_chunks.append(all_jobs[i : i + rl_group_size])
        i += rl_group_size
    rng.shuffle(group_chunks)

    # Yield batches of scenarios_per_step groups
    batch = []
    for chunk in group_chunks:
        batch.extend(chunk)
        if len(batch) >= scenarios_per_step * rl_group_size:
            yield batch
            batch = []
    if batch:
        yield batch


def run_training(config: Config, splits: dict, run_dir: Path):
    """On-policy GRPO fine-tuning with Unsloth + vLLM LoRA hot-loading.

    Each training step:
      1. Roll out a small batch of episodes using the CURRENT policy
         (trained agent via LoRA in vLLM, opponent via base model).
      2. Compute group-normalised advantages (GRPO).
      3. Compute reference log probs (LoRA disabled = base model).
      4. Apply clipped policy gradient loss + optional KL penalty.
      5. Save LoRA checkpoint and hot-reload into vLLM.
    """
    import os
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from collections import defaultdict
    import contextlib
    import pickle

    # ---- Distributed init ----
    # torchrun sets RANK, WORLD_SIZE, LOCAL_RANK env vars. If running without
    # torchrun (single-process debugging), default to a 1-rank world so the
    # rest of the code still works.
    from datetime import timedelta
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        torch.cuda.set_device(local_rank)
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}")

    tc = config.train
    # Effective player roles. Single-role legacy: [tc.train_role]. Joint
    # self-play (v7): tc.train_roles set to a list of length >= 2 (typically
    # ["buyer","seller"]). In joint mode both roles share tc.lora_name, so
    # the opponent always plays the current policy.
    effective_roles = list(tc.train_roles) if tc.train_roles else [tc.train_role]
    is_joint = len(set(effective_roles)) > 1
    # For single-role we resolve trained/opponent model configs as before.
    # For joint we treat the buyer-config side as "trained-style" (sampling
    # params for both roles must match in joint mode); both call_fns end up
    # pointing at the same LoRA name.
    train_role = effective_roles[0]
    trained_args = config.buyer_model if train_role == "buyer" else config.seller_model
    opponent_args = config.seller_model if train_role == "buyer" else config.buyer_model

    if is_main:
        if is_joint:
            logger.info(f"Training mode: JOINT self-play; roles sampled per scenario from {effective_roles}")
        else:
            logger.info(f"Training role: {train_role}")
        logger.info(f"Trained model: {trained_args.model_name}")
        logger.info(f"Opponent model: {opponent_args.model_name} @ {opponent_args.base_url}")
        logger.info(f"Distributed: world_size={world_size} rank={rank} local_rank={local_rank}")

    # ---- vLLM clients for rollouts (async) ----
    # v8: build four ModelArgs explicitly so sampling params route by role and
    # player-status, not by static train_role[0] binding. Each per-role pair
    # carries the role's sampling settings; in joint mode an OPTIONAL
    # opponent override (tc.opponent_temperature / opponent_top_p) replaces
    # the role's settings only when the role is the opponent.
    def _build_args(yaml_section, *, model_name, opponent_override):
        temp = yaml_section.temperature
        topp = yaml_section.top_p
        if opponent_override:
            if tc.opponent_temperature is not None:
                temp = tc.opponent_temperature
            if tc.opponent_top_p is not None:
                topp = tc.opponent_top_p
        return ModelArgs(
            model_name=model_name,
            base_url=yaml_section.base_url,
            api_key=yaml_section.api_key,
            temperature=temp,
            top_p=topp,
            presence_penalty=yaml_section.presence_penalty,
            max_completion_tokens=yaml_section.max_completion_tokens,
            top_k=yaml_section.top_k,
            enable_thinking=yaml_section.enable_thinking,
        )

    # Player call_fns always sample the trained LoRA. Opponent call_fns use
    # the LoRA in joint mode (self-play) and the base model otherwise.
    buyer_player_args = _build_args(
        config.buyer_model, model_name=tc.lora_name, opponent_override=False,
    )
    seller_player_args = _build_args(
        config.seller_model, model_name=tc.lora_name, opponent_override=False,
    )
    buyer_opponent_args = _build_args(
        config.buyer_model,
        model_name=(tc.lora_name if is_joint else config.buyer_model.model_name),
        opponent_override=is_joint,
    )
    seller_opponent_args = _build_args(
        config.seller_model,
        model_name=(tc.lora_name if is_joint else config.seller_model.model_name),
        opponent_override=is_joint,
    )

    if is_main and is_joint:
        logger.info(
            f"Joint sampling: buyer player T={buyer_player_args.temperature}/p={buyer_player_args.top_p}; "
            f"seller player T={seller_player_args.temperature}/p={seller_player_args.top_p}; "
            f"buyer opponent T={buyer_opponent_args.temperature}/p={buyer_opponent_args.top_p}; "
            f"seller opponent T={seller_opponent_args.temperature}/p={seller_opponent_args.top_p}"
        )

    import httpx
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=2048, max_keepalive_connections=1024),
        timeout=httpx.Timeout(connect=30.0, read=1200.0, write=1200.0, pool=60.0),
    )
    async_client = openai.AsyncOpenAI(
        base_url=trained_args.base_url,
        api_key=trained_args.api_key or "unused",
        http_client=http_client,
    )

    async def buyer_player_call_fn(messages):
        return await _call_vllm_async(
            async_client, buyer_player_args, messages, return_reasoning=True,
        )

    async def seller_player_call_fn(messages):
        return await _call_vllm_async(
            async_client, seller_player_args, messages, return_reasoning=True,
        )

    async def buyer_opponent_call_fn(messages):
        return await _call_vllm_async(async_client, buyer_opponent_args, messages)

    async def seller_opponent_call_fn(messages):
        return await _call_vllm_async(async_client, seller_opponent_args, messages)

    # Legacy aliases for codepaths that still expect a single (trained, opponent)
    # pair (e.g. _verify_lora_active_in_vllm). Points to the player/opponent
    # for the legacy primary role.
    trained_model_args = buyer_player_args if train_role == "buyer" else seller_player_args
    opponent_model_args = seller_opponent_args if train_role == "buyer" else buyer_opponent_args
    async def trained_call_fn(messages):
        return await _call_vllm_async(
            async_client, trained_model_args, messages, return_reasoning=True,
        )
    async def opponent_call_fn(messages):
        return await _call_vllm_async(async_client, opponent_model_args, messages)

    # ---- 1. Load model with HF Transformers + PEFT (DDP-friendly) ----
    load_path = config.train_model_path or trained_args.model_name
    if is_main:
        logger.info(f"Loading model for training from: {load_path}")
    if not tc.lora_target_modules:
        raise ValueError(
            "train.lora_target_modules is not set in the YAML config. "
            "This is model-architecture-specific (e.g. Qwen3: "
            "[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]; "
            "Qwen3.5 Mamba-hybrid: [o_proj, down_proj, out_proj, in_proj_qkv, in_proj_z]). "
            "Set it explicitly — no implicit default."
        )
    # attn_implementation: prefer flash_attention_2 if installed, else fall
    # back to PyTorch's built-in sdpa. On H100/GH200, sdpa uses fused kernels
    # (FlashAttention-style) via torch backends, so the gap to FA2 is small.
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except Exception:
        attn_impl = "sdpa"
    if is_main:
        logger.info(f"attn_implementation = {attn_impl}")
    if tc.load_in_4bit:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            load_path, quantization_config=bnb_cfg, attn_implementation=attn_impl,
        )
        base_model = prepare_model_for_kbit_training(base_model)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            load_path, torch_dtype=torch.bfloat16, attn_implementation=attn_impl,
        )
    base_model = base_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(load_path)

    lora_cfg = LoraConfig(
        r=tc.lora_r,
        lora_alpha=tc.lora_alpha,
        lora_dropout=tc.lora_dropout,
        target_modules=list(tc.lora_target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    # LoRA-on-frozen-base requires this so input embeddings produce grads.
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # DDP wrap
    if world_size > 1:
        ddp_model = DDP(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False, gradient_as_bucket_view=True,
        )
    else:
        ddp_model = model  # 1-rank fallback: act as identity wrapper
    # Helper to get the unwrapped PEFT model regardless of DDP state.
    inner_model = ddp_model.module if world_size > 1 else ddp_model

    # Ensure tokenizer has pad token
    text_tok = _get_text_tokenizer(tokenizer)
    if text_tok.pad_token is None:
        text_tok.pad_token = text_tok.eos_token

    # GRPO hyperparameters
    clip_eps = tc.clip_eps
    kl_coeff = tc.kl_coeff

    optimizer = AdamW(ddp_model.parameters(), lr=tc.learning_rate)
    # Scheduler is created lazily after the first rollout, once we know the
    # actual per-turn sample count N — episode-based estimates are too low
    # by ~1.5x for per-turn CoT-RL and would cause the LR to hit zero by
    # ~step 194 instead of 300. See project_lr_schedule_bug.md.
    scheduler = None
    _effective_batch = tc.per_device_batch_size * tc.gradient_accumulation_steps

    vllm_base_url = trained_args.base_url

    # Save initial LoRA checkpoint and register with vLLM (rank 0 only).
    init_ckpt = run_dir / "checkpoints" / "step_0"
    if is_main:
        init_ckpt.mkdir(parents=True, exist_ok=True)
        inner_model.save_pretrained(str(init_ckpt))
        tokenizer.save_pretrained(str(init_ckpt))
        _sync_lora_to_vllm(str(init_ckpt), tc.lora_name, vllm_base_url)
    if world_size > 1:
        dist.barrier()

    # Build model configs for evaluation (uses vLLM endpoints)
    # Trained agent eval config uses the LoRA name
    trained_eval_args = ModelArgs(
        model_name=tc.lora_name,
        base_url=trained_args.base_url,
        api_key=trained_args.api_key,
        temperature=trained_args.temperature,
        top_p=trained_args.top_p,
        presence_penalty=trained_args.presence_penalty,
        max_completion_tokens=trained_args.max_completion_tokens,
        top_k=trained_args.top_k,
        enable_thinking=trained_args.enable_thinking,
    )
    if is_joint:
        # Both roles use the LoRA at the same sampling params during val too.
        buyer_model = make_model_config(trained_eval_args)
        seller_model = make_model_config(trained_eval_args)
    elif train_role == "buyer":
        buyer_model = make_model_config(trained_eval_args)
        seller_model = make_model_config(opponent_args)
    else:
        buyer_model = make_model_config(opponent_args)
        seller_model = make_model_config(trained_eval_args)

    # Pre-build scenario batches. Pass effective_roles so joint training
    # draws a per-group role at batch-build time (deterministic via the rng
    # seeded by config.data.seed).
    batches = list(_build_scenario_batches(
        splits["train"], config.bargaining,
        tc.scenarios_per_step, tc.rl_group_size, config.data.seed,
        train_roles=effective_roles,
    ))
    total_batches = len(batches)
    episodes_per_step = tc.scenarios_per_step * tc.rl_group_size

    if is_main:
        logger.info(
            f"Training plan: {tc.num_steps} steps, "
            f"{episodes_per_step} episodes/step "
            f"({tc.scenarios_per_step} scenarios × {tc.rl_group_size} group size), "
            f"{total_batches} scenario batches available"
        )
    if config.wandb.enabled and is_main:
        import wandb
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("val/*", step_metric="train/step")
        wandb.define_metric("test/*", step_metric="train/step")
        wandb.config.update({
            "episodes_per_step": episodes_per_step,
            "total_scenario_batches": total_batches,
        }, allow_val_change=True)

    # ---- 2. On-policy training loop ----
    for step in range(1, tc.num_steps + 1):
        step_t0 = time.time()

        # Cycle through batches (reshuffle when exhausted)
        batch_idx = (step - 1) % total_batches
        if batch_idx == 0 and step > 1:
            random.Random(config.data.seed + step).shuffle(batches)
        jobs = batches[batch_idx]

        # 2a/b/c. Rollout, advantages, tokenization — RANK 0 ONLY.
        # The full tokenized list is then broadcast to all ranks for sharding.
        rollout_t0 = time.time()
        step_payload: dict = {"skip": False}
        if is_main:
            episode_samples = []
            rewards_so_far = []
            completed = 0
            failed = 0
            total_llm_calls = 0

            async def _rollout_batch():
                # v8: dispatch per-job role-aware call_fns. Player gets its
                # role's player call_fn; opponent gets the OTHER role's
                # opponent call_fn (which may carry an opponent override).
                def _pair(role):
                    if role == "buyer":
                        return buyer_player_call_fn, seller_opponent_call_fn
                    return seller_player_call_fn, buyer_opponent_call_fn

                coros = []
                for task, gid, job_role in jobs:
                    player_fn, opp_fn = _pair(job_role)
                    coros.append(generate_episode_async(
                        task, job_role,
                        player_fn, opp_fn,
                        signed_reward=tc.signed_reward,
                        offer_sanity_mult=tc.offer_sanity_mult,
                    ))
                return await asyncio.gather(*coros, return_exceptions=True)

            rollout_results = asyncio.run(_rollout_batch())

            deal_flags = []  # True iff this episode produced a deal
            for (task, gid, job_role), result in zip(jobs, rollout_results):
                if isinstance(result, Exception):
                    warnings.warn(f"Episode generation failed: {result}")
                    failed += 1
                    continue

                episode = result
                completed += 1
                n_turns = len(episode["turns"])
                total_llm_calls += n_turns * 2
                rewards_so_far.append(episode["reward"])
                deal_flags.append(episode.get("deal_price") is not None)

                episode_samples.append({
                    "turns": episode["turns"],
                    "reward": episode["reward"],
                    "group_id": gid,
                    "train_role": job_role,
                    "first_trained_offer": episode.get("first_trained_offer"),
                    "buyer_res_price": episode.get("buyer_res_price", task.buyer_res_price),
                    "seller_res_price": episode.get("seller_res_price", task.seller_res_price),
                })

            if not episode_samples:
                logger.warning(f"Step {step}: no episodes generated, skipping")
                step_payload["skip"] = True
            else:
                mean_reward = float(np.mean([s["reward"] for s in episode_samples]))
                deal_rate = float(np.mean(deal_flags)) if deal_flags else 0.0

                # 2b. Optional within-group reward transform (applied BEFORE
                # the group-mean / group-std advantage normalisation).
                # Original s["reward"] is preserved for logging; the transform
                # writes into s["transformed_reward"], which feeds the
                # advantage. When transform is "none" they are identical.
                rt_kind = (tc.reward_transform or "none").lower()
                if rt_kind not in ("none", "rank"):
                    raise ValueError(f"Unknown reward_transform: {tc.reward_transform!r}; "
                                     f"expected 'none' or 'rank'")
                if rt_kind == "rank":
                    by_gid = defaultdict(list)
                    for idx, s in enumerate(episode_samples):
                        by_gid[s["group_id"]].append(idx)
                    for gid, idxs in by_gid.items():
                        raw = [episode_samples[i]["reward"] for i in idxs]
                        ranks = _rank_transform_rewards(raw)
                        for i, r in zip(idxs, ranks):
                            episode_samples[i]["transformed_reward"] = float(r)
                else:
                    for s in episode_samples:
                        s["transformed_reward"] = s["reward"]

                # Group statistics on ORIGINAL rewards (existing diagnostics).
                group_rewards = defaultdict(list)
                for s in episode_samples:
                    group_rewards[s["group_id"]].append(s["reward"])
                group_stds_raw = {gid: np.std(rs) for gid, rs in group_rewards.items()}
                mean_group_std = float(np.mean(list(group_stds_raw.values())))
                frac_zero_std_groups = float(np.mean([s < 1e-6 for s in group_stds_raw.values()]))

                # Group statistics on TRANSFORMED rewards (drives advantages).
                group_rewards_t = defaultdict(list)
                for s in episode_samples:
                    group_rewards_t[s["group_id"]].append(s["transformed_reward"])
                group_stds_t = {gid: np.std(rs) for gid, rs in group_rewards_t.items()}
                group_stats = {
                    gid: (np.mean(rs), group_stds_t[gid] + 1e-8)
                    for gid, rs in group_rewards_t.items()
                }
                mean_transformed_reward = float(np.mean(
                    [s["transformed_reward"] for s in episode_samples]))
                mean_transformed_group_std = float(np.mean(list(group_stds_t.values())))

                norm_first_offers = []
                group_norm_first_offers: dict = defaultdict(list)
                for s in episode_samples:
                    fo = s["first_trained_offer"]
                    spread = s["buyer_res_price"] - s["seller_res_price"]
                    if fo is not None and spread > 1e-6:
                        # In joint mode each sample carries its own player
                        # role; fall back to legacy train_role for older
                        # single-role configs.
                        sample_role = s.get("train_role", train_role)
                        if sample_role == "buyer":
                            nfo = (s["buyer_res_price"] - fo) / spread
                        else:
                            nfo = (fo - s["seller_res_price"]) / spread
                        norm_first_offers.append(nfo)
                        group_norm_first_offers[s["group_id"]].append(nfo)
                mean_norm_fo = float(np.mean(norm_first_offers)) if norm_first_offers else float("nan")
                group_nfo_stds = [np.std(v) for v in group_norm_first_offers.values() if len(v) > 1]
                mean_group_nfo_std = float(np.mean(group_nfo_stds)) if group_nfo_stds else float("nan")
                episode_advantages = []
                for s in episode_samples:
                    mean_r, std_r = group_stats[s["group_id"]]
                    episode_advantages.append((s["transformed_reward"] - mean_r) / std_r)

                # 2c. Tokenize per-turn (per-turn CoT-RL).
                max_seq_len = tc.max_prompt_length + tc.max_completion_length
                tokenized: list = []
                debug_cot = bool(config.debug) and step == 1
                _debug_printed = False
                for i, s in enumerate(episode_samples):
                    adv_i = episode_advantages[i]
                    for ti, turn in enumerate(s["turns"]):
                        tok = _tokenize_turn(
                            tokenizer,
                            turn["prompt_msgs"],
                            turn["content"],
                            turn["reasoning"],
                            max_seq_len,
                            enable_thinking=trained_args.enable_thinking,
                        )
                        if tok is not None:
                            tokenized.append((tok[0], tok[1], adv_i, s["reward"]))

                            if debug_cot and not _debug_printed and i == 0 and ti == 0:
                                _debug_printed = True
                                full_ids, mask = tok
                                ct_kwargs = ({} if trained_args.enable_thinking is None
                                             else {"enable_thinking": trained_args.enable_thinking})
                                prompt_text_sample = tokenizer.apply_chat_template(
                                    turn["prompt_msgs"], tokenize=False, add_generation_prompt=True,
                                    **ct_kwargs,
                                )
                                prompt_has_think = "<think>" in prompt_text_sample
                                model_generated_think = bool(turn["reasoning"])
                                logger.info(
                                    f"[debug-cot] prompt_text_len={len(prompt_text_sample)} "
                                    f"content_len={len(turn['content'])} reasoning_len={len(turn['reasoning'])} "
                                    f"prompt_has_think={prompt_has_think} "
                                    f"model_generated_think={model_generated_think} "
                                    f"mask.sum()={int(mask.sum())} full_len={full_ids.shape[0]}"
                                )
                                logger.info(f"[debug-cot] content[:300]={turn['content'][:300]!r}")
                                logger.info(f"[debug-cot] reasoning[:300]={turn['reasoning'][:300]!r}")
                                logger.info(f"[debug-cot] prompt_text[-400:]={prompt_text_sample[-400:]!r}")
                                assert not prompt_has_think, (
                                    "prompt leaked <think> from a previous turn — "
                                    "per-turn CoT-RL invariant broken"
                                )

                if not tokenized:
                    logger.warning(f"Step {step}: no valid tokenized samples, skipping")
                    step_payload["skip"] = True
                else:
                    step_payload.update({
                        "tokenized": tokenized,
                        "mean_reward": mean_reward,
                        "deal_rate": deal_rate,
                        "mean_group_std": mean_group_std,
                        "frac_zero_std_groups": frac_zero_std_groups,
                        "mean_norm_fo": mean_norm_fo,
                        "mean_group_nfo_std": mean_group_nfo_std,
                        "mean_transformed_reward": mean_transformed_reward,
                        "mean_transformed_group_std": mean_transformed_group_std,
                        "reward_transform": rt_kind,
                        "completed": completed,
                        "failed": failed,
                        "total_llm_calls": total_llm_calls,
                    })

        rollout_elapsed = time.time() - rollout_t0

        # Broadcast step_payload from rank 0 to all ranks
        if world_size > 1:
            obj_list = [step_payload]
            dist.broadcast_object_list(obj_list, src=0)
            step_payload = obj_list[0]

        if step_payload.get("skip", False):
            if world_size > 1:
                dist.barrier()
            continue

        tokenized = step_payload["tokenized"]
        mean_reward = step_payload["mean_reward"]
        deal_rate = step_payload["deal_rate"]
        mean_group_std = step_payload["mean_group_std"]
        frac_zero_std_groups = step_payload["frac_zero_std_groups"]
        mean_norm_fo = step_payload["mean_norm_fo"]
        mean_group_nfo_std = step_payload["mean_group_nfo_std"]
        mean_transformed_reward = step_payload["mean_transformed_reward"]
        mean_transformed_group_std = step_payload["mean_transformed_group_std"]
        reward_transform_kind = step_payload["reward_transform"]
        completed = step_payload["completed"]
        failed = step_payload["failed"]
        total_llm_calls = step_payload["total_llm_calls"]

        # ---- Length-bucket LPT shard: each rank picks its slice deterministically ----
        if world_size > 1:
            order = sorted(range(len(tokenized)),
                           key=lambda i: int(tokenized[i][0].shape[0]),
                           reverse=True)
            bins = [[] for _ in range(world_size)]
            bin_loads = [0] * world_size
            for idx in order:
                tlen = int(tokenized[idx][0].shape[0])
                target = bin_loads.index(min(bin_loads))
                bins[target].append(idx)
                bin_loads[target] += tlen
            local_indices = bins[rank]
        else:
            local_indices = list(range(len(tokenized)))
        local_tok = [tokenized[i] for i in local_indices]
        # Equalize sample counts across ranks. Without this, ranks do
        # different numbers of forwards before each backward → DDP gradient
        # hooks desync → NCCL all-reduce SeqNum mismatch → NCCL timeout.
        # Truncate every rank to the global MIN. Drops a small number of
        # samples on long-shard ranks; safer than reordering collectives.
        # v8: log how many samples the DDP MIN-truncation drops. Long-shard
        # ranks lose samples to keep mini-batch counts equal. Bias can creep
        # in if buyer-player and seller-player rollouts have systematically
        # different turn counts.
        ddp_truncation_dropped = 0
        ddp_truncation_pre = 0
        if world_size > 1:
            n_pre = len(local_tok)
            tn = torch.tensor([n_pre], device=device, dtype=torch.long)
            tn_max = torch.tensor([n_pre], device=device, dtype=torch.long)
            tn_sum = torch.tensor([n_pre], device=device, dtype=torch.long)
            dist.all_reduce(tn, op=dist.ReduceOp.MIN)
            dist.all_reduce(tn_max, op=dist.ReduceOp.MAX)
            dist.all_reduce(tn_sum, op=dist.ReduceOp.SUM)
            global_min = int(tn.item())
            global_max = int(tn_max.item())
            global_pre = int(tn_sum.item())
            local_tok = local_tok[:global_min]
            n_dropped_local = n_pre - global_min
            td = torch.tensor([n_dropped_local], device=device, dtype=torch.long)
            dist.all_reduce(td, op=dist.ReduceOp.SUM)
            ddp_truncation_dropped = int(td.item())
            ddp_truncation_pre = global_pre
            if is_main:
                logger.info(
                    f"Step {step}: DDP MIN-truncation dropped "
                    f"{ddp_truncation_dropped}/{ddp_truncation_pre} samples "
                    f"({100*ddp_truncation_dropped/max(ddp_truncation_pre,1):.1f}%); "
                    f"rank counts: min={global_min} max={global_max}"
                )
        if not local_tok:
            if is_main:
                logger.warning(f"Step {step}: a rank received 0 samples, skipping")
            if world_size > 1:
                dist.barrier()
            continue

        # 2d. Reference log probs at masked positions (disable LoRA = base model).
        # Each rank computes ref logprobs over its local shard only; the
        # training loop below also consumes only the local shard, so no
        # all-gather needed. Model stays in train() mode throughout for
        # consistency between ref_lp and cur_lp under any future
        # dropout > 0 setting; the actual forward is wrapped in no_grad
        # inside _compute_masked_logprobs_nograd.
        inner_model.disable_adapter_layers()
        ref_logprobs = []
        for input_ids, loss_mask, _, _ in local_tok:
            lp = _compute_masked_logprobs_nograd(inner_model, input_ids, loss_mask)
            ref_logprobs.append(lp.detach().cpu())
        inner_model.enable_adapter_layers()

        # 2e. Training step (per-optimizer-step LR decay; multi-epoch)
        inner_model.train()

        batch_size = tc.per_device_batch_size
        accum_steps = tc.gradient_accumulation_steps
        n_samples = len(local_tok)
        local_n_mini_batches = (n_samples + batch_size - 1) // batch_size

        # Synchronize n_mini_batches across ranks: take the MIN so every rank
        # takes the same number of optimizer steps. Drop leftover mini-batches
        # on long-shard ranks. Without this, scheduler.step() drifts and DDP
        # may hang on the final unbalanced backward.
        if world_size > 1:
            t = torch.tensor([local_n_mini_batches], device=device, dtype=torch.long)
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            n_mini_batches = int(t.item())
        else:
            n_mini_batches = local_n_mini_batches
        expected_optim_steps = (
            ((n_mini_batches + accum_steps - 1) // accum_steps) * tc.epochs_per_rollout
        )

        if is_main:
            logger.info(
                f"Step {step}: starting optimization phase "
                f"(N_local={n_samples} mini_batches={n_mini_batches} "
                f"batch={batch_size} accum={accum_steps} "
                f"epochs={tc.epochs_per_rollout}, ~{expected_optim_steps} optimizer steps)"
            )

        # Lazy scheduler init: built from observed first-step optimizer-step
        # budget. Identical on all ranks because n_mini_batches was MIN-synced.
        if scheduler is None:
            _t_max = tc.num_steps * expected_optim_steps
            def _make_clamped_cosine(T_max):
                def _lambda(t):
                    return 0.5 * (1.0 + math.cos(math.pi * min(t, T_max) / T_max))
                return _lambda
            scheduler = LambdaLR(optimizer, lr_lambda=_make_clamped_cosine(_t_max))
            if is_main:
                logger.info(
                    f"LR scheduler: clamped-cosine LambdaLR T_max={_t_max} "
                    f"({tc.num_steps} training steps × {expected_optim_steps} optimizer steps/step at step 1)"
                )

        optim_t0 = time.time()

        optimizer.zero_grad()
        total_loss = 0.0
        total_pg = 0.0
        total_kl = 0.0
        n_accum = 0
        optim_step_count = 0
        interval_pg_sum = 0.0
        interval_kl_sum = 0.0
        interval_mb_count = 0
        interval_gnorm_sum = 0.0
        interval_optim_count = 0
        # Ratio statistics for diagnostics (mean ratio across all tokens in
        # the optimization phase; clip_frac semantics depend on loss_type).
        total_ratio_sum = 0.0
        total_ratio_count = 0
        total_clip_count = 0

        # Per-sample importance-sampling cache. When tc.is_correction is
        # True, we capture cur_lp on the first forward pass per sample and
        # use it as old_lp for the PPO ratio in subsequent epochs. With
        # is_correction=False we keep the legacy behaviour (ratio against
        # ref_lp = base policy) so pre-v5 YAMLs reproduce v3/v4 results.
        old_lp_cache: Dict[int, torch.Tensor] = {}

        for epoch in range(tc.epochs_per_rollout):
            indices = list(range(n_samples))
            random.Random(config.data.seed + step * 1000 + epoch + rank * 31).shuffle(indices)

            for step_i in range(n_mini_batches):
                batch_start = step_i * batch_size
                batch_idx_list = indices[batch_start : batch_start + batch_size]

                batch_loss = torch.tensor(0.0, device=device)
                batch_pg = 0.0
                batch_kl = 0.0

                for idx in batch_idx_list:
                    input_ids, loss_mask, adv, _ = local_tok[idx]
                    ref_lp = ref_logprobs[idx].to(device)

                    ids_t = input_ids.unsqueeze(0).to(device)
                    logits = ddp_model(ids_t).logits[0]  # (T, V)
                    shift_logits = logits[:-1]
                    shift_labels = input_ids[1:].to(device)
                    shift_mask = loss_mask[1:].to(device)
                    log_probs = torch.log_softmax(shift_logits, dim=-1)
                    token_lp = log_probs.gather(
                        1, shift_labels.unsqueeze(1)
                    ).squeeze(1)
                    cur_lp = token_lp[shift_mask]

                    if tc.is_correction:
                        if idx not in old_lp_cache:
                            old_lp_cache[idx] = cur_lp.detach().clone()
                        old_lp = old_lp_cache[idx]
                    else:
                        old_lp = ref_lp

                    ratio = torch.exp(cur_lp - old_lp)
                    adv_t = torch.tensor(adv, device=device, dtype=cur_lp.dtype)

                    if tc.loss_type == "grpo":
                        pg1 = ratio * adv_t
                        pg2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t
                        pg_loss = -torch.min(pg1, pg2).mean()
                        with torch.no_grad():
                            clipped = ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps))
                    elif tc.loss_type == "cispo":
                        w = torch.clamp(ratio, max=tc.epsilon_high).detach()
                        pg_loss = -(w * adv_t * cur_lp).mean()
                        with torch.no_grad():
                            clipped = ratio > tc.epsilon_high
                    else:
                        raise ValueError(
                            f"Unknown loss_type: {tc.loss_type!r}; expected 'grpo' or 'cispo'"
                        )

                    with torch.no_grad():
                        n_tok = ratio.numel()
                        total_ratio_sum += ratio.sum().item()
                        total_ratio_count += n_tok
                        total_clip_count += int(clipped.sum().item())

                    if tc.kl_estimator == "weighted_legacy":
                        kl = (torch.exp(ref_lp) * (ref_lp - cur_lp)).mean()
                    elif tc.kl_estimator == "k3":
                        log_r = ref_lp - cur_lp
                        kl = (torch.exp(log_r) - log_r - 1.0).mean()
                    else:
                        raise ValueError(
                            f"Unknown kl_estimator: {tc.kl_estimator!r}; "
                            f"expected 'weighted_legacy' or 'k3'"
                        )

                    sample_loss = pg_loss + kl_coeff * kl
                    batch_loss = batch_loss + sample_loss / (len(batch_idx_list) * accum_steps)
                    batch_pg += pg_loss.item()
                    batch_kl += kl.item()

                # Skip gradient sync on accumulation passes (only the last
                # backward in the bucket needs synced gradients).
                is_last_in_accum = (
                    (step_i + 1) % accum_steps == 0 or step_i == n_mini_batches - 1
                )
                if world_size > 1 and not is_last_in_accum:
                    sync_ctx = ddp_model.no_sync()
                else:
                    sync_ctx = contextlib.nullcontext()
                with sync_ctx:
                    batch_loss.backward()
                total_loss += batch_loss.item()
                mb_pg = batch_pg / len(batch_idx_list)
                mb_kl = batch_kl / len(batch_idx_list)
                total_pg += mb_pg
                total_kl += mb_kl
                n_accum += 1
                interval_pg_sum += mb_pg
                interval_kl_sum += mb_kl
                interval_mb_count += 1

                if is_last_in_accum:
                    gnorm = torch.nn.utils.clip_grad_norm_(
                        ddp_model.parameters(), tc.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    optim_step_count += 1
                    try:
                        interval_gnorm_sum += float(gnorm)
                    except Exception:
                        pass
                    interval_optim_count += 1

                    if is_main and optim_step_count % tc.optim_log_interval == 0:
                        avg_pg_i = interval_pg_sum / max(interval_mb_count, 1)
                        avg_kl_i = interval_kl_sum / max(interval_mb_count, 1)
                        avg_gn_i = interval_gnorm_sum / max(interval_optim_count, 1)
                        logger.info(
                            f"optim {optim_step_count}/{expected_optim_steps}: "
                            f"pg={avg_pg_i:.4f} kl={avg_kl_i:.4f} "
                            f"grad_norm={avg_gn_i:.3f} "
                            f"lr={scheduler.get_last_lr()[0]:.2e}"
                        )
                        interval_pg_sum = 0.0
                        interval_kl_sum = 0.0
                        interval_mb_count = 0
                        interval_gnorm_sum = 0.0
                        interval_optim_count = 0

        optim_elapsed = time.time() - optim_t0
        if is_main:
            logger.info(
                f"Step {step}: optimization phase complete "
                f"({optim_elapsed:.1f}s, {optim_step_count} updates)"
            )
        train_elapsed = time.time() - step_t0

        avg_loss = total_loss / max(n_accum, 1)
        avg_pg = total_pg / max(n_accum, 1)
        avg_kl = total_kl / max(n_accum, 1)
        mean_ratio = total_ratio_sum / max(total_ratio_count, 1)
        clip_frac = total_clip_count / max(total_ratio_count, 1)

        # 2f. Save LoRA checkpoint and hot-reload into vLLM (rank 0 only).
        ckpt_dir = run_dir / "checkpoints" / f"step_{step}"
        if is_main:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            inner_model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            _sync_lora_to_vllm(str(ckpt_dir), tc.lora_name, vllm_base_url)

            # One-shot sanity check after the first trained checkpoint: confirm
            # vLLM actually applies the LoRA (i.e. q/k/v get merged into qkv_proj
            # rather than silently dropped). Only run once.
            if step == 1:
                _verify_lora_active_in_vllm(
                    vllm_base_url, tc.lora_name, trained_args.model_name,
                    enable_thinking=trained_args.enable_thinking,
                )

            # Log to wandb
            if config.wandb.enabled:
                wandb_payload = {
                    "train/step": step,
                    "train/loss": avg_loss,
                    "train/pg_loss": avg_pg,
                    "train/kl": avg_kl,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/mean_reward": round(mean_reward, 4),
                    "train/deal_rate": round(deal_rate, 4),
                    "train/episodes": completed,
                    "train/episodes_failed": failed,
                    "train/samples": len(tokenized),
                    "train/rollout_time_s": round(rollout_elapsed, 1),
                    "train/step_time_s": round(train_elapsed, 1),
                    "train/rollout_rps": round(
                        total_llm_calls / rollout_elapsed if rollout_elapsed > 0 else 0, 2),
                    "train/mean_group_std": round(mean_group_std, 4),
                    "train/frac_zero_std_groups": round(frac_zero_std_groups, 4),
                    "train/mean_norm_first_offer": round(mean_norm_fo, 4) if not math.isnan(mean_norm_fo) else None,
                    "train/mean_group_norm_fo_std": round(mean_group_nfo_std, 4) if not math.isnan(mean_group_nfo_std) else None,
                    "train/loss_type": tc.loss_type,
                    "train/is_correction": int(bool(tc.is_correction)),
                    "train/mean_ratio": round(mean_ratio, 4),
                    "train/clip_frac": round(clip_frac, 4),
                    "train/ddp_samples_dropped": ddp_truncation_dropped,
                    "train/ddp_samples_pre_trunc": ddp_truncation_pre,
                    "train/ddp_drop_fraction": (
                        round(ddp_truncation_dropped / ddp_truncation_pre, 4)
                        if ddp_truncation_pre > 0 else 0.0
                    ),
                }
                if reward_transform_kind != "none":
                    wandb_payload["train/mean_transformed_reward"] = round(mean_transformed_reward, 4)
                    wandb_payload["train/mean_transformed_group_std"] = round(mean_transformed_group_std, 4)
                wandb.log(wandb_payload)

            # Log to console
            if step % tc.logging_steps == 0 or step == 1:
                tr_extra = ""
                if reward_transform_kind != "none":
                    tr_extra = (f"tr_reward={mean_transformed_reward:.3f} "
                                f"tr_group_std={mean_transformed_group_std:.3f} ")
                logger.info(
                    f"Step {step}/{tc.num_steps}: loss={avg_loss:.4f} "
                    f"pg={avg_pg:.4f} kl={avg_kl:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} "
                    f"reward={mean_reward:.3f} deal_rate={deal_rate:.3f} "
                    f"group_std={mean_group_std:.3f} zero_std={frac_zero_std_groups:.2f} "
                    f"norm_fo={mean_norm_fo:.3f} group_fo_std={mean_group_nfo_std:.3f} "
                    f"{tr_extra}"
                    f"loss={tc.loss_type} is_corr={int(bool(tc.is_correction))} "
                    f"mean_ratio={mean_ratio:.3f} clip_frac={clip_frac:.3f} "
                    f"({train_elapsed:.0f}s)"
                )

            # 2g. Periodic validation (rank 0 only — uses vLLM, GPU 0 idle here)
            if step % tc.eval_interval == 0 and splits.get("val"):
                logger.info(f"Running validation (step {step})...")
                val_df = asyncio.run(
                    run_evaluation("val", splits["val"], config, buyer_model,
                                   seller_model, step=step)
                )
                val_df.to_csv(run_dir / f"val_step_{step}.csv", index=False)

            if step % tc.save_steps == 0:
                logger.info(f"Saved checkpoint at step {step}")

        # Barrier so all ranks wait for rank 0's save+sync+val before next step.
        if world_size > 1:
            dist.barrier()

    # ---- 3. Test evaluation (rank 0 only) ----
    if is_main and splits.get("test"):
        logger.info("Running test evaluation...")
        test_df = asyncio.run(
            run_evaluation("test", splits["test"], config, buyer_model,
                           seller_model, step=tc.num_steps)
        )
        test_df.to_csv(run_dir / "test_results.csv", index=False)

    # ---- 4. Save final model (rank 0 only) ----
    if is_main:
        final_dir = run_dir / "final_model"
        inner_model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        logger.info(f"Saved final model to {final_dir}")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    return inner_model, tokenizer


# ---------------------------------------------------------------------------
# Batch-size profiling (Stage B of the DDP rewrite plan)
# ---------------------------------------------------------------------------

def run_profile_batch(config: Config, splits: dict, run_dir: Path,
                      sweep_sizes: list, mem_cap_gb: float):
    """Profile peak GPU memory and throughput at various per_device_batch
    sizes, using synthetic samples whose lengths mimic the real distribution
    seen in production. Recommends the largest batch that stays under the
    memory cap. No vLLM server needed.

    Synthetic samples: random token IDs in [0, vocab_size), lengths drawn
    uniformly from [800, 2500] tokens, loss_mask covering the last 20-60% of
    each sequence (mimicking per-turn CoT-RL: prompt + completion).
    """
    import os
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.optim import AdamW
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    import contextlib

    from datetime import timedelta
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        torch.cuda.set_device(local_rank)
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}")

    tc = config.train
    trained_args = config.buyer_model if tc.train_role == "buyer" else config.seller_model
    load_path = config.train_model_path or trained_args.model_name

    if is_main:
        logger.info(f"[profile_batch] Loading model from {load_path}")
    if not tc.lora_target_modules:
        raise ValueError("train.lora_target_modules must be set")
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except Exception:
        attn_impl = "sdpa"
    if is_main:
        logger.info(f"[profile_batch] attn_implementation = {attn_impl}")
    base_model = AutoModelForCausalLM.from_pretrained(
        load_path, torch_dtype=torch.bfloat16, attn_implementation=attn_impl,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    vocab_size = tokenizer.vocab_size

    lora_cfg = LoraConfig(
        r=tc.lora_r, lora_alpha=tc.lora_alpha, lora_dropout=tc.lora_dropout,
        target_modules=list(tc.lora_target_modules), task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    if world_size > 1:
        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=False, gradient_as_bucket_view=True)
    else:
        ddp_model = model
    inner_model = ddp_model.module if world_size > 1 else ddp_model

    # ---- Generate synthetic samples (deterministic seed) ----
    n_samples_total = 512
    rng = random.Random(config.data.seed)
    max_seq_len = tc.max_prompt_length + tc.max_completion_length
    samples = []
    for _ in range(n_samples_total):
        # Realistic token-length distribution: 800–2500
        length = rng.randint(800, min(2500, max_seq_len))
        # Mask covers the last 20–60% of the sequence (completion span)
        completion_frac = rng.uniform(0.2, 0.6)
        completion_start = int(length * (1 - completion_frac))
        ids = torch.randint(0, vocab_size, (length,), dtype=torch.long)
        mask = torch.zeros(length, dtype=torch.bool)
        mask[completion_start:] = True
        adv = rng.uniform(-1.0, 1.0)
        samples.append((ids, mask, adv, 0.0))

    # LPT shard locally so each rank gets a balanced slice
    if world_size > 1:
        order = sorted(range(len(samples)),
                       key=lambda i: int(samples[i][0].shape[0]), reverse=True)
        bins = [[] for _ in range(world_size)]
        loads = [0] * world_size
        for idx in order:
            t = bins.index(min(bins, key=lambda b: loads[bins.index(b)]))
            # cleaner: pick smallest-load bin
            t = loads.index(min(loads))
            bins[t].append(idx)
            loads[t] += int(samples[idx][0].shape[0])
        local_indices = bins[rank]
    else:
        local_indices = list(range(len(samples)))
    local_samples = [samples[i] for i in local_indices]

    if is_main:
        logger.info(
            f"[profile_batch] Generated {n_samples_total} synthetic samples; "
            f"local shard size {len(local_samples)} (world_size={world_size})"
        )

    # ---- Sweep ----
    summary = []  # list of (batch_size, peak_mem_gb_max, tokens_per_s_min, status)
    clip_eps = tc.clip_eps
    kl_coeff = tc.kl_coeff

    for bsz in sweep_sizes:
        if is_main:
            logger.info(f"[profile_batch] Trying per_device_batch_size={bsz}")
        # Fresh optimizer for clean Adam state
        optimizer = AdamW(ddp_model.parameters(), lr=tc.learning_rate)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        n_local = len(local_samples)
        n_mb_local = (n_local + bsz - 1) // bsz
        if world_size > 1:
            t = torch.tensor([n_mb_local], device=device, dtype=torch.long)
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            n_mb = int(t.item())
        else:
            n_mb = n_mb_local

        status = "ok"
        total_tokens = 0
        t0 = time.time()
        try:
            inner_model.train()
            optimizer.zero_grad()
            for step_i in range(n_mb):
                batch_loss = torch.tensor(0.0, device=device)
                idx0 = step_i * bsz
                batch = local_samples[idx0: idx0 + bsz]
                for ids, mask, adv, _ in batch:
                    total_tokens += int(mask.sum().item())
                    ids_t = ids.unsqueeze(0).to(device)
                    logits = ddp_model(ids_t).logits[0]
                    shift_logits = logits[:-1]
                    shift_labels = ids[1:].to(device)
                    shift_mask = mask[1:].to(device)
                    log_probs = torch.log_softmax(shift_logits, dim=-1)
                    token_lp = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
                    cur_lp = token_lp[shift_mask]
                    # Use a frozen "ref" = current logprobs detached, so ratio=1
                    ref_lp = cur_lp.detach()
                    ratio = torch.exp(cur_lp - ref_lp)
                    adv_t = torch.tensor(adv, device=device, dtype=cur_lp.dtype)
                    pg1 = ratio * adv_t
                    pg2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t
                    pg_loss = -torch.min(pg1, pg2).mean()
                    kl = (torch.exp(ref_lp) * (ref_lp - cur_lp)).mean()
                    sample_loss = pg_loss + kl_coeff * kl
                    batch_loss = batch_loss + sample_loss / max(len(batch), 1)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), tc.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
        except torch.cuda.OutOfMemoryError as e:
            status = "OOM"
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                status = "OOM"
                torch.cuda.empty_cache()
            else:
                raise
        elapsed = time.time() - t0

        peak_mem_b = torch.cuda.max_memory_allocated(device)
        peak_mem_gb = peak_mem_b / (1024 ** 3)
        tps = total_tokens / elapsed if elapsed > 0 else 0.0

        # Aggregate worst-case across ranks: MAX peak_mem, MIN tps
        if world_size > 1:
            peak_t = torch.tensor([peak_mem_gb], device=device, dtype=torch.float32)
            dist.all_reduce(peak_t, op=dist.ReduceOp.MAX)
            peak_mem_gb = float(peak_t.item())
            tps_t = torch.tensor([tps], device=device, dtype=torch.float32)
            dist.all_reduce(tps_t, op=dist.ReduceOp.MIN)
            tps = float(tps_t.item())
            status_t = torch.tensor([0 if status == "ok" else 1], device=device, dtype=torch.long)
            dist.all_reduce(status_t, op=dist.ReduceOp.MAX)
            if int(status_t.item()) > 0:
                status = "OOM"

        if is_main:
            logger.info(
                f"[profile_batch] bsz={bsz:>3}  peak_mem={peak_mem_gb:5.1f} GB  "
                f"tokens/s={tps:8.0f}  status={status}"
            )
        summary.append((bsz, peak_mem_gb, tps, status))

        # Stop early if OOM or memory cap exceeded
        if status == "OOM" or peak_mem_gb > mem_cap_gb:
            if is_main:
                logger.info(
                    f"[profile_batch] stopping sweep at bsz={bsz} "
                    f"(status={status}, peak_mem={peak_mem_gb:.1f} GB > cap {mem_cap_gb} GB)"
                )
            break

    # ---- Recommendation ----
    if is_main:
        valid = [(b, m, tp) for (b, m, tp, st) in summary if st == "ok" and m <= mem_cap_gb]
        if not valid:
            logger.warning("[profile_batch] no valid batch size below memory cap")
            recommendation = None
        else:
            # Largest batch where tokens/s >= 95% of next-smaller batch's tokens/s.
            # If the largest valid batch's throughput dipped, walk back.
            valid.sort(key=lambda t: t[0])  # ascending by bsz
            recommendation = valid[-1][0]
            for i in range(len(valid) - 1, 0, -1):
                b_i, _, tp_i = valid[i]
                _, _, tp_prev = valid[i - 1]
                if tp_prev > 0 and tp_i < 0.95 * tp_prev:
                    recommendation = valid[i - 1][0]
                else:
                    break

        # Write summary CSV
        import csv
        with open(run_dir / "profile_batch_summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["batch_size", "peak_mem_gb", "tokens_per_s", "status"])
            for row in summary:
                w.writerow(row)

        logger.info("[profile_batch] === SUMMARY ===")
        logger.info(f"{'bsz':>4} {'peak_mem_gb':>12} {'tokens/s':>10} {'status':>8}")
        for bsz, mem, tps, st in summary:
            logger.info(f"{bsz:>4} {mem:>12.1f} {tps:>10.0f} {st:>8}")
        logger.info(f"[profile_batch] Recommendation: per_device_batch_size = {recommendation}")
        logger.info(f"[profile_batch] Wrote {run_dir / 'profile_batch_summary.csv'}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_model_config(model_args: ModelArgs) -> ModelConfig:
    """Create a ModelConfig from model args (each model has its own endpoint)."""
    client = openai.AsyncOpenAI(
        base_url=model_args.base_url,
        api_key=model_args.api_key,
    )
    extra_body: Dict[str, Any] = {"top_k": model_args.top_k}
    if model_args.enable_thinking is not None:
        extra_body["chat_template_kwargs"] = {
            "enable_thinking": bool(model_args.enable_thinking)
        }
    args = {
        "model": model_args.model_name,
        "temperature": model_args.temperature,
        "top_p": model_args.top_p,
        "presence_penalty": model_args.presence_penalty,
        "max_completion_tokens": model_args.max_completion_tokens,
        "extra_body": extra_body,
    }
    return ModelConfig(
        client=client,
        api_method=client.chat.completions.create,
        use_system_message=False,
        args=args,
    )


def _stamp_code_version(run_dir: Path):
    """Write code_version.txt and a code_snapshot/ copy of bargaining_rl.py
    plus the resolved config.yaml into the run dir. Idempotent."""
    import shutil
    cv_path = run_dir / "code_version.txt"
    if not cv_path.exists():
        cv_path.write_text(CODE_VERSION + "\n")
    snap_dir = run_dir / "code_snapshot"
    snap_dir.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve()
    dst = snap_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)


def main():
    import os
    parser = argparse.ArgumentParser(description="Bargaining RL — evaluation and training")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument(
        "--mode", choices=["eval", "train", "profile_batch"], default=None,
        help="Override mode from config (profile_batch sweeps per_device_batch_size)",
    )
    parser.add_argument(
        "--profile_batch_sizes", type=str, default="16,24,32,40,48,56",
        help="Comma-separated batch sizes for profile_batch mode",
    )
    parser.add_argument(
        "--profile_batch_mem_cap_gb", type=float, default=85.0,
        help="Stop sweep when peak GPU memory exceeds this many GB",
    )
    args, unknown = parser.parse_known_args()

    config = load_config(args.config, unknown)
    if args.mode:
        config.mode = args.mode

    # Global seeding for reproducibility. Each torchrun rank also offsets
    # by its rank below so different ranks shuffle differently while
    # remaining deterministic across reruns.
    _seed = int(getattr(config.data, "seed", 42) or 42)
    random.seed(_seed)
    np.random.seed(_seed)
    try:
        import torch
        torch.manual_seed(_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_seed)
    except Exception:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    # Suppress noisy per-request httpx logs
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # ---- Distributed-aware setup ----
    # Under torchrun every rank runs main(). Rank 0 creates the run dir and
    # stamps it; the timestamp must be shared so all ranks agree on the path.
    # CRITICAL: pin each rank to its own CUDA device BEFORE init_process_group
    # so NCCL collectives don't all land on GPU 0.
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        import torch
        import torch.distributed as dist
        from datetime import timedelta
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            # Long timeout: rollouts (rank 0 only) and validation can take
            # 5-30 minutes; other ranks wait at the next collective. Default
            # 10-minute NCCL timeout would fire spuriously.
            dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    is_main = (rank == 0)

    # Suppress per-rank logging spam: non-zero ranks log only WARNING+.
    if not is_main:
        logging.getLogger("__main__").setLevel(logging.WARNING)

    if is_main:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = Path(config.output_dir) / f"rl_{config.mode}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        # Persist config, code version, and a code snapshot for provenance.
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(asdict(config), f, default_flow_style=False)
        _stamp_code_version(run_dir)
        run_dir_str = str(run_dir)
    else:
        run_dir_str = None
        timestamp = None

    if distributed:
        obj = [run_dir_str, timestamp]
        dist.broadcast_object_list(obj, src=0)
        run_dir_str, timestamp = obj
    run_dir = Path(run_dir_str)

    if is_main:
        logger.info(f"Output directory: {run_dir}")
        logger.info(f"Code version: {CODE_VERSION}")
        logger.info(f"Mode: {config.mode}")
        logger.info(f"Buyer model: {config.buyer_model.model_name} @ {config.buyer_model.base_url}")
        logger.info(f"Seller model: {config.seller_model.model_name} @ {config.seller_model.base_url}")
        if distributed:
            logger.info(f"Distributed: world_size={world_size}")

    # Setup wandb (rank 0 only)
    if config.wandb.enabled and is_main:
        import wandb
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=config.wandb.run_name or f"{config.mode}_{timestamp}",
            config=asdict(config),
            tags=config.wandb.tags,
            dir=str(run_dir),
        )
        print(f"wandb run: {wandb.run.get_url()}")

    # Load and split data
    splits = load_scenarios(config.data)

    if config.mode == "eval":
        if not is_main:
            # Eval is sync and rank-0-only; non-zero ranks just exit.
            return
        buyer_model = make_model_config(config.buyer_model)
        seller_model = make_model_config(config.seller_model)
        for split_name in ["val", "test"]:
            if not splits[split_name]:
                logger.info(f"Skipping {split_name}: no scenarios")
                continue
            df = asyncio.run(
                run_evaluation(split_name, splits[split_name], config, buyer_model, seller_model)
            )
            df.to_pickle(run_dir / f"{split_name}_results.pkl")
            df.to_csv(run_dir / f"{split_name}_results.csv", index=False)
            logger.info(f"Saved {split_name} results to {run_dir}")

    elif config.mode == "train":
        run_training(config, splits, run_dir)

    elif config.mode == "profile_batch":
        sizes = [int(s.strip()) for s in args.profile_batch_sizes.split(",") if s.strip()]
        run_profile_batch(config, splits, run_dir, sizes, args.profile_batch_mem_cap_gb)

    if config.wandb.enabled and is_main:
        import wandb
        wandb.finish()

    logger.info("Done.")


if __name__ == "__main__":
    main()
