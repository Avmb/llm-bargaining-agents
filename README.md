# Used Car Salesbots? Honesty and Credulity of LLMs as Bargaining Agents under Partial Information

Code, data, and experiment outputs for the paper studying LLM agents that
bargain over a text channel under varying information transparency, and the
effect of reinforcement-learning fine-tuning on their honesty and credulity.

> Paper: *https://arxiv.org/abs/2605.31445*

## What's here

```
data/                 The bargaining-scenarios dataset + schema (data/README.md)
scenario_generation/  The pipeline that generated the dataset
core/                 Reference notebook defining the simulator/task/prompt/judge code,
                      plus the honesty/credulity judge rubrics (honesty_templates.py)
analysis/             Cross-experiment analysis (the honesty-vs-utility scatter plots)
zeroshot_eval/        Zero-shot 5-model evaluation notebooks (paper Section 4)
rl/                   RL fine-tuning code, configs, launch scripts, and eval notebooks
                      (paper Section 5 + appendices)
results/              Pre-computed experiment outputs cited in the paper
   zeroshot/          one directory per model
   rl_evals/          One dir per fine-tuned variant (paired vs base)
```

Each `results/.../` directory contains `config/experiment.json` (run
configuration), `data/*.pkl` (per-trial dataframes, including deal prices and
judge honesty / credulity scores), and `figures/` (auto-generated analysis
plots). The raw per-trial dialogue logs are not included to keep the repository
small; representative annotated dialogues are reproduced in the paper appendix,
and re-running the notebooks regenerates the full logs under `./experiments/`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

API keys and tokens are read from the environment (never from files in this
repo). Export whatever the experiment you want to run needs:

```bash
export OPENAI_API_KEY=...      # GPT-5.x agents + the GPT-5.2 judge
export ANTHROPIC_API_KEY=...   # claude-* agents
export HF_TOKEN=...            # downloading Qwen weights for RL training
export WANDB_API_KEY=...       # optional: training logging (or set wandb.enabled=false in the config)
```

Open-weight models (Qwen3.5-9B for the zero-shot evals, Qwen3-8B for RL) are
served locally with [vLLM](https://docs.vllm.ai). The notebooks talk to a vLLM
server over an OpenAI-compatible endpoint (default `http://localhost:8000/v1`);
set `VLLM_SSH_HOST` if the server runs on another machine you reach over SSH.

## Dataset

`data/scenarios_by_reservation_ranges.jsonl` holds 4561 commodity-bargaining
scenarios across four price tiers; the paper uses the first ten `low`-tier
scenarios. See `data/README.md` for the schema. Regenerate it with
`scenario_generation/bargaining_scenario_generator.ipynb`. The dataset is also
on the HuggingFace Hub at
[AnvaMiba/llm-bargaining-scenarios](https://huggingface.co/datasets/AnvaMiba/llm-bargaining-scenarios).

## Reproducing the zero-shot model evaluation (Section 4)

Each notebook in `zeroshot_eval/` runs one model in self-play across the four
transparency conditions (`full`, `buyer_unaware`, `seller_unaware`,
`both_unaware`), 10 scenarios x 8 trials x 6 rounds = 320 trials, and has the
GPT-5.2 judge rate honesty/credulity:

| notebook | agents |
|----------|--------|
| `eval_gpt52.ipynb`   | GPT-5.2 |
| `eval_gpt55.ipynb`   | GPT-5.5 |
| `eval_sonnet46.ipynb`| claude-sonnet-4-6 |
| `eval_opus47.ipynb`  | claude-opus-4-7 |
| `eval_qwen35.ipynb`  | Qwen3.5-9B (local vLLM) |

The pre-computed outputs are in `results/zeroshot/`. Per-condition deal-rate,
welfare, NBS-deviation and honesty/credulity tables and plots are produced in
the later cells of each notebook (and saved under each result dir's `figures/`).

The cross-experiment honesty-vs-utility scatter plots (paper figures comparing
all models, and the trained variants against the base) are produced by
`analysis/make_honesty_utility_scatter.py`, which reads the dataframes in
`results/` and writes PDFs to `analysis/figures/`:

```bash
python analysis/make_honesty_utility_scatter.py
```

## Reproducing the RL fine-tuning (Section 5)

Training (`rl/bargaining_rl.py`) runs across **two nodes of 4 GPUs each**: one
node serves the opponent/rollout policy with vLLM, the other runs DDP
optimisation. We used NVIDIA GH200 nodes; the launcher is `rl/slurm_train.sh`
(Slurm; set your partition and `HF_TOKEN`/`WANDB_API_KEY` in the environment).

```bash
sbatch --export=CONFIG=configs/<config>.yaml,MODE=train rl/slurm_train.sh
```

Configs for the reported variants (`rl/configs/`):

| variant | config |
|---------|--------|
| buyer-side GRPO  | `train_buyer_with_seller_qwen3_no_think_v6_grpo.yaml` |
| buyer-side CISPO | `train_buyer_with_seller_qwen3_no_think_v6_cispo.yaml` |
| buyer no-rank ablation | `..._v6_grpo_noranktx.yaml`, `..._v6_cispo_noranktx.yaml` |
| seller-side GRPO/CISPO | `train_seller_with_buyer_qwen3_no_think_v6_{grpo,cispo}_largebatch.yaml` |
| joint self-play GRPO/CISPO | `train_joint_qwen3_no_think_v7_{grpo,cispo}_largebatch.yaml` |

Loss definitions (GRPO / CISPO) and the rank reward transform are documented in
the paper appendices.

## Evaluating checkpoints

`rl/eval_checkpoints.py` hot-loads a LoRA adapter into a running vLLM server and
runs the validation split. The paired comparisons reported in the paper
(trained vs. base, identical scenarios/seed) were produced by the notebooks in
`rl/eval_notebooks/` (named by role/loss); their pre-computed outputs are in
`results/rl_evals/`.

## Trained checkpoints

The fine-tuned LoRA adapters are **not** included in this repository. They are
released on the HuggingFace Hub at
[AnvaMiba/qwen3-8b-bargaining-lora](https://huggingface.co/AnvaMiba/qwen3-8b-bargaining-lora):
a single repo with one subfolder per variant (`buyer-grpo`, `buyer-cispo`,
`buyer-grpo-norank`, `buyer-cispo-norank`, `seller-grpo`, `seller-cispo`,
`joint-grpo`, `joint-cispo`). Load one with
`PeftModel.from_pretrained(base, "AnvaMiba/qwen3-8b-bargaining-lora", subfolder="<variant>")`.

## License

Released under the MIT License (see `LICENSE`).

## Citation

```bibtex
@misc{micelibarone2026usedcarsalesbotshonesty,
      title={Used Car Salesbots? Honesty and Credulity of LLMs as Bargaining Agents under Partial Information}, 
      author={Antonio Valerio Miceli-Barone and Vaishak Belle and Shay B. Cohen},
      year={2026},
      eprint={2605.31445},
      archivePrefix={arXiv},
      primaryClass={cs.GT},
      url={https://arxiv.org/abs/2605.31445}, 
}
```
