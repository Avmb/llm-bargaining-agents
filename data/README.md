# Bargaining scenarios dataset

`scenarios_by_reservation_ranges.jsonl` is a single JSON object keyed by four
price tiers (`low`, `medium`, `high`, `very_high`). Each tier maps to a list of
scenarios; the four tiers contain 1516, 869, 886, and 1290 scenarios
respectively (4561 in total). The experiments in the paper use the first ten
scenarios of the `low` tier.

Each scenario is an object with:

| field | description |
|-------|-------------|
| `product_name` | short name of the traded commodity (e.g. "1 kg of white rice") |
| `product_description` | 2–3 sentence description of the item |
| `buyer_persona` | 2–3 sentence, second-person description of the buyer's situation/incentives |
| `seller_persona` | 2–3 sentence, second-person description of the seller's situation/incentives |
| `seller_res_price_range` | `[lo, hi]`; the seller's reservation price is sampled `~ Uniform[lo, hi]` per trial |
| `buyer_res_price_range` | `[lo, hi]`; the buyer's reservation price is sampled `~ Uniform[lo, hi]` per trial |

The two ranges are the lower and upper halves of the generator's overall price
band split at its midpoint, so `seller_res_price_range[1] == buyer_res_price_range[0]`
and every sampled trial satisfies `v_B > v_S`.

The generation pipeline that produced this file is in
`../scenario_generation/`.
