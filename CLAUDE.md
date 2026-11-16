# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Course research starter (Project 04): does a forecaster cope with a *new missingness pattern*? Pure NumPy, CPU only, no DL framework. Data is a **synthetic** six-sensor benchmark (not Beijing) — claims must stay scoped to this benchmark. Not a git repo.

## Project definition (course brief)

**Question:** why can a forecasting model fail when the same fraction of measurements is missing in longer bursts rather than scattered gaps? Occasional misses vs. an hour-long outage are different information problems; a model trained on one pattern may learn a shortcut that doesn't transfer.

**Starting point:** run persistence and the supplied GRU, compare dev errors under scatter vs. block masks, track time since most recent observation as well as missing fraction. Imputation uses past observations only; means/scaling fit on train. Synthetic data supports mechanism tests, not claims about a real sensor network.

**Research steps required:**
1. Propose an explanation for the gap. Pick one and name a competing explanation.
2. Make a comparison that distinguishes them, e.g. age inputs vs. same GRU with ages zeroed, then check whether a gap remains after controlling the relevant ages. Matching overall missing fraction alone does not isolate an age effect.
3. Develop a targeted training change, a diagnostic for when predictions are unreliable, or a well-supported limitation. Compare against persistence and a mask-aware recurrent baseline at matched budgets. Test the explanation, not just the model ranking.
4. Freeze main choices before evaluating the reserved mechanism. Report results across seeds, actual exposure of forecasting windows (ages, observed counts), and the boundary of any synthetic-data conclusion. A real-data extension is welcome but must not replace the controlled test.

**Report must:** explain a mechanism behind the shift and test a response to it, or establish a useful limit. "Outages hurt" or "observation age helps" is already expected and is not enough on its own.

**Background:** Che et al. (2018), GRU-D. It already uses observation timing, so treat it as background, not a contribution. Course link: RNNs/GRUs.

## Commands

```bash
pip install -r requirements.txt          # numpy only
python check_data.py                     # integrity + mask counts + GRU finite-diff grads + causality; the "test suite"
python starter.py --quick                # fast smoke run (2 epochs, 320 samples/epoch)
python starter.py --seed 940             # standard run on dev; repeat for 941, 942
python starter.py --final --seed 940     # held-out test period + value-dependent mask
python generate_data.py                  # regenerate data/*.npz + metadata.json deterministically
```

Output is JSON to stdout. Standard run takes ~3s/seed (see `VALIDATION.json` for reference numbers).

## Hard rules from the assignment

- **Never run `--final` or tune on its results** until claim, preprocessing, method and settings are frozen. Don't run it casually for checking.
- Targets are for loss/eval only — never use them (or any future value) to fill inputs. Normalization stats come from training observations only.
- Overlapping forecast windows aren't independent replicates; the replication unit is the seed. Report per-seed results + uncertainty.
- Adding an age input alone is not a research contribution; matching total missing fraction alone is not a sufficient control.
- `MANIFEST_SHA256.json` hashes the shipped files — edits to `starter.py` etc. will break those hashes (expected; don't "fix" by editing the manifest unless asked).

## Architecture (starter.py)

- `impose(observed, kind, seed)` — adds artificial missingness on top of natural NaNs. Kinds: `scatter` (train), `block` (8-step outages, dev), `value` (high values hidden, final only). Removes exactly `int(.25 * n_valid)` per sensor for every kind, so counts match across mechanisms — but observation age / co-missingness per window do **not** match.
- `windows(raw, target, mean, std, use_age)` — causal forward fill (partition starts from train mean), builds features `[normalized filled values, observed mask, age/48 or zeros]` → 18 features. Lookback 24, horizon 6, target is sensor 0. Also returns persistence baseline (last filled sensor-0 value) and window end indices.
- `GRU` — 8-unit NumPy GRU with hand-written BPTT in `step` (grad-norm clip 5, lr .025). `check_data.py` finite-difference-checks every param; rerun it after touching the GRU math.
- `run` — trains two arms (age features vs. zeros in the same slots) with identical init, examples and minibatch order (both seeded by `seed`), then evaluates each on dev (`scatter`, `block`) or test (+`value`). Mask seeds: train `seed+10`, eval `seed+20`. Targets that are naturally missing are dropped; artificial masks never change target selection, so all conditions/methods are scored on the same examples.

Data files: `data/synthetic_{940,941,942}.npz` with `{train,dev,test}_{observed,natural_missing,target}`; splits are chronological 2800/1000/1000.
