# Stale-Fill Trust: Project 04

This file gives guidance to Claude Code (claude.ai/code) for work in this repository.

Course research project (Project 04, course link: RNNs and GRUs). The same
fraction of readings can be missing in long bursts or in scattered gaps. The
question: in the burst case, why can a forecaster fail? The code is pure NumPy on CPU. There is no
deep-learning framework and no GPU. The data is a synthetic six-sensor benchmark,
not the Beijing dataset, so every claim stays scoped to this benchmark.

The repository is a git repo on branch `main`. The shipped starter (`starter.py`,
`check_data.py`, `generate_data.py`, `data/`) comes from the course. All research
code is in `research/`, all result files are in `results/`, and the report is in
`report/`.

## Design philosophy

The project explains a mechanism. It does not chase a better model ranking. To
count, a result must separate two explanations that make different predictions.

1. Falsifiers come first. `research/HYPOTHESES.md` fixed the hypotheses H1 to H4
   and their falsifiers before experiments E1 to E5 ran. You do not edit a
   falsifier after you see a result. If a new step needs new hypotheses, add a
   new section with a date. Do not rewrite the old one.
2. Paired comparisons on the same windows. Every condition and every method is
   scored on the same forecast windows and the same targets. Artificial masks
   only remove inputs. They never change which targets are scored.
3. The oracle is a measuring tool, not a method. `research/oracle.py` replays the
   generator's random numbers to recover the true model, then runs a Kalman
   filter. Its error is a lower bound. Use its gap between masks to split a
   model's gap into "information lost" (oracle gap) and "model failure" (excess
   over the oracle). Never present the oracle as a competing forecaster.
4. The replication unit is the data seed (940, 941, 942). Overlapping windows are
   not independent. Each seed uses 5 GRU initializations (`INITS`), averaged
   inside the seed. Report per-seed values and the range over seeds.
5. Exposure is reported with every result. Counting removed readings does not
   control observation age. Report sensor-0 age at the forecast origin, observed
   counts per window, and the number of distinct outages behind stale windows.
6. Negative results are results. If a fix does not work, report it with the same
   care as a fix that works.

## Hard rules from the assignment

- Never run `python starter.py --final` or any code that reads the `test_*`
  arrays until the claim, preprocessing, method and settings are frozen and the
  user asks for it. Never tune on final results. `research/common.load` exposes
  only `train_*` and `dev_*`, and `test_no_test_partition_access` enforces it.
- Targets are for loss and evaluation only. Never use a target or any future
  value to fill an input. Fill is causal (forward fill from the train mean).
- Mean and scale come from training observations only.
- An age input alone is not a contribution (GRU-D, Che et al. 2018, already uses
  timing). Matching the total missing fraction alone is not a sufficient control.
- `MANIFEST_SHA256.json` hashes the shipped files. Do not edit `starter.py`,
  `check_data.py` or `generate_data.py`. Do not edit the manifest unless the user
  asks. New behavior goes in `research/`.
- The final test changes two things at once: a later time period and a
  value-dependent mask. To attribute an effect to the mask, also evaluate the
  frozen methods on `scatter` and `block` masks on the same test period.

## Status

Step 1 (mechanism) is done, commits "Step 1a" to "Step 1h". Step 1b (why the
GRU does not learn age discounting, H5 and H6, E6 and E7) is done. Step 2 (a
targeted training change, a reliability diagnostic, or a scoped limit) and the
frozen final evaluation are not started.

Step 1 claim (dev only, 3 of 3 seeds): 52% to 90% of the scatter-trained GRU's
block-minus-scatter MSE gap is model failure, not lost information. The trigger
is a stale value at the forecast origin, on sensor 0 or on any other sensor. The
cause is that the forecast does not discount a held value by its age.

| ID | Explanation | Verdict |
|---|---|---|
| H1 | Stale sensor-0 value at the origin | Trigger, not cause. The oracle loses 3 to 50 times less on the same stale windows. |
| H2 | Lost temporal context anywhere in the window | Rejected. An outage that ends 8 or 16 steps before the origin costs about 0 MSE. |
| H3 | Training-support shortcut (scatter training never shows stale fills) | Half right. Block or mixed training at the same budget does not remove the stale-origin loss. |
| H4 | Cross-sensor substitution failure | Holds, but wider. Outages on the 5 other sensors also cost the GRU much more than the oracle. |
| H5 | The hidden state forgets how old a value is | Partly right. Probes read "old or not" (0.71 to 0.81) but not "how old" (0.64 to 0.69, input 0.99 to 1.00). Jacobian radius 0.59 to 0.65. A new readout on frozen h removes at most 1% of the stale excess. |
| H6 | Targets too noisy to learn the discount | Rejected (3 of 3). A clean window-oracle teacher keeps 0.81 to 1.07 of the stale excess. Oversampling stale windows lowers one fixed weight at all ages. |

Numbers that later steps build on:

- Kalman sensitivity to the last sensor-0 reading drops from 0.32 at age 0 to
  0.01 at age 9+. The GRU sensitivity goes from 0.21 to 0.28.
- The age input changes block RMSE by at most 0.0003. At 48 epochs the trust
  curve starts to bend down with age (E4b, E7), but 58% to 90% of the stale
  excess remains (E7). Learning the discount is possible but very slow.
- Block masks make sensor 0 stale at the origin in about 1 in 5 windows, scatter
  masks in about 1 in 50. The stale bins come from only 21 to 23 distinct
  outages per seed, so the controlled intervention E2 is the clean estimate.
- Seed 942 has about 3 times the error scale of the other seeds. A mean over
  seeds is dominated by 942. Always show per-seed values next to a mean.

The full write-up is `report/report.html`. The source of truth for every number
is `results/*.json`.

## Commands

You run every command from the repository root inside a Python 3 environment
with NumPy. Python 3 and NumPy 2.3.5 are the tested set-up.

```bash
pip install -r requirements.txt            # numpy only
python check_data.py                       # data integrity, mask counts, GRU finite-difference gradients, causality
python research/test_research.py           # research helpers: features equal starter, oracle, VALIDATION.json, no test access
python starter.py --quick                  # smoke run (2 epochs, 320 samples per epoch)
python starter.py --seed 940               # standard dev run, repeat for 941 and 942 (about 3 s each)
python generate_data.py                    # rebuild data/*.npz and metadata.json, deterministic
```

Experiments are plain scripts. Each one trains or loads its GRUs, prints a
per-seed summary, and writes one JSON file to `results/`.

```bash
python research/e0_baseline_exposure.py    # E0: baseline, fills-only arm, exposure statistics
python research/e1_age_stratified.py       # E1: error by sensor-0 age, paired gap, composition split
python research/e2_block_position.py       # E2: one 8-step outage moved through the window
python research/e3_window_oracle.py        # E3: 24-step window oracle, distinct-outage counts
python research/e4_training_masks.py       # E4: scatter / block / mixed training, stale-value sensitivity
python research/e4b_budget_capacity.py     # E4b: epochs x hidden units (12, 48) x (8, 32)
python research/e5_cross_sensor.py         # E5: 2x2 sensor factorial mask and ridge reference
python research/e6_representation.py      # E6: probes on h, readout refit, fixed-point dynamics
python research/e7_distillation.py         # E7: window-oracle teacher, noise doses, stale oversampling, 48 epochs
python report/build_report.py              # results/*.json + report/template.html -> report/report.html
```

The scripts use `multiprocessing` (up to 32 processes, 48 for E4b) and set the
BLAS thread variables to 1. If you run on a login node, reduce `procs` in
`run_parallel`.

## Architecture

### Starter (`starter.py`, do not edit)

- `impose(observed, kind, seed)` adds artificial missingness on top of natural
  NaNs. Kinds: `scatter` (train), `block` (8-step outages, dev), `value` (high
  values hidden, final only). Each kind removes exactly `int(.25 * n_valid)`
  readings per sensor. Counts match across kinds, but age and co-missingness per
  window do not.
- `windows(raw, target, mean, std, use_age)` makes 18 features per step:
  normalized filled values, observed mask, and age/48 (or zeros). Lookback 24,
  horizon 6, target sensor 0. It also returns the persistence forecast.
- `GRU` is an 8-unit NumPy GRU with hand-written BPTT (gradient-norm clip 5,
  learning rate 0.025). If you change the GRU math, run `check_data.py` again.
- `run` trains two arms (age features, or zeros in the same slots) with the same
  initialization and minibatch order. Mask seeds: train `seed+10`, eval `seed+20`.

### Research code (`research/`)

- `common.py` holds the shared helpers. `make_split` rebuilds `starter.windows`
  exactly (a test asserts this) but accepts any mask, including a 6-tuple of
  kinds per sensor. Feature modes: `age` (full), `zero` (ages zeroed, the
  starter arm), `fill` (mask and age channels zeroed). `train_set` with `mixed`
  picks scatter or block per window, 50/50, at an unchanged budget.
- `get_model` trains a GRU or loads it from the cache in `results/models/`. The
  file name is `{seed}_{train_kind}_{mode}_{init}[_e{epochs}_h{hidden}].npz`.
  `init=0` reproduces the starter exactly. The cache is not in git.
  `cached_model(name, build)` does the same for custom arms (E7 uses
  `{seed}_mixed_zero_{init}_{arm}.npz`).
- `oracle.py` is the Bayes-optimal reference: generator replay (`recover`), a
  full-history Kalman filter (`kalman_forecast`), and a filter restarted per
  window (`batched_restart`). It skips missing readings.
- `test_research.py` is a plain script, not pytest. It runs every `test_*`
  function and prints `PASS`.
- `HYPOTHESES.md` holds the pre-registered hypotheses and falsifiers.

### Report (`report/`)

`build_report.py` reads every `results/*.json`, fills the `{{KEY}}` slots in
`template.html`, and writes `report.html`. An empty slot makes it fail. Every
number in the report must come from a result file. Do not type numbers into the
template.

## Conventions

- Code style follows the existing files: 2-space indent, short helper functions,
  one script per experiment named `eN_<topic>.py`, one result file per script.
- New experiments use `common.py` helpers (`load`, `make_split`, `get_model`,
  `run_parallel`, `save`, `summarize`) and keep `INITS`, `SEEDS` and the mask
  seeds unchanged, so new results stay paired with the old ones.
- If you change training code in `common.py`, delete the affected files in
  `results/models/` first. Otherwise `get_model` loads stale weights.
- After you add a result file, run `python research/test_research.py` and
  `python report/build_report.py` again.
- Commit messages are one short line in the form `<type>: <description>` (for
  example `docs: add HTML report`), with no `Step` prefix and no attribution
  lines. `.claude/rules/git.md` lists the types.
- Write report text in plain, short sentences. State the seeds behind each
  claim ("3 of 3 seeds") and the range over seeds.

## Layout

```
starter.py, check_data.py, generate_data.py   # shipped starter (hashed in MANIFEST_SHA256.json)
VALIDATION.json                               # reference numbers for the standard runs
data/synthetic_{940,941,942}.npz              # {train,dev,test}_{observed,natural_missing,target}, 2800/1000/1000
data/metadata.json
research/                                     # hypotheses, helpers, oracle, tests, experiments E0-E7
results/eN_*.json                             # one file per experiment, committed
results/models/                               # GRU weight cache (gitignored)
report/template.html, build_report.py         # report source
report/report.html                            # built report
```
