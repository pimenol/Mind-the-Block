# Step 1: candidate explanations for the scatter-to-block gap

Written before running E1 to E5. The falsifiers are fixed here.

## Observation to explain

If the same 25% of readings per sensor is removed in 8-step blocks, a GRU trained with scattered gaps does worse on dev. Persistence also gets worse. The age input does not change the result. The replication unit is the data seed (940, 941, 942).

## Hypotheses

| ID | Explanation | Prediction | Falsified if |
|---|---|---|---|
| H1 (primary) | Stale own-sensor value at the forecast origin | Error is a function of sensor-0 age at the origin. The gap is a composition effect: block has more stale origins. | Block windows with age0 = 0 are still worse than scatter windows with age0 = 0, or the age-standardized gap stays large. |
| H2 | Lost temporal context: an outage damages the hidden state wherever it sits in the window | An 8-step block early in the window (age0 = 0 at the origin) also costs error. | An early block costs about nothing. |
| H3 (main competitor) | Training-support shortcut: scatter training taught the GRU that the filled value is almost always fresh. The model over-trusts stale fills. This is a model failure, not lost information. | The Bayes-optimal Kalman gap is much smaller than the GRU gap. A block- or mixed-trained GRU at the same budget closes the excess at high age. The GRU sensitivity to the stale fill is too high compared with the Kalman sensitivity. | The oracle gap is about equal to the GRU gap, and block training does not help at age0 >= 3. |
| H4 | Cross-sensor substitution failure: when sensor 0 is stale, the GRU does not fall back on the fresh correlated sensors | Kalman with all sensors beats Kalman with sensor 0 only mostly at high age0. Blocks on sensor 0 alone cause most of the gap. A ridge model trained with stale examples beats the GRU mostly at high age0. | Blocks on the other sensors cost as much as blocks on sensor 0, or the ridge advantage is the same at all ages. |

## Experiments

| ID | Tests | Design |
|---|---|---|
| E0 | Baseline | Reproduce the starter. Exposure: ages at origin, observed counts per window, co-missingness. Ablation arm with fills only. |
| E1 | H1 | Age-stratified error by sensor-0 age at the origin. Age-standardized gap. |
| E2 | H2, H4 | Controlled 8-step block inserted at offset 0, 4, 8, 16 from the window end, on sensor 0, other sensors, or all sensors. Same windows and targets. |
| E3 | H3, H4 | Kalman filter on the recovered true model: Bayes-optimal error under each mask, with all sensors and with sensor 0 only. |
| E4 | H3 | GRU trained on scatter, block, or mixed masks at the same budget. Sensitivity of the forecast to the stale fill, by age. |
| E5 | H4 | 2x2 factorial mask (sensor 0 pattern x other sensors pattern) and a ridge model on last-step features. |

## Step 1b (2026-11-17): why the GRU does not learn age discounting

Written before running E6 and E7. The falsifiers are fixed here. Step 1 showed that the GRU does not discount a held sensor-0 value by its age, and that more epochs, more units, and block or mixed training do not fix it. Step 1b asks why. H5 asks if the information is in the hidden state. H6 asks if the network can learn the right function when the target has no noise.

| ID | Explanation | Prediction | Falsified if |
|---|---|---|---|
| H5 (representation, not readout) | The hidden state h at the origin does not encode how stale sensor 0 is. Under a held input, the state relaxes to an input-driven fixed point in a few steps, so the time since the last reading is erased. The readout cannot discount what h does not hold. | (a) A binary probe for staleness (age0 >= 3 vs < 3) on h has balanced accuracy near a shuffled-label control, for a linear probe and a kNN probe. (b) With all inputs held, h converges in about 3 steps, and the spectral radius of the Jacobian dh_t/dh_(t-1) at the fixed point is well below 1. (c) A linear readout refit on frozen h removes less than 50% of the stale-origin excess over the window oracle. | Probe balanced accuracy >= 0.75 with selectivity >= 0.15 over the control, or the mixed-trained GRU with a mixed-window readout refit removes >= 50% of the excess. Then the information is in h and the readout is the bottleneck. |
| H6 (learnability, not expressivity) | The 8-unit GRU can represent age discounting, but noisy targets give too weak a gradient for it. Sub-question: target noise or too few stale windows. | The same GRU (same budget, init, minibatch order, mixed-mask train windows), trained on the noise-free 24-step window-oracle forecast, cuts the stale-origin excess (age0 >= 3) on dev block by >= 50%, and its sensitivity to the held value falls with age. Fixed Gaussian noise added to the teacher, with scale a x the oracle standard deviation, a in {0, .25, .5, 1}, raises the excess monotonically. a = 1 matches the real-target arm. Oversampling stale-origin windows 5x with real targets separates noise from sample count. | The distilled GRU keeps >= 50% of the excess, or its sensitivity at age 9+ stays >= 0.15. Then the limit is the architecture or its inductive bias, not the training signal. |

The window oracle is a teacher for a diagnostic only. It uses the true generator parameters and only the readings inside the window, so no future value enters an input. It is not a method.

| ID | Tests | Design |
|---|---|---|
| E6 | H5 | Staleness probes on h (linear, kNN, shuffled control, age-input GRU as positive control). Readout refit on frozen h in three cells. Fixed-point relaxation with all inputs held, Jacobian spectral radius, and the forecast under a held sensor 0. |
| E7 | H6 | Distillation from the window oracle with a noise dose-response, a stale-oversampling arm with real targets, and 48-epoch controls. |

### Step 1b post-hoc additions (2026-11-17, written after the first E6 and E7 runs)

These analyses were added after seeing results. They do not change the falsifiers above.

- E6: a second probe among stale windows only (age0 >= 1), age0 >= 3 vs 1-2. The registered probe (age0 >= 3 vs < 3) can use the last-step mask bit, so it does not separate "old or not" from "how old".
- E7: the lag-1, 3 and 6 autocorrelation of the real residual (target minus teacher) on train windows. It explains why the a = 1 arm does not match the real-target arm in 2 of 3 seeds.

Verdicts by the registered criteria: H5 (a) is met for the falsifier in 2 of 3 seeds (mixed-trained probe >= 0.75), (b) and (c) hold in 3 of 3 seeds, so H5 is partly right. H6 is falsified in 3 of 3 seeds.

## Step 1c (2026-11-17): does the mechanism hold beyond the 8-unit GRU and beyond synthetic data?

Written before running E8 and E9. The falsifiers are fixed here. Step 1b found that the GRU memory settles in a few steps and keeps "old or not" but not "how old" (H5 partly right), and that a clean teacher does not teach the discount (H6 rejected). Step 1c asks two questions. First, is the failure specific to a recurrent memory? A transformer reads every step directly and has no settling memory. If settling (H5) were the main cause, a transformer would discount old values. Second, does the failure appear on real data with natural outages?

New models use PyTorch (CPU) and live only in `research/`. The starter and the NumPy GRU are not changed.

| ID | Claim | Falsified if |
|---|---|---|
| T1 | A small transformer (2 layers, 2 heads, width 32, last-token readout, Adam, same train windows and the same 12 x 1600 budget) shows the same failure on synthetic data: its excess over the 24-step window oracle is >= 50% of its block-minus-scatter MSE gap. | Excess share < 50% in 2 of 3 seeds. |
| T2 | The transformer's trust in the last sensor-0 value does not fall with age: `e4.gru_sensitivity` at age 9+ >= 0.15 (the Kalman value is about 0.01). | Trust at age 9+ < 0.15 in 2 of 3 seeds. |
| T3 (tests H5 as the cause) | If T1 and T2 hold, a settling memory is not needed for the failure. The transformer's final representation holds "how old" (linear probe, stale windows only, age0 >= 3 vs 1-2) with balanced accuracy >= 0.85, but its trust curve stays flat. The information is present and unused. | Probe < 0.85 in 2 of 3 seeds. Then the transformer also drops the age, and the representation account stays open. |
| T4 (optimizer control) | A PyTorch GRU (8 units, Adam, same data and budget) behaves like the NumPy GRU (plain SGD with clipping): trust at age 9+ >= 0.15. So a transformer difference is due to the architecture, not the optimizer. | Torch GRU trust at age 9+ < 0.15 in 2 of 3 seeds. Then the optimizer is a lever by itself. |
| R1 | Real data (UCI Beijing Multi-Site Air Quality, PM2.5 at 6 stations, hourly, 24-step window, 6-step horizon): for the scatter-trained GRU, windows with the target station 3 or more steps stale at the origin carry >= 50% of the block-minus-scatter MSE gap (gap summed over windows). | < 50% in 2 of 3 target stations. |
| R2 | On the real data, an age-aware linear reference (ridge on train, last observed value of each station x one-hot age bin) gives a weight on the target station's last value that falls with age, and the scatter-trained GRU trust curve does not fall. | The reference weight does not fall from age 0 to age 9+. Then trusting a stale value is close to right on this data. This is reported as a limit, not as support. |

On real data there is no true model, so no oracle. The ridge reference is a learned model, not a lower bound. Replication on real data uses 3 target stations, each with 5 GRU or transformer inits. The real test period (the last 6 months) is never loaded.

| ID | Tests | Design |
|---|---|---|
| E8 | T1-T4, H2-H6 on a transformer | Transformer and torch GRU on the synthetic seeds: excess over the window oracle, trust by age, moving outage (E2), held sensor 0 curve, probes on the final representation, attention from the last token by age, mixed training, clean teacher, 48 epochs. |
| E9 | R1, R2, H3 on real data | Beijing PM2.5 with imposed scatter and block masks on top of the natural gaps: gap by age bin, trust curve against the ridge reference, scatter vs mixed training, GRU and transformer. |

### Step 1c verdicts (2026-11-17, after E8 and E9)

By the registered criteria: T1 holds in 3 of 3 seeds (model-failure share 0.70 to 0.92). T2 holds in 2 of 3 seeds (trust at age 9+ 0.27, 0.21, 0.12). T3 is falsified in 3 of 3 seeds ("how old" probe 0.56 to 0.57): the transformer also drops the age. T4 holds in 2 of 3 seeds (0.26, 0.16, 0.11). R1 holds in 2 of 3 stations (0.66, 0.74, 0.48). R2 holds in 3 of 3 stations: the reference trust falls from age 0 to 9+, the GRU trust rises.

Not registered, reported as description: transformer attention from the last token is close to uniform (0.036 to 0.059 per step) and does not change with age. On real data the GRU error on stale windows is only 1% to 11% above the ridge reference.
