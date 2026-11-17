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
