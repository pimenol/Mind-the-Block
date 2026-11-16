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
