"""E0: reproduce the starter gap with 5 init seeds, add a fills-only
ablation, a Kalman oracle reference, and window exposure statistics."""
import common
from common import (INITS, SEEDS, get_model, load, make_split, rmse,
                    run_parallel, save, summarize)
import numpy as np
import oracle

MODES = ['age', 'zero', 'fill']


def warm(seed, train_kind, mode, init):
  get_model(load(seed), seed, train_kind, mode, init)


def run_lengths(valid_col):
  runs, cur = [], 0
  for v in valid_col:
    if not v:
      cur += 1
    elif cur:
      runs.append(cur)
      cur = 0
  if cur:
    runs.append(cur)
  return np.array(runs)


def per_seed(seed):
  d = load(seed)
  std0 = d['std'][0]
  m, _ = oracle.recover(seed)
  out = dict(seed=seed, conditions={})
  for kind in ['scatter', 'block']:
    s = make_split(d, 'dev', kind, seed + 20, 'zero')
    y_raw = d['dev_target'][s['ends'] + 6]
    kf_mean, kf_var = oracle.kalman_forecast(s['raw'], m, 'dev', s['ends'])
    kf0, _ = oracle.kalman_forecast(s['raw'], m, 'dev', s['ends'], sensors=[0])
    art = np.isnan(s['raw']) & ~d['dev_natural_missing']
    a0 = s['age_end'][:, 0]
    rec = dict(
        n=int(len(s['y'])),
        persistence=rmse(s['persistence'], s['y'], std0),
        kalman=float(np.sqrt(np.mean((kf_mean - y_raw) ** 2))),
        kalman_sensor0_only=float(np.sqrt(np.mean((kf0 - y_raw) ** 2))),
        kalman_expected=float(np.sqrt(kf_var.mean())),
        exposure=dict(
            age0_mean=float(a0.mean()),
            age0_hist={str(k): int((np.minimum(a0, 9) == k).sum()) for k in range(10)},
            age0_ge3=float((a0 >= 3).mean()),
            age_all_mean=float(s['age_end'].mean()),
            min_age_across_sensors_gt0=float((s['age_end'].min(1) > 0).mean()),
            n_stale_sensors_mean=float((s['age_end'] > 0).sum(1).mean()),
            obs_count_s0_mean=float(s['obs_count'][:, 0].mean()),
            obs_count_all_mean=float(s['obs_count'].mean()),
            obs_count_s0_min=int(s['obs_count'][:, 0].min()),
            artificial_missing_per_sensor=art.sum(0).tolist(),
            gap_run_mean_s0=float(run_lengths(s['valid'][:, 0]).mean()),
            gap_run_max_s0=int(run_lengths(s['valid'][:, 0]).max()),
            co_missing_s0_and_any_other=float((~s['valid'][:, 0] & (~s['valid'][:, 1:]).any(1)).mean()),
        ),
        gru={})
    for mode in MODES:
      sm = make_split(d, 'dev', kind, seed + 20, mode)
      rm = [rmse(get_model(d, seed, 'scatter', mode, i).predict(sm['x']), sm['y'], std0)
            for i in INITS]
      rec['gru'][mode] = dict(per_init=rm, **summarize(rm))
    out['conditions'][kind] = rec
  sc, bl = out['conditions']['scatter'], out['conditions']['block']
  out['gap'] = dict(
      gru_zero=bl['gru']['zero']['mean'] - sc['gru']['zero']['mean'],
      gru_age=bl['gru']['age']['mean'] - sc['gru']['age']['mean'],
      gru_fill=bl['gru']['fill']['mean'] - sc['gru']['fill']['mean'],
      persistence=bl['persistence'] - sc['persistence'],
      kalman=bl['kalman'] - sc['kalman'],
      kalman_sensor0_only=bl['kalman_sensor0_only'] - sc['kalman_sensor0_only'])
  return out


if __name__ == '__main__':
  run_parallel(warm, [(s, 'scatter', m, i) for s in SEEDS for m in MODES for i in INITS])
  res = run_parallel(per_seed, [(s,) for s in SEEDS])
  for r in res:
    c = r['conditions']
    print(r['seed'], {k: round(v, 4) for k, v in r['gap'].items()})
    for kind in c:
      print('  ', kind, 'gru zero %.4f age %.4f fill %.4f pers %.4f kf %.4f kf0 %.4f' % (
          c[kind]['gru']['zero']['mean'], c[kind]['gru']['age']['mean'],
          c[kind]['gru']['fill']['mean'], c[kind]['persistence'], c[kind]['kalman'],
          c[kind]['kalman_sensor0_only']), 'age0>=3 %.3f' % c[kind]['exposure']['age0_ge3'])
  save('e0_baseline_exposure.json', res)
