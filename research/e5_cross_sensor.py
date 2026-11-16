"""E5 (H4): which sensors' outages cause the gap, and can a simple learned
model use the remaining cross-sensor information without oracle knowledge?

1. 2x2 factorial mask on real starter masks: sensor-0 pattern {scatter,
   block} x other-sensor pattern {scatter, block}. Same windows, same targets.
2. Ridge regression on the flattened 24x18 window plus last-step
   fill x fresh and fill x age interactions (lets a linear model gate trust
   by age). Fit on train windows under scatter or block masks, fixed
   lambda = 1 (not tuned on dev). This is a learned, non-oracle reference.
"""
import common
from common import INITS, SEEDS, get_model, load, make_split, rmse, run_parallel, save, summarize, train_set
import numpy as np
import oracle

LAMBDA = 1.0
FACTORIAL = {
    'S0 scatter / others scatter': ('scatter',) * 6,
    'S0 block / others scatter': ('block',) + ('scatter',) * 5,
    'S0 scatter / others block': ('scatter',) + ('block',) * 5,
    'S0 block / others block': ('block',) * 6,
}


def ridge_features(x):
  last = x[:, -1]
  fill, mask, age = last[:, :6], last[:, 6:12], last[:, 12:]
  return np.concatenate([x.reshape(len(x), -1), fill * mask, fill * age, np.ones((len(x), 1))], 1)


def fit_ridge(x, y):
  f = ridge_features(x)
  reg = LAMBDA * np.eye(f.shape[1])
  reg[-1, -1] = 0
  return np.linalg.solve(f.T @ f + reg, f.T @ y)


def per_seed(seed):
  d = load(seed)
  std0 = d['std'][0]
  m, _ = oracle.recover(seed)
  ridges = {k: fit_ridge(*train_set(d, k, 'age', seed)) for k in ['scatter', 'block']}
  grus = {k: [get_model(d, seed, k, 'zero', i) for i in INITS] for k in ['scatter', 'block']}
  out = dict(seed=seed, factorial={})
  errs = {}
  for name, kinds in FACTORIAL.items():
    s = make_split(d, 'dev', kinds, seed + 20, 'age')
    sz = make_split(d, 'dev', kinds, seed + 20, 'zero')
    y_raw = d['dev_target'][s['ends'] + 6]
    e = {}
    for k, ms in grus.items():
      e[f'gru_{k}'] = np.mean([(mm.predict(sz['x']) - sz['y']) ** 2 for mm in ms], 0)
    for k, w in ridges.items():
      e[f'ridge_{k}'] = (ridge_features(s['x']) @ w - s['y']) ** 2
    e['persistence'] = (s['persistence'] - s['y']) ** 2
    kf, _ = oracle.kalman_forecast(s['raw'], m, 'dev', s['ends'])
    kf0, _ = oracle.kalman_forecast(s['raw'], m, 'dev', s['ends'], sensors=[0])
    e['kalman'] = ((kf - y_raw) / std0) ** 2
    e['kalman_s0'] = ((kf0 - y_raw) / std0) ** 2
    errs[name] = (e, s['age_end'][:, 0])
    out['factorial'][name] = {k: float(np.sqrt(v.mean()) * std0) for k, v in e.items()}
  base = errs['S0 scatter / others scatter']
  full = errs['S0 block / others block']
  stale = (base[1] == 0) & (full[1] >= 5)
  fresh = (base[1] == 0) & (full[1] == 0)
  out['n_stale'], out['n_fresh'] = int(stale.sum()), int(fresh.sum())
  out['paired_block_minus_scatter'] = {
      k: dict(stale=float((full[0][k] - base[0][k])[stale].mean() * std0 ** 2),
              fresh=float((full[0][k] - base[0][k])[fresh].mean() * std0 ** 2),
              all=float((full[0][k] - base[0][k]).mean() * std0 ** 2)) for k in base[0]}
  return out


if __name__ == '__main__':
  res = run_parallel(per_seed, [(s,) for s in SEEDS])
  for r in res:
    print('seed', r['seed'], 'n_stale', r['n_stale'], 'n_fresh', r['n_fresh'])
    for name, v in r['factorial'].items():
      print('  %-28s' % name, {k: round(x, 4) for k, x in v.items()})
    for k, v in r['paired_block_minus_scatter'].items():
      print('   paired %-15s' % k, {a: round(b, 4) for a, b in v.items()})
  save('e5_cross_sensor.json', res)
