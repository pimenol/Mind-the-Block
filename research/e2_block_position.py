"""E2 (H2, H4): controlled block-position intervention.

Start from each dev window under the scatter mask. Insert one 8-step outage
that ends `offset` steps before the forecast origin, on sensor 0 only, on
the 5 other sensors, or on all sensors. Fill and age are recomputed causally
from the outage start, so everything after the block is exactly what the
model would see. Same windows and targets in every cell, so the cost of an
outage is a paired difference against the unmodified scatter window.

If H2 (lost temporal context) holds, an early block (offset 16, sensor 0
fresh at the origin) costs error. If H1 holds, only offset 0 costs error.
"""
import common
from common import INITS, SEEDS, get_model, load, make_split, run_parallel, save
import numpy as np
import oracle

OFFSETS = [0, 4, 8, 16]
SENSOR_SETS = {'s0': [0], 'others': [1, 2, 3, 4, 5], 'all': [0, 1, 2, 3, 4, 5]}
BLOCK = 8
MIN_END = common.LOOKBACK + 16 + BLOCK  # same windows for every offset


def intervene(s, d, offset, sensors, mode):
  """Return (features, persistence, use-matrix, starts) after inserting the
  outage into each window of split dict s."""
  mean, std = d['mean'], d['std']
  ends = s['ends']
  start = ends - offset - BLOCK + 1
  steps = offset + BLOCK
  x = s['x'].copy()
  prev = s['filled'][start - 1].copy()
  a = s['age'][start - 1].copy()
  blocked = np.zeros(6, bool)
  blocked[sensors] = True
  use = np.ones((len(ends), steps, 6), bool)
  for k in range(steps):
    t = start + k
    v = s['valid'][t].copy()
    if k < BLOCK:
      v[:, blocked] = False
      use[:, k, blocked] = False
    prev = np.where(v, s['raw'][t], prev)
    a = np.where(v, 0, a + 1)
    row = common.LOOKBACK - steps + k
    f = (prev - mean) / std
    m = v.astype(float)
    ag = np.minimum(a, 48) / 48
    if mode == 'zero':
      ag = np.zeros_like(ag)
    x[:, row] = np.concatenate([f, m, ag], 1)
  pers = (prev[:, 0] - mean[0]) / std[0]
  return x, pers, use, start, a[:, 0]


def evaluate(seed, train_kinds=('scatter',), mode='zero'):
  d = load(seed)
  std0 = d['std'][0]
  m, _ = oracle.recover(seed)
  s = make_split(d, 'dev', 'scatter', seed + 20, mode)
  keep = s['ends'] >= MIN_END
  for k in ['x', 'y', 'persistence', 'ends', 'age_end']:
    s[k] = s[k][keep]
  y_raw = d['dev_target'][s['ends'] + 6]
  models = {tk: [get_model(d, seed, tk, mode, i) for i in INITS] for tk in train_kinds}
  means, covs, t, c = oracle.kalman_states(s['raw'], m, 'dev')
  means0, covs0, _, _ = oracle.kalman_states(s['raw'], m, 'dev', sensors=[0])

  def errors(x, pers, use=None, start=None):
    e = {f'gru_{tk}': np.mean([(mm.predict(x) - s['y']) ** 2 for mm in ms], 0)
         for tk, ms in models.items()}
    e['persistence'] = (pers - s['y']) ** 2
    if use is None:
      kf, _ = oracle.kalman_forecast(s['raw'], m, 'dev', s['ends'])
      kf0, _ = oracle.kalman_forecast(s['raw'], m, 'dev', s['ends'], sensors=[0])
    else:
      kf, _ = oracle.batched_restart(means, covs, t, c, s['raw'], m, start, s['ends'], use)
      use0 = use.copy()
      use0[:, :, 1:] = False
      kf0, _ = oracle.batched_restart(means0, covs0, t, c, s['raw'], m, start, s['ends'], use0)
    e['kalman'] = ((kf - y_raw) / std0) ** 2
    e['kalman_s0'] = ((kf0 - y_raw) / std0) ** 2
    return e

  base = errors(s['x'], s['persistence'])
  out = dict(seed=seed, n=int(len(s['y'])), baseline={k: float(v.mean() * std0 ** 2) for k, v in base.items()},
             cells=[])
  for name, sensors in SENSOR_SETS.items():
    for off in OFFSETS:
      x, pers, use, start, age0 = intervene(s, d, off, sensors, mode)
      e = errors(x, pers, use, start)
      out['cells'].append(dict(
          sensors=name, offset=off, mean_age0_origin=float(age0.mean()),
          delta_mse={k: float((e[k] - base[k]).mean() * std0 ** 2) for k in e}))
  return out


if __name__ == '__main__':
  res = run_parallel(evaluate, [(s,) for s in SEEDS])
  for r in res:
    print('seed', r['seed'], 'n', r['n'], {k: round(v, 4) for k, v in r['baseline'].items()})
    for c in r['cells']:
      print('  %-6s off %2d age0 %.1f' % (c['sensors'], c['offset'], c['mean_age0_origin']),
            {k: round(v, 4) for k, v in c['delta_mse'].items()})
  save('e2_block_position.json', res)
