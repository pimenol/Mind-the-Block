"""E3b: a window-limited oracle and the effective sample size of stale bins.

The full Kalman oracle knows the true parameters, the seasonal phase and the
whole dev history. This variant restarts from the stationary prior at the
first step of each 24-step window, so it sees only the readings in the
window (it still knows the true parameters and the time index). If its
block-minus-scatter gap is still much smaller than the GRU gap, then the
information that the GRU loses is present inside the window itself.

Also counts distinct sensor-0 outage events behind the stale windows: many
overlapping windows share one outage, so window counts overstate evidence.
"""
import common
from common import INITS, SEEDS, get_model, load, make_split, run_parallel, save
import numpy as np
import oracle
import e2_block_position as e2


def window_kf(s, m, use=None, split='dev', return_var=False):
  n = len(s['raw'])
  means = np.zeros((n, 5))
  covs = np.broadcast_to(np.eye(5) * oracle.Q / (1 - oracle.PHI ** 2), (n, 5, 5)).copy()
  t = np.arange(n) + oracle.OFFSET[split]
  c = oracle.season(t)
  starts = s['ends'] - common.LOOKBACK + 1
  if use is None:
    use = np.ones((len(starts), common.LOOKBACK, 6), bool)
  # starts-1 indexes the stationary prior; it must stay >= 0.
  mean, var = oracle.batched_restart(means, covs, t, c, s['raw'], m, np.maximum(starts, 1), s['ends'], use)
  return (mean, var) if return_var else mean


def per_seed(seed):
  d = load(seed)
  std0 = d['std'][0]
  m, _ = oracle.recover(seed)
  sS = make_split(d, 'dev', 'scatter', seed + 20, 'zero')
  sB = make_split(d, 'dev', 'block', seed + 20, 'zero')
  y = d['dev_target'][sS['ends'] + 6]
  ok = sS['ends'] >= common.LOOKBACK
  eS = ((window_kf(sS, m) - y) / std0) ** 2
  eB = ((window_kf(sB, m) - y) / std0) ** 2
  kS, _ = oracle.kalman_forecast(sS['raw'], m, 'dev', sS['ends'])
  kB, _ = oracle.kalman_forecast(sB['raw'], m, 'dev', sB['ends'])
  gS = np.mean([(get_model(d, seed, 'scatter', 'zero', i).predict(sS['x']) - sS['y']) ** 2 for i in INITS], 0)
  gB = np.mean([(get_model(d, seed, 'scatter', 'zero', i).predict(sB['x']) - sB['y']) ** 2 for i in INITS], 0)
  a0S, a0B = sS['age_end'][:, 0], sB['age_end'][:, 0]
  stale = ok & (a0S == 0) & (a0B >= 5)
  fresh = ok & (a0S == 0) & (a0B == 0)

  def g(eb, es, sel):
    return float((eb - es)[sel].mean() * std0 ** 2)

  out = dict(seed=seed, mse_gap=dict(
      kalman_full=g(((kB - y) / std0) ** 2, ((kS - y) / std0) ** 2, ok),
      kalman_window24=g(eB, eS, ok), gru_scatter=g(gB, gS, ok)),
      stale=dict(kalman_window24=g(eB, eS, stale), gru_scatter=g(gB, gS, stale),
                 kalman_full=g(((kB - y) / std0) ** 2, ((kS - y) / std0) ** 2, stale)),
      fresh=dict(kalman_window24=g(eB, eS, fresh), gru_scatter=g(gB, gS, fresh)),
      rmse_window24=dict(scatter=float(np.sqrt(eS[ok].mean()) * std0), block=float(np.sqrt(eB[ok].mean()) * std0)))
  # Distinct outage events: windows with age0 >= k share the same last
  # observation time of sensor 0 when they belong to one outage.
  last_obs = sB['ends'] - a0B
  out['exposure'] = {
      f'age0>={k}': dict(windows=int((a0B >= k).sum()), distinct_outages=int(len(np.unique(last_obs[a0B >= k]))))
      for k in [3, 5, 9]}
  # E2 offset-0 cells with the window oracle.
  keep = sS['ends'] >= e2.MIN_END
  s2 = make_split(d, 'dev', 'scatter', seed + 20, 'zero')
  for k in ['x', 'y', 'persistence', 'ends', 'age_end']:
    s2[k] = s2[k][keep]
  base = ((window_kf(s2, m) - d['dev_target'][s2['ends'] + 6]) / std0) ** 2
  cells = {}
  for name, sensors in e2.SENSOR_SETS.items():
    use = np.ones((len(s2['ends']), common.LOOKBACK, 6), bool)
    use[:, -e2.BLOCK:, sensors] = False
    e = ((window_kf(s2, m, use) - d['dev_target'][s2['ends'] + 6]) / std0) ** 2
    cells[name] = float((e - base).mean() * std0 ** 2)
  out['e2_offset0_window24'] = cells
  return out


if __name__ == '__main__':
  res = run_parallel(per_seed, [(s,) for s in SEEDS])
  for r in res:
    print(r['seed'], 'gap', {k: round(v, 4) for k, v in r['mse_gap'].items()},
          'stale', {k: round(v, 4) for k, v in r['stale'].items()},
          'fresh', {k: round(v, 4) for k, v in r['fresh'].items()})
    print('   rmse24', r['rmse_window24'], 'e2', {k: round(v, 4) for k, v in r['e2_offset0_window24'].items()})
    print('   exposure', r['exposure'])
  save('e3_window_oracle.json', res)
