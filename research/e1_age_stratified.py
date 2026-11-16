"""E1 (H1) + E3 (H3/H4): error by sensor-0 age at the forecast origin.

For each seed and mask: per-bin MSE of the scatter-trained GRU (mean over 5
inits), persistence, Kalman (all sensors) and Kalman (sensor 0 only). The
MSE gap block - scatter is split into a composition part (block has more
stale origins) and a within-age part (same age, still worse):
  gap = sum_b (wB_b - wS_b) * mseS_b  +  sum_b wB_b * (mseB_b - mseS_b)
Bins for the split are 0, 1, 2, 3+ so every bin has scatter windows.
"""
import common
from common import (AGE_LABELS, INITS, SEEDS, get_model, load, make_split,
                    per_bin, run_parallel, save)
import numpy as np
import oracle

SPLIT_BINS = [(0, 0), (1, 1), (2, 2), (3, 10**6)]


def sq_errors(seed, d, kind, m):
  s = make_split(d, 'dev', kind, seed + 20, 'zero')
  std0 = d['std'][0]
  y_raw = d['dev_target'][s['ends'] + 6]
  gru = np.mean([(get_model(d, seed, 'scatter', 'zero', i).predict(s['x']) - s['y']) ** 2
                 for i in INITS], 0)
  kf, kvar = oracle.kalman_forecast(s['raw'], m, 'dev', s['ends'])
  kf0, _ = oracle.kalman_forecast(s['raw'], m, 'dev', s['ends'], sensors=[0])
  return s['age_end'][:, 0], dict(
      gru=gru, persistence=(s['persistence'] - s['y']) ** 2,
      kalman=((kf - y_raw) / std0) ** 2, kalman_s0=((kf0 - y_raw) / std0) ** 2,
      kalman_expected=kvar / std0 ** 2)


def decompose(aS, eS, aB, eB, std0):
  comp = within = 0.
  rows = []
  for lo, hi in SPLIT_BINS:
    sS, sB = (aS >= lo) & (aS <= hi), (aB >= lo) & (aB <= hi)
    wS, wB = sS.mean(), sB.mean()
    mS, mB = eS[sS].mean(), eB[sB].mean()
    comp += (wB - wS) * mS
    within += wB * (mB - mS)
    rows.append(dict(bin=f'{lo}+' if hi > 100 else str(lo), wS=float(wS), wB=float(wB),
                     nS=int(sS.sum()), nB=int(sB.sum()),
                     mseS=float(mS * std0 ** 2), mseB=float(mB * std0 ** 2)))
  total = eB.mean() - eS.mean()
  return dict(total_mse_gap=float(total * std0 ** 2), composition=float(comp * std0 ** 2),
              within_age=float(within * std0 ** 2), bins=rows)


def per_seed(seed):
  d = load(seed)
  std0 = d['std'][0]
  m, _ = oracle.recover(seed)
  aS, eS = sq_errors(seed, d, 'scatter', m)
  aB, eB = sq_errors(seed, d, 'block', m)
  out = dict(seed=seed, per_bin={}, decomposition={})
  for name in eS:
    out['per_bin'][name] = dict(scatter=per_bin(eS[name], aS, std0),
                                block=per_bin(eB[name], aB, std0))
    if name != 'kalman_expected':
      out['decomposition'][name] = decompose(aS, eS[name], aB, eB[name], std0)
  # Headline H1 test: fresh origin only (age0 = 0).
  out['fresh_origin_gap'] = {
      n: float((eB[n][aB == 0].mean() - eS[n][aS == 0].mean()) * std0 ** 2) for n in eS}
  out['fresh_origin_n'] = dict(scatter=int((aS == 0).sum()), block=int((aB == 0).sum()))
  # Paired design: the same window under both masks, so window difficulty
  # cancels. delta_w = errB(w) - errS(w), grouped by block age0 among
  # windows whose scatter origin is fresh.
  sB = make_split(d, 'dev', 'block', seed + 20, 'zero')
  sS = make_split(d, 'dev', 'scatter', seed + 20, 'zero')
  assert np.array_equal(sB['ends'], sS['ends'])
  stale_in_window = (sB['x'][:, :, 6] == 0).sum(1)  # sensor-0 missing steps
  base = aS == 0
  paired = {}
  for n in eS:
    delta = eB[n] - eS[n]
    rows = []
    for (lo, hi), lab in zip(common.AGE_BINS, AGE_LABELS):
      sel = base & (aB >= lo) & (aB <= hi)
      rows.append(dict(bin=lab, n=int(sel.sum()),
                       delta=float(delta[sel].mean() * std0 ** 2) if sel.any() else None))
    # H2 inside real masks: both origins fresh, split by how many sensor-0
    # steps are missing inside the block window.
    both = base & (aB == 0)
    interior = []
    for lo, hi in [(0, 2), (3, 5), (6, 8), (9, 24)]:
      sel = both & (stale_in_window >= lo) & (stale_in_window <= hi)
      interior.append(dict(missing_s0_steps=f'{lo}-{hi}', n=int(sel.sum()),
                           delta=float(delta[sel].mean() * std0 ** 2) if sel.any() else None))
    paired[n] = dict(by_block_age=rows, both_fresh_by_interior_missing=interior,
                     both_fresh=float(delta[both].mean() * std0 ** 2), n_both_fresh=int(both.sum()),
                     total=float(delta.mean() * std0 ** 2))
  out['paired'] = paired
  return out


if __name__ == '__main__':
  res = run_parallel(per_seed, [(s,) for s in SEEDS])
  for r in res:
    for n in ['gru', 'persistence', 'kalman', 'kalman_s0']:
      p = r['paired'][n]
      print(r['seed'], '%-12s paired total %.4f both-fresh %.4f (n=%d)' % (n, p['total'], p['both_fresh'], p['n_both_fresh']),
            [(row['bin'], row['n'], None if row['delta'] is None else round(row['delta'], 3)) for row in p['by_block_age']],
            [(row['missing_s0_steps'], row['n'], None if row['delta'] is None else round(row['delta'], 3)) for row in p['both_fresh_by_interior_missing']])
  for r in res:
    print('seed', r['seed'], 'fresh-origin MSE gap', {k: round(v, 5) for k, v in r['fresh_origin_gap'].items()})
    for n, dec in r['decomposition'].items():
      print('  %-12s total %.5f comp %.5f within %.5f' % (n, dec['total_mse_gap'], dec['composition'], dec['within_age']))
    for n in ['gru', 'persistence', 'kalman', 'kalman_s0']:
      print('  %-12s block per-bin MSE' % n, [(row['bin'], row['n'], None if row['mse'] is None else round(row['mse'], 3))
                                              for row in r['per_bin'][n]['block']])
  save('e1_age_stratified.json', res)
