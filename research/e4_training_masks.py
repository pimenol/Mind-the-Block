"""E4 (H3): is the excess a learned shortcut that training coverage fixes?

Train the same GRU on scatter, block, or mixed (50/50 per window) masks at
the same budget (12 epochs x 1600 windows, same init and minibatch order),
with the age channel on ('age') or zeroed ('zero'). Evaluate on dev scatter
and block. Also measure the sensitivity of the forecast to the stale
sensor-0 value, d forecast / d (held value), by sensor-0 age at the origin,
and compare with the Kalman filter's sensitivity to the same reading.
A model that treats the stale fill as current keeps sensitivity near its
fresh value; the Bayes-optimal sensitivity decays with age.
"""
import common
from common import (AGE_BINS, AGE_LABELS, INITS, SEEDS, get_model, load,
                    make_split, rmse, run_parallel, save, summarize)
import numpy as np
import oracle
import e2_block_position as e2

KINDS = ['scatter', 'block', 'mixed']
MODES = ['zero', 'age']
DELTA = .1


def warm(seed, kind, mode, init):
  get_model(load(seed), seed, kind, mode, init)


def gru_sensitivity(model, x, age0):
  xp = x.copy()
  rows = np.arange(x.shape[1])[None] >= (x.shape[1] - 1 - age0)[:, None]
  xp[:, :, 0] += DELTA * rows
  return (model.predict(xp) - model.predict(x)) / DELTA


def kalman_sensitivity(seed, d, s, m):
  means, covs, t, c = oracle.kalman_states(s['raw'], m, 'dev')
  age0 = s['age_end'][:, 0].astype(int)
  starts = s['ends'] - age0
  ok = starts >= 1
  steps = int(age0[ok].max()) + 1
  use = np.ones((ok.sum(), steps, 6), bool)
  bump = np.zeros((ok.sum(), 6))
  bump[:, 0] = 1.
  base, _ = oracle.batched_restart(means, covs, t, c, s['raw'], m, starts[ok], s['ends'][ok], use)
  pert, _ = oracle.batched_restart(means, covs, t, c, s['raw'], m, starts[ok], s['ends'][ok], use, bump=bump)
  out = np.full(len(age0), np.nan)
  out[ok] = pert - base
  return out


def by_bin(values, age0):
  rows = []
  for (lo, hi), lab in zip(AGE_BINS, AGE_LABELS):
    sel = (age0 >= lo) & (age0 <= hi) & np.isfinite(values)
    rows.append(dict(bin=lab, n=int(sel.sum()), value=float(values[sel].mean()) if sel.any() else None))
  return rows


def per_seed(seed):
  d = load(seed)
  std0 = d['std'][0]
  m, _ = oracle.recover(seed)
  out = dict(seed=seed, arms={})
  sB0 = make_split(d, 'dev', 'block', seed + 20, 'zero')
  age0B = sB0['age_end'][:, 0]
  age0S = make_split(d, 'dev', 'scatter', seed + 20, 'zero')['age_end'][:, 0]
  out['kalman_sensitivity'] = by_bin(kalman_sensitivity(seed, d, sB0, m), age0B)
  for mode in MODES:
    sS = make_split(d, 'dev', 'scatter', seed + 20, mode)
    sB = make_split(d, 'dev', 'block', seed + 20, mode)
    for kind in KINDS:
      models = [get_model(d, seed, kind, mode, i) for i in INITS]
      rS = [rmse(mm.predict(sS['x']), sS['y'], std0) for mm in models]
      rB = [rmse(mm.predict(sB['x']), sB['y'], std0) for mm in models]
      eS = np.mean([(mm.predict(sS['x']) - sS['y']) ** 2 for mm in models], 0)
      eB = np.mean([(mm.predict(sB['x']) - sB['y']) ** 2 for mm in models], 0)
      fresh = age0S == 0
      paired = []
      for (lo, hi), lab in zip(AGE_BINS, AGE_LABELS):
        sel = fresh & (age0B >= lo) & (age0B <= hi)
        paired.append(dict(bin=lab, n=int(sel.sum()),
                           delta=float((eB - eS)[sel].mean() * std0 ** 2) if sel.any() else None))
      sens = np.mean([gru_sensitivity(mm, sB['x'], age0B) for mm in models], 0)
      out['arms'][f'{kind}/{mode}'] = dict(
          train=kind, mode=mode, rmse_scatter=summarize(rS), rmse_block=summarize(rB),
          gap=summarize(np.array(rB) - np.array(rS)), per_init_gap=(np.array(rB) - np.array(rS)).tolist(),
          paired_delta_by_block_age=paired, sensitivity_s0=by_bin(sens, age0B))
    cells = e2.evaluate(seed, train_kinds=tuple(KINDS), mode=mode)
    out[f'e2_cells/{mode}'] = [c for c in cells['cells'] if c['offset'] == 0]
  out['persistence'] = dict(scatter=rmse(sS['persistence'], sS['y'], std0),
                            block=rmse(sB['persistence'], sB['y'], std0))
  return out


if __name__ == '__main__':
  run_parallel(warm, [(s, k, md, i) for s in SEEDS for k in KINDS for md in MODES for i in INITS])
  res = run_parallel(per_seed, [(s,) for s in SEEDS])
  for r in res:
    print('seed', r['seed'], 'persistence', {k: round(v, 4) for k, v in r['persistence'].items()})
    print('   kalman sens', [(x['bin'], None if x['value'] is None else round(x['value'], 2)) for x in r['kalman_sensitivity']])
    for name, a in r['arms'].items():
      print('  %-13s scatter %.4f block %.4f gap %+.4f [%+.4f,%+.4f]' % (
          name, a['rmse_scatter']['mean'], a['rmse_block']['mean'], a['gap']['mean'], a['gap']['min'], a['gap']['max']),
          'sens', [None if x['value'] is None else round(x['value'], 2) for x in a['sensitivity_s0']])
    for mode in MODES:
      for c in r[f'e2_cells/{mode}']:
        print('   e2', mode, c['sensors'], {k: round(v, 4) for k, v in c['delta_mse'].items()})
  save('e4_training_masks.json', res)
