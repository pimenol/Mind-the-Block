"""E9 (Step 1c, R1, R2, H2, H3): the Step 1 tests on real data.

Beijing PM2.5 (research/real_data.py). Three target stations are the
replication units, each with 5 inits. Masks are starter.impose scatter and
block on top of the natural gaps, mask seeds unit+10 (train) and unit+20
(dev), as for the synthetic seeds. Only windows with an observed target are
scored. Models: the NumPy GRU and the Step 1c transformer, each trained on
scatter or on the 50/50 mix, same budget as Step 1.

There is no true model, so no oracle. The reference is a ridge model fitted
on train mixed windows: for every station, its last observed value times a
one-hot age bin, plus the age-bin one-hots. Its weight on the target
station's last value per age bin is the empirical best linear trust (R2).
"""
import numpy as np

import common
from common import AGE_BINS, INITS, age_bin_index, cached_model, make_split, run_parallel, save, summarize, train_gru, train_set
import e2_block_position as e2
import e4_training_masks as e4
import real_data as rd
import torch_models as tm

STALE = 3
ARMS = {'gru_scatter': ('gru', 'scatter'), 'gru_mixed': ('gru', 'mixed'),
        'tf_scatter': ('tf', 'scatter'), 'tf_mixed': ('tf', 'mixed')}


def model(d, target, arm, init):
  arch, kind = ARMS[arm]
  seed = rd.SEED[target]
  name = f'bj_{target}_{arm}_{init}'

  def build():
    tx, ty = train_set(d, kind, 'zero', seed)
    return train_gru(tx, ty, seed, init) if arch == 'gru' else tm.train('tf', tx, ty, seed, init)
  return cached_model(name, build) if arch == 'gru' else tm.cached('tf', name, build)


def warm(target, arm, init):
  model(rd.load_real(target), target, arm, init)


def ref_features(s, d):
  """Last observed value (normalized) x one-hot age bin, per station."""
  last = (s['filled'][s['ends']] - d['mean']) / d['std']
  nb = len(AGE_BINS)
  cols = []
  for j in range(6):
    oh = np.eye(nb)[age_bin_index(s['age_end'][:, j])]
    cols += [oh * last[:, j:j + 1], oh]
  return np.concatenate(cols + [np.ones((len(last), 1))], 1)


def ridge(a, y, lam=1e-2):
  return np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ y)


def per_target(target):
  d = rd.load_real(target)
  seed = rd.SEED[target]
  std0 = d['std'][0]
  sS = make_split(d, 'dev', 'scatter', seed + 20, 'zero')
  sB = make_split(d, 'dev', 'block', seed + 20, 'zero')
  ok = sS['ends'] >= common.LOOKBACK
  age0 = sB['age_end'][:, 0]
  stale = ok & (age0 >= STALE)
  # Ridge reference, fitted on train mixed windows (same pick as train_set).
  a = make_split(d, 'train', 'scatter', seed + 10, 'zero')
  b = make_split(d, 'train', 'block', seed + 10, 'zero')
  pick = np.random.default_rng(seed + 77).random(len(a['y'])) < .5
  f_tr = np.where(pick[:, None], ref_features(b, d), ref_features(a, d))
  w = ridge(f_tr, a['y'])
  rS = (ref_features(sS, d) @ w - sS['y']) ** 2
  rB = (ref_features(sB, d) @ w - sB['y']) ** 2
  nb = len(AGE_BINS)
  out = dict(target=target, dates=d['dates'], n_windows=int(ok.sum()), n_stale=int(stale.sum()),
             stale_share=dict(scatter=float((sS['age_end'][ok, 0] >= STALE).mean()), block=float((age0[ok] >= STALE).mean())),
             distinct_outages_stale=int(len(np.unique((sB['ends'] - age0)[stale]))),
             persistence=dict(rmse_scatter=common.rmse(sS['persistence'][ok], sS['y'][ok], std0),
                              rmse_block=common.rmse(sB['persistence'][ok], sB['y'][ok], std0)),
             reference=dict(rmse_scatter=float(np.sqrt(rS[ok].mean()) * std0), rmse_block=float(np.sqrt(rB[ok].mean()) * std0),
                            gap=float((rB - rS)[ok].mean() * std0 ** 2),
                            mse_stale_block=float(rB[stale].mean() * std0 ** 2),
                            trust=[dict(bin=lab, value=float(w[k])) for k, lab in enumerate(common.AGE_LABELS)]),
             arms={})
  assert len(w) == 12 * 6 + 1 and nb == 6
  # Moving outage windows (E2 design).
  keep = sS['ends'] >= e2.MIN_END
  s2 = {**sS}
  for k in ['x', 'y', 'persistence', 'ends', 'age_end']:
    s2[k] = s2[k][keep]
  for arm in ARMS:
    models = [model(d, target, arm, i) for i in INITS]
    eS = [(mm.predict(sS['x']) - sS['y']) ** 2 for mm in models]
    eB = [(mm.predict(sB['x']) - sB['y']) ** 2 for mm in models]
    gap_w = np.mean([b_ - a_ for a_, b_ in zip(eS, eB)], 0)
    r = dict(rmse_scatter=summarize([float(np.sqrt(e[ok].mean()) * std0) for e in eS]),
             rmse_block=summarize([float(np.sqrt(e[ok].mean()) * std0) for e in eB]),
             gap=summarize([float((b_ - a_)[ok].mean() * std0 ** 2) for a_, b_ in zip(eS, eB)]),
             stale_gap_share=float(gap_w[stale].sum() / gap_w[ok].sum()),
             gap_by_bin=e4.by_bin(np.where(ok, gap_w * std0 ** 2, np.nan), age0),
             mse_stale_block=float(np.mean(eB, 0)[stale].mean() * std0 ** 2),
             trust=e4.by_bin(np.mean([e4.gru_sensitivity(mm, sB['x'], age0.astype(int)) for mm in models], 0), age0))
    base = np.mean([(mm.predict(s2['x']) - s2['y']) ** 2 for mm in models], 0)
    r['moving_s0'] = {}
    for off in [0, 8, 16]:
      x, _, _, _, _ = e2.intervene(s2, d, off, [0], 'zero')
      e = np.mean([(mm.predict(x) - s2['y']) ** 2 for mm in models], 0)
      r['moving_s0'][str(off)] = float((e - base).mean() * std0 ** 2)
    out['arms'][arm] = r
  return out


if __name__ == '__main__':
  run_parallel(warm, [(t, a, i) for t in rd.TARGETS for a in ARMS for i in INITS], procs=48)
  res = run_parallel(per_target, [(t,) for t in rd.TARGETS])
  for r in res:
    ref = r['reference']
    print(r['target'], 'windows', r['n_windows'], 'stale', r['n_stale'], 'outages', r['distinct_outages_stale'],
          'stale share', {k: round(v, 3) for k, v in r['stale_share'].items()},
          'persistence', {k: round(v, 4) for k, v in r['persistence'].items()})
    print('  reference     rmseS %.4f rmseB %.4f gap %.4f stale mse %.4f' % (
        ref['rmse_scatter'], ref['rmse_block'], ref['gap'], ref['mse_stale_block']),
        'trust', [round(t['value'], 2) for t in ref['trust']])
    for arm, a in r['arms'].items():
      print('  %-12s rmseS %.4f rmseB %.4f gap %.4f stale share %.2f stale mse %.4f' % (
          arm, a['rmse_scatter']['mean'], a['rmse_block']['mean'], a['gap']['mean'], a['stale_gap_share'], a['mse_stale_block']),
          'trust', [None if b['value'] is None else round(b['value'], 2) for b in a['trust']],
          'moving s0', {k: round(v, 4) for k, v in a['moving_s0'].items()})
  save('e9_beijing.json', res)
