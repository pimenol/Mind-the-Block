"""E4b: is the missing age-gating a budget or capacity limit?

Same GRU code, age channel on, trained on scatter or block masks, with
(epochs, hidden) in {12, 48} x {8, 32}. If the stale-trust pattern (forecast
sensitivity to the stale sensor-0 value does not decay with age) goes away
with more steps or units, the starter's failure is an optimization/capacity
limit. If it stays, the plain-GRU input encoding is the bottleneck.
"""
import common
from common import AGE_BINS, AGE_LABELS, SEEDS, get_model, load, make_split, rmse, run_parallel, save, summarize
import numpy as np
from e4_training_masks import gru_sensitivity, by_bin

KINDS = ['scatter', 'block']
SETTINGS = [(12, 8), (48, 8), (12, 32), (48, 32)]
INITS = [0, 1, 2]


def warm(seed, kind, epochs, hidden, init):
  get_model(load(seed), seed, kind, 'age', init, epochs, hidden)


def per_seed(seed):
  d = load(seed)
  std0 = d['std'][0]
  sS = make_split(d, 'dev', 'scatter', seed + 20, 'age')
  sB = make_split(d, 'dev', 'block', seed + 20, 'age')
  a0B, a0S = sB['age_end'][:, 0], sS['age_end'][:, 0]
  stale = (a0S == 0) & (a0B >= 5)
  out = dict(seed=seed, n_stale=int(stale.sum()), arms=[])
  for kind in KINDS:
    for epochs, hidden in SETTINGS:
      ms = [get_model(d, seed, kind, 'age', i, epochs, hidden) for i in INITS]
      pS = [m.predict(sS['x']) for m in ms]
      pB = [m.predict(sB['x']) for m in ms]
      rS = [rmse(p, sS['y'], std0) for p in pS]
      rB = [rmse(p, sB['y'], std0) for p in pB]
      eS = np.mean([(p - sS['y']) ** 2 for p in pS], 0)
      eB = np.mean([(p - sB['y']) ** 2 for p in pB], 0)
      sens = np.mean([gru_sensitivity(m, sB['x'], a0B) for m in ms], 0)
      out['arms'].append(dict(
          train=kind, epochs=epochs, hidden=hidden, rmse_scatter=summarize(rS), rmse_block=summarize(rB),
          gap=summarize(np.array(rB) - np.array(rS)),
          stale_paired_delta=float((eB - eS)[stale].mean() * std0 ** 2),
          sensitivity_s0=by_bin(sens, a0B)))
  return out


if __name__ == '__main__':
  run_parallel(warm, [(s, k, e, h, i) for s in SEEDS for k in KINDS for e, h in SETTINGS for i in INITS], procs=48)
  res = run_parallel(per_seed, [(s,) for s in SEEDS])
  for r in res:
    print('seed', r['seed'], 'stale windows', r['n_stale'])
    for a in r['arms']:
      print('  %-7s e%-2d h%-2d scatter %.4f block %.4f gap %+.4f stale-delta %.3f sens' % (
          a['train'], a['epochs'], a['hidden'], a['rmse_scatter']['mean'], a['rmse_block']['mean'], a['gap']['mean'],
          a['stale_paired_delta']), [None if x['value'] is None else round(x['value'], 2) for x in a['sensitivity_s0']])
  save('e4b_budget_capacity.json', res)
