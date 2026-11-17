"""E7 (H6): can the same GRU learn age discounting from a clean target?

Train the 8-unit zero-age GRU on mixed-mask train windows (same inputs,
init, budget and minibatch order as the cached mixed/zero model), but swap
the target:

  real        the true future value (the cached mixed/zero model)
  teach_a     the 24-step window-oracle forecast plus fixed Gaussian noise
              with scale a x the oracle standard deviation, a in TEACH
  over5       real targets, stale-origin windows (age0 >= 3) repeated 5x in
              the pool that the epochs sample from; same 1600 per epoch
  *_e48       the same at 48 epochs

The window oracle sees only the readings inside the window (plus the true
parameters and time index), so no future value enters an input. It is a
diagnostic teacher, not a method. a = 1 gives targets with the same spread
as the real ones: it must score like the real arm.

The excess is the dev block MSE minus the window-oracle MSE on the same
windows, for stale origins (age0 >= 3) and fresh origins (age0 = 0).
"""
import common
from common import (INITS, SEEDS, cached_model, get_model, load, make_split,
                    rmse, run_parallel, save, summarize, train_gru)
import numpy as np
import oracle
import e3_window_oracle as e3
import e4_training_masks as e4

TEACH = [0., .25, .5, 1.]
STALE = 3
ARMS = (['real'] + [f'teach{a}' for a in TEACH] + ['over5', 'real_e48', 'teach0.0_e48'])


def mixed_with_teacher(d, seed, m):
  """Mixed train windows (same pick as common.train_set) with teacher."""
  std0, mean0 = d['std'][0], d['mean'][0]
  a = make_split(d, 'train', 'scatter', seed + 10, 'zero')
  b = make_split(d, 'train', 'block', seed + 10, 'zero')
  pick = np.random.default_rng(seed + 77).random(len(a['y'])) < .5
  x = np.where(pick[:, None, None], b['x'], a['x'])
  ma, va = e3.window_kf(a, m, split='train', return_var=True)
  mb, vb = e3.window_kf(b, m, split='train', return_var=True)
  t_mean = (np.where(pick, mb, ma) - mean0) / std0
  t_sd = np.sqrt(np.where(pick, vb, va)) / std0
  age0 = np.where(pick, b['age_end'][:, 0], a['age_end'][:, 0])
  return x, a['y'], t_mean, t_sd, age0


def arm_model(d, seed, arm, init, data):
  x, y, t_mean, t_sd, age0 = data
  if arm == 'real':
    return get_model(d, seed, 'mixed', 'zero', init)
  epochs = 48 if arm.endswith('_e48') else 12
  base = arm.replace('_e48', '')

  def build():
    if base == 'real':
      return train_gru(x, y, seed, init, epochs=epochs)
    if base == 'over5':
      idx = np.concatenate([np.arange(len(y))] + [np.flatnonzero(age0 >= STALE)] * 4)
      return train_gru(x[idx], y[idx], seed, init, epochs=epochs)
    a = float(base.replace('teach', ''))
    z = np.random.default_rng(seed + 707).standard_normal(len(y))  # fixed per window
    return train_gru(x, t_mean + a * t_sd * z, seed, init, epochs=epochs)
  return cached_model(f'{seed}_mixed_zero_{init}_{arm}', build)


def warm(seed, arm, init):
  d = load(seed)
  m, _ = oracle.recover(seed)
  arm_model(d, seed, arm, init, mixed_with_teacher(d, seed, m))


def per_seed(seed):
  d = load(seed)
  std0 = d['std'][0]
  m, _ = oracle.recover(seed)
  data = mixed_with_teacher(d, seed, m)
  x, y, t_mean, t_sd, age0_tr = data
  sS = make_split(d, 'dev', 'scatter', seed + 20, 'zero')
  sB = make_split(d, 'dev', 'block', seed + 20, 'zero')
  y_raw = d['dev_target'][sB['ends'] + 6]
  e_kf = ((e3.window_kf(sB, m) - y_raw) / std0) ** 2
  age0 = sB['age_end'][:, 0]
  ok = sB['ends'] >= common.LOOKBACK
  stale, fresh = ok & (age0 >= STALE), ok & (age0 == 0)
  out = dict(seed=seed, n_stale=int(stale.sum()), n_fresh=int(fresh.sum()),
             n_train_stale=int((age0_tr >= STALE).sum()),
             teacher_train_mse=float(((t_mean - y) ** 2).mean() * std0 ** 2),
             teacher_mean_var=float((t_sd ** 2).mean() * std0 ** 2),
             # Diagnostic for the a = 1 control: real residuals of neighbouring
             # (overlapping) windows are correlated, the synthetic noise is not.
             residual_autocorr={f'lag{k}': float(np.corrcoef((y - t_mean)[:-k], (y - t_mean)[k:])[0, 1])
                                for k in (1, 3, 6)},
             oracle=dict(stale=float(e_kf[stale].mean() * std0 ** 2), fresh=float(e_kf[fresh].mean() * std0 ** 2)),
             arms={})
  for arm in ARMS:
    models = [arm_model(d, seed, arm, i, data) for i in INITS]
    eB = [(mm.predict(sB['x']) - sB['y']) ** 2 for mm in models]
    rS = [rmse(mm.predict(sS['x']), sS['y'], std0) for mm in models]
    rB = [rmse(mm.predict(sB['x']), sB['y'], std0) for mm in models]
    ex_st = [float((e - e_kf)[stale].mean() * std0 ** 2) for e in eB]
    ex_fr = [float((e - e_kf)[fresh].mean() * std0 ** 2) for e in eB]
    sens = np.mean([e4.gru_sensitivity(mm, sB['x'], age0.astype(int)) for mm in models], 0)
    out['arms'][arm] = dict(
        rmse_scatter=summarize(rS), rmse_block=summarize(rB),
        gap=summarize(np.array(rB) - np.array(rS)),
        excess_stale=summarize(ex_st), excess_fresh=summarize(ex_fr),
        per_init_excess_stale=ex_st,
        mse_stale=float(np.mean(eB, 0)[stale].mean() * std0 ** 2),
        sensitivity_s0=e4.by_bin(sens, age0))
  base = out['arms']['real']['excess_stale']['mean']
  for arm in ARMS:
    out['arms'][arm]['excess_vs_real'] = out['arms'][arm]['excess_stale']['mean'] / base
  return out


if __name__ == '__main__':
  run_parallel(warm, [(s, a, i) for s in SEEDS for a in ARMS if a != 'real' for i in INITS], procs=48)
  res = run_parallel(per_seed, [(s,) for s in SEEDS])
  for r in res:
    print('seed', r['seed'], 'stale n', r['n_stale'], 'train stale', r['n_train_stale'],
          'oracle', {k: round(v, 4) for k, v in r['oracle'].items()},
          'teacher train mse %.4f mean var %.4f' % (r['teacher_train_mse'], r['teacher_mean_var']),
          'resid autocorr', {k: round(v, 2) for k, v in r['residual_autocorr'].items()})
    for arm, a in r['arms'].items():
      print('  %-13s rmseS %.4f rmseB %.4f excess stale %.4f [%.4f,%.4f] fresh %.4f ratio %.2f' % (
          arm, a['rmse_scatter']['mean'], a['rmse_block']['mean'], a['excess_stale']['mean'],
          a['excess_stale']['min'], a['excess_stale']['max'], a['excess_fresh']['mean'], a['excess_vs_real']),
          'sens', [None if b['value'] is None else round(b['value'], 2) for b in a['sensitivity_s0']])
  save('e7_distillation.json', res)
