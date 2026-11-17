"""E8 (Step 1c, T1-T4): the same tests on a small transformer and a torch GRU.

Synthetic seeds, the same train and dev windows, masks and mask seeds as
Step 1. Models come from torch_models (Adam, same minibatch order and budget
as common.train_gru). The NumPy GRU numbers are read from the cached
starter-style models, so every row is paired on the same windows.

Measures per arm (mean over 5 inits inside a seed):
  gap         block-minus-scatter MSE on all dev windows (E3 definition)
  excess      share of that gap above the 24-step window-oracle gap (T1)
  stale       excess over the window oracle on block windows with age0 >= 3
  trust       e4.gru_sensitivity by sensor-0 age bin (T2)
  moving      E2 outage on sensor 0 or the 5 others at offset 0, 8, 16 (H2, H4)
  probes      e6 probes on the final representation (T3)
  held        e6.held_sensor0 curve, k = 0..16 (selected arms)
  attention   transformer only: last-token attention on the last step and on
              the last real sensor-0 step, by age bin
"""
import numpy as np

import common
from common import INITS, SEEDS, get_model, load, make_split, run_parallel, save, summarize, train_set
import oracle
import e2_block_position as e2
import e3_window_oracle as e3
import e4_training_masks as e4
import e6_representation as e6
import e7_distillation as e7
import torch_models as tm

e6.final_h = lambda model, x: model.hidden(x)  # model-agnostic hidden state for the probes
STALE = 3
# name: (arch, train kind, feature mode, target, epochs)
ARMS = {
    'tf_scatter': ('tf', 'scatter', 'zero', 'real', 12),
    'tf_mixed': ('tf', 'mixed', 'zero', 'real', 12),
    'tf_scatter_age': ('tf', 'scatter', 'age', 'real', 12),
    'tf_mixed_teach': ('tf', 'mixed', 'zero', 'teach', 12),
    'tf_scatter_e48': ('tf', 'scatter', 'zero', 'real', 48),
    'tgru_scatter': ('tgru', 'scatter', 'zero', 'real', 12),
    'tgru_mixed': ('tgru', 'mixed', 'zero', 'real', 12),
}
HELD_ARMS = ['tf_scatter', 'tf_mixed', 'tgru_scatter']


def arm_model(d, seed, arm, init, m=None):
  arch, kind, mode, target, epochs = ARMS[arm]

  def build():
    if target == 'teach':
      x, _, t_mean, _, _ = e7.mixed_with_teacher(d, seed, m if m is not None else oracle.recover(seed)[0])
      return tm.train(arch, x, t_mean, seed, init, epochs=epochs)
    tx, ty = train_set(d, kind, mode, seed)
    return tm.train(arch, tx, ty, seed, init, epochs=epochs)
  return tm.cached(arch, f'{seed}_{arm}_{init}', build)


def warm(seed, arm, init):
  arm_model(load(seed), seed, arm, init)


def attention_by_age(models, x, age0):
  out = {}
  for layer in (0, -1):
    att = np.mean([mm.attention(x, layer) for mm in models], 0)
    last_real = att[np.arange(len(x)), np.clip(common.LOOKBACK - 1 - age0, 0, None)]
    out[f'layer{layer % 2}'] = dict(on_last_step=e4.by_bin(att[:, -1], age0),
                                    on_last_real_s0=e4.by_bin(last_real, age0))
  return out


def per_seed(seed):
  d = load(seed)
  std0 = d['std'][0]
  m, _ = oracle.recover(seed)
  rng = np.random.default_rng(seed + 808)
  dev = {md: (make_split(d, 'dev', 'scatter', seed + 20, md), make_split(d, 'dev', 'block', seed + 20, md))
         for md in ['zero', 'age']}
  sS, sB = dev['zero']
  y_raw = d['dev_target'][sS['ends'] + 6]
  ok = sS['ends'] >= common.LOOKBACK
  kS = ((e3.window_kf(sS, m) - y_raw) / std0) ** 2
  kB = ((e3.window_kf(sB, m) - y_raw) / std0) ** 2
  kf_gap = float((kB - kS)[ok].mean() * std0 ** 2)
  age0 = sB['age_end'][:, 0]
  stale = ok & (age0 >= STALE)
  # Probe sets, as in E6: train mixed windows, dev block windows.
  tr = {md: make_split(d, 'train', 'scatter', seed + 10, md) for md in ['zero', 'age']}
  trB = {md: make_split(d, 'train', 'block', seed + 10, md) for md in ['zero', 'age']}
  pick = np.random.default_rng(seed + 77).random(len(tr['zero']['y'])) < .5
  x_mix = {md: np.where(pick[:, None, None], trB[md]['x'], tr[md]['x']) for md in ['zero', 'age']}
  age_tr = np.where(pick, trB['zero']['age_end'][:, 0], tr['zero']['age_end'][:, 0])
  l_tr, l_te = age_tr >= STALE, age0 >= STALE
  ws_tr, ws_te = age_tr >= 1, age0 >= 1
  # E2 moving outage windows and the E5 factorial cells.
  keep = sS['ends'] >= e2.MIN_END
  s2 = {md: {**dev[md][0]} for md in ['zero', 'age']}
  for md in s2:
    for k in ['x', 'y', 'persistence', 'ends', 'age_end']:
      s2[md][k] = s2[md][k][keep]
  fac = {md: {c: make_split(d, 'dev', kinds, seed + 20, md) for c, kinds in
              [('s0_block', ('block',) + ('scatter',) * 5), ('others_block', ('scatter',) + ('block',) * 5)]}
         for md in ['zero', 'age']}

  def evaluate(models, mode):
    xS, xB = dev[mode][0]['x'], dev[mode][1]['x']
    eS = [(mm.predict(xS) - sS['y']) ** 2 for mm in models]
    eB = [(mm.predict(xB) - sB['y']) ** 2 for mm in models]
    gap = [float((b - a)[ok].mean() * std0 ** 2) for a, b in zip(eS, eB)]
    r = dict(
        rmse_scatter=summarize([float(np.sqrt(a[ok].mean()) * std0) for a in eS]),
        rmse_block=summarize([float(np.sqrt(b[ok].mean()) * std0) for b in eB]),
        gap=summarize(gap), excess_share=summarize([(g - kf_gap) / g for g in gap]),
        excess_stale=summarize([float((b - kB)[stale].mean() * std0 ** 2) for b in eB]),
        trust=e4.by_bin(np.mean([e4.gru_sensitivity(mm, xB, age0.astype(int)) for mm in models], 0), age0))
    base = np.mean([(mm.predict(s2[mode]['x']) - s2[mode]['y']) ** 2 for mm in models], 0)
    r['moving'] = {}
    for name, sensors in [('s0', [0]), ('others', [1, 2, 3, 4, 5])]:
      r['moving'][name] = {}
      for off in [0, 8, 16]:
        x, _, _, _, _ = e2.intervene(s2[mode], d, off, sensors, mode)
        e = np.mean([(mm.predict(x) - s2[mode]['y']) ** 2 for mm in models], 0)
        r['moving'][name][str(off)] = float((e - base).mean() * std0 ** 2)
    r['factorial'] = {c: float(np.mean([(mm.predict(fs['x']) - fs['y']) ** 2 for mm in models], 0)[fs['ends'] >= common.LOOKBACK].mean() * std0 ** 2
                               - np.mean(eS, 0)[ok].mean() * std0 ** 2) for c, fs in fac[mode].items()}
    return r

  out = dict(seed=seed, window_oracle_gap=kf_gap, n_stale=int(stale.sum()), arms={})
  numpy_gru = [get_model(d, seed, 'scatter', 'zero', i) for i in INITS]
  out['arms']['gru_scatter'] = evaluate(numpy_gru, 'zero')
  for arm, (arch, kind, mode, target, epochs) in ARMS.items():
    models = [arm_model(d, seed, arm, i, m) for i in INITS]
    r = evaluate(models, mode)
    hm, ht = [mm.hidden(x_mix[mode]) for mm in models], [mm.hidden(dev[mode][1]['x']) for mm in models]
    r['probe'] = [e6.probes(a, l_tr, b, l_te, rng) for a, b in zip(hm, ht)]
    r['probe_within_stale'] = [e6.probes(a[ws_tr], l_tr[ws_tr], b[ws_te], l_te[ws_te], rng) for a, b in zip(hm, ht)]
    if arch == 'tf':
      r['attention'] = attention_by_age(models, dev[mode][1]['x'], age0.astype(int))
    if arm in HELD_ARMS:
      r['held_sensor0'] = e6.held_sensor0(seed, d, m, models, mode)
    out['arms'][arm] = r
  return out


if __name__ == '__main__':
  run_parallel(warm, [(s, a, i) for s in SEEDS for a in ARMS for i in INITS], procs=48)
  res = run_parallel(per_seed, [(s,) for s in SEEDS])
  for r in res:
    print('seed', r['seed'], 'window oracle gap %.4f' % r['window_oracle_gap'], 'stale n', r['n_stale'])
    for arm, a in r['arms'].items():
      pr = np.mean([p['linear'] for p in a.get('probe', [{'linear': np.nan}])])
      pw = np.mean([p['linear'] for p in a.get('probe_within_stale', [{'linear': np.nan}])])
      print('  %-15s rmseS %.4f rmseB %.4f gap %.4f excess %.2f stale %.4f probe %.2f how-old %.2f' % (
          arm, a['rmse_scatter']['mean'], a['rmse_block']['mean'], a['gap']['mean'], a['excess_share']['mean'],
          a['excess_stale']['mean'], pr, pw),
          'trust', [None if b['value'] is None else round(b['value'], 2) for b in a['trust']],
          'moving s0', {k: round(v, 4) for k, v in a['moving']['s0'].items()})
  save('e8_transformer.json', res)
