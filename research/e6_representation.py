"""E6 (H5): is sensor-0 staleness represented in the GRU hidden state?

Three tools, all on frozen, already trained GRUs (zero-age arm, plus the
age-input arm, where age is an input):

1. Probes. Decode staleness (sensor-0 age >= 3 at the origin) from the final
   hidden state h_T with a least-squares linear classifier and a
   class-balanced kNN. Fit on train mixed-mask windows, score balanced
   accuracy on dev block windows. A probe fit on shuffled train labels is
   the control (selectivity = probe - control). The same probes on the raw
   input window give the input ceiling: the window always holds the age.
2. Readout refit. Ridge from h_T (linear, and linear + quadratic terms) to
   the target, fit on train windows, with the GRU frozen. If a new readout
   removes the stale-origin excess over the window oracle, the information
   is in h and the trained readout is the bottleneck.
3. Dynamics. (i) Hold all 18 inputs at the last row after the origin and run
   the GRU on: how fast does h converge, and what is the spectral radius of
   the step Jacobian dh_t/dh_(t-1) at the reached state. (ii) Hold sensor 0
   for k = 0..16 steps before the origin: the forecast sensitivity to the
   held value, next to the window-oracle sensitivity, and how well k can be
   decoded from h.
"""
import common
from common import (INITS, SEEDS, get_model, load, make_split, run_parallel,
                    save)
import numpy as np
import oracle
import e2_block_position as e2
import e3_window_oracle as e3
from starter import sigmoid

STALE = 3
HOLDS = list(range(17))
KNN = 25
EXTRA = 16  # steps of held input after the origin, for the fixed point
ARMS = [('scatter', 'zero'), ('mixed', 'zero'), ('scatter', 'age')]


def gru_step(p, v, h):
  z = sigmoid(v @ p['wz'] + h @ p['uz'] + p['bz'])
  r = sigmoid(v @ p['wr'] + h @ p['ur'] + p['br'])
  n = np.tanh(v @ p['wn'] + (r * h) @ p['un'] + p['bn'])
  return z * h + (1 - z) * n


def final_h(model, x):
  return model.predict(x, cache=True)[1]


def standardize(a, b):
  mu, sd = a.mean(0), a.std(0) + 1e-8
  return (a - mu) / sd, (b - mu) / sd


def ridge(a, y, lam=1e-3):
  a1 = np.c_[a, np.ones(len(a))]
  reg = lam * len(a) * np.eye(a1.shape[1])
  reg[-1, -1] = 0
  return np.linalg.solve(a1.T @ a1 + reg, a1.T @ y)


def apply(w, a):
  return np.c_[a, np.ones(len(a))] @ w


def balanced_acc(pred, lab):
  return float(0.5 * (pred[lab].mean() + (~pred[~lab]).mean()))


def linear_probe(a_tr, l_tr, a_te, l_te):
  """Least-squares classifier with class weights, threshold 0."""
  a_tr, a_te = standardize(a_tr, a_te)
  w8 = np.where(l_tr, .5 / l_tr.mean(), .5 / (~l_tr).mean())
  sw = np.sqrt(w8)[:, None]
  w = ridge(a_tr * sw, np.where(l_tr, 1., -1.) * sw[:, 0], lam=1e-2)
  return balanced_acc(apply(w, a_te) > 0, l_te)


def knn_probe(a_tr, l_tr, a_te, l_te, k=KNN):
  """kNN that compares the class share among neighbours with the prior."""
  a_tr, a_te = standardize(a_tr, a_te)
  d2 = (a_te ** 2).sum(1)[:, None] - 2 * a_te @ a_tr.T + (a_tr ** 2).sum(1)[None]
  nn = np.argpartition(d2, k, axis=1)[:, :k]
  share = l_tr[nn].mean(1)
  return balanced_acc(share > l_tr.mean(), l_te)


def probes(a_tr, l_tr, a_te, l_te, rng):
  shuf = rng.permutation(l_tr)
  return dict(linear=linear_probe(a_tr, l_tr, a_te, l_te),
              knn=knn_probe(a_tr, l_tr, a_te, l_te),
              linear_control=linear_probe(a_tr, shuf, a_te, l_te),
              knn_control=knn_probe(a_tr, shuf, a_te, l_te))


def r2(pred, y):
  return float(1 - ((pred - y) ** 2).mean() / y.var())


def quad(h):
  iu = np.triu_indices(h.shape[1])
  return np.c_[h, (h[:, :, None] * h[:, None, :])[:, iu[0], iu[1]]]


def jacobian_radius(p, v, h, eps=1e-5):
  """Spectral radius of dh_t/dh_(t-1) at (v, h), finite differences."""
  out = []
  base = gru_step(p, v, h)
  jac = np.empty((len(h), h.shape[1], h.shape[1]))
  for j in range(h.shape[1]):
    hp = h.copy()
    hp[:, j] += eps
    jac[:, :, j] = (gru_step(p, v, hp) - base) / eps
  for jm in jac:
    out.append(np.abs(np.linalg.eigvals(jm)).max())
  return np.array(out)


def fixed_point(model, x):
  """Hold the last input row for EXTRA steps after the origin."""
  p = model.p
  h = final_h(model, x)
  v = x[:, -1]
  steps, preds = [], [h @ p['o'] + p['c']]
  for _ in range(EXTRA):
    hn = gru_step(p, v, h)
    steps.append(float(np.linalg.norm(hn - h, axis=1).mean()))
    h = hn
    preds.append(h @ p['o'] + p['c'])
  rad = jacobian_radius(p, v, h)
  moved = np.abs(preds[-1] - preds[0]).mean()
  return dict(step_norm=steps, radius_median=float(np.median(rad)),
              radius_q90=float(np.quantile(rad, .9)), forecast_shift=float(moved),
              final_step_norm=steps[-1])


def held_sensor0(seed, d, m, models, mode):
  """Hold sensor 0 for k steps before the origin on dev scatter windows."""
  s = make_split(d, 'dev', 'scatter', seed + 20, mode)
  keep = (s['ends'] >= e2.MIN_END)
  for key in ['x', 'y', 'persistence', 'ends', 'age_end']:
    s[key] = s[key][keep]
  ends = s['ends']
  means, covs, t, c = oracle.kalman_states(s['raw'], m, 'dev')
  rows, hs, ks, halves = [], [], [], []
  for k in HOLDS:
    x, _, use, start, age0 = e2.intervene(s, d, 0, [0], mode, block=k)
    # The held value is the reading at ends - k. Keep windows where it is real.
    ok = s['valid'][ends - k, 0]
    lag = age0.astype(int)
    gs = np.mean([e4_sens(mm, x, lag) for mm in models], 0)
    st = ends - lag
    use2 = np.ones((len(ends), int(lag.max()) + 1, 6), bool)
    use2[:, 1:, 0] = False
    bump = np.zeros((len(ends), 6))
    bump[:, 0] = 1.
    base, _ = oracle.batched_restart(means, covs, t, c, s['raw'], m, st, ends, use2)
    pert, _ = oracle.batched_restart(means, covs, t, c, s['raw'], m, st, ends, use2, bump=bump)
    ks_sens = pert - base  # same sensor in and out, so raw ratio = normalized ratio
    h = final_h(models[0], x)
    hs.append(h[ok])
    ks.append(np.full(ok.sum(), k))
    halves.append(np.arange(ok.sum()) < ok.sum() // 2)
    rows.append(dict(k=k, n=int(ok.sum()), gru_sensitivity=float(gs[ok].mean()),
                     kalman_sensitivity=float(ks_sens[ok].mean())))
  # Decode k from h (init 0) with a split by window halves.
  hs, ks, half = np.concatenate(hs), np.concatenate(ks), np.concatenate(halves)
  a_tr, a_te = standardize(hs[half], hs[~half])
  w = ridge(a_tr, ks[half])
  return dict(curve=rows, k_decode_r2=r2(apply(w, a_te), ks[~half]))


def e4_sens(model, x, age0, delta=.1):
  xp = x.copy()
  rows = np.arange(x.shape[1])[None] >= (x.shape[1] - 1 - age0)[:, None]
  xp[:, :, 0] += delta * rows
  return (model.predict(xp) - model.predict(x)) / delta


def per_seed(seed):
  d = load(seed)
  std0 = d['std'][0]
  m, _ = oracle.recover(seed)
  rng = np.random.default_rng(seed + 606)
  out = dict(seed=seed, arms={})
  tr = {md: make_split(d, 'train', 'scatter', seed + 10, md) for md in ['zero', 'age']}
  trB = {md: make_split(d, 'train', 'block', seed + 10, md) for md in ['zero', 'age']}
  pick = np.random.default_rng(seed + 77).random(len(tr['zero']['y'])) < .5  # same as train_set('mixed')
  devB = {md: make_split(d, 'dev', 'block', seed + 20, md) for md in ['zero', 'age']}
  y_tr = tr['zero']['y']
  age_tr = np.where(pick, trB['zero']['age_end'][:, 0], tr['zero']['age_end'][:, 0])
  age_te = devB['zero']['age_end'][:, 0]
  l_tr, l_te = age_tr >= STALE, age_te >= STALE
  # Window oracle on dev block (reference for the excess).
  y_dev_raw = d['dev_target'][devB['zero']['ends'] + 6]
  kf = e3.window_kf(devB['zero'], m)
  e_kf = ((kf - y_dev_raw) / std0) ** 2
  stale_te = l_te & (devB['zero']['ends'] >= common.LOOKBACK)
  fresh_te = (age_te == 0) & (devB['zero']['ends'] >= common.LOOKBACK)
  out['n_dev_stale'] = int(stale_te.sum())
  out['n_train_stale'] = int(l_tr.sum())
  out['window_oracle_mse'] = dict(stale=float(e_kf[stale_te].mean() * std0 ** 2),
                                  fresh=float(e_kf[fresh_te].mean() * std0 ** 2))
  # Input ceiling: probes on the flattened raw input window (zero arm features).
  x_tr_mixed = {md: np.where(pick[:, None, None], trB[md]['x'], tr[md]['x']) for md in ['zero', 'age']}
  flat_tr = x_tr_mixed['zero'][:, :, 6:12].reshape(len(y_tr), -1)
  flat_te = devB['zero']['x'][:, :, 6:12].reshape(len(age_te), -1)
  out['input_ceiling'] = probes(flat_tr, l_tr, flat_te, l_te, rng)
  # Post hoc (added after the first run): among stale windows only (age >= 1,
  # so the last-step mask is 0 for all), decode age >= 3 vs 1-2. This asks
  # for "how old", not "stale or not".
  ws_tr, ws_te = age_tr >= 1, age_te >= 1
  out['input_ceiling_within_stale'] = probes(flat_tr[ws_tr], l_tr[ws_tr], flat_te[ws_te], l_te[ws_te], rng)
  for kind, mode in ARMS:
    models = [get_model(d, seed, kind, mode, i) for i in INITS]
    xs = tr[mode]['x'] if kind == 'scatter' else x_tr_mixed[mode]
    res = dict(probe=[], refit=[], fixed_point=[])
    for mm in models:
      h_mixed = final_h(mm, x_tr_mixed[mode])
      h_te = final_h(mm, devB[mode]['x'])
      res['probe'].append(probes(h_mixed, l_tr, h_te, l_te, rng))
      res.setdefault('probe_within_stale', []).append(
          probes(h_mixed[ws_tr], l_tr[ws_tr], h_te[ws_te], l_te[ws_te], rng))
      # Age regression among stale windows (age >= 1): calibration by bin.
      st_tr, st_te = age_tr >= 1, age_te >= 1
      a_tr, a_te = standardize(h_mixed[st_tr], h_te[st_te])
      w = ridge(a_tr, np.log1p(age_tr[st_tr]))
      pred_age = np.expm1(apply(w, a_te))
      calib = []
      for lo, hi in common.AGE_BINS[1:]:
        sel = (age_te[st_te] >= lo) & (age_te[st_te] <= hi)
        calib.append(float(pred_age[sel].mean()) if sel.any() else None)
      res.setdefault('age_r2_stale', []).append(r2(apply(w, a_te), np.log1p(age_te[st_te])))
      res.setdefault('age_calibration', []).append(calib)
      # Readout refit cells. Fit sets: own training windows, and mixed windows.
      e_gru = (mm.predict(devB[mode]['x']) - devB[mode]['y']) ** 2
      cell = dict(gru_stale=float(e_gru[stale_te].mean() * std0 ** 2),
                  gru_fresh=float(e_gru[fresh_te].mean() * std0 ** 2))
      for fit_name, xfit in [('own', xs), ('mixed', x_tr_mixed[mode])]:
        hf = final_h(mm, xfit)
        for feat_name, fn in [('linear', lambda a: a), ('quadratic', quad)]:
          w = ridge(fn(hf), y_tr, lam=1e-4)
          e = (apply(w, fn(h_te)) - devB[mode]['y']) ** 2
          cell[f'{fit_name}_{feat_name}_stale'] = float(e[stale_te].mean() * std0 ** 2)
          cell[f'{fit_name}_{feat_name}_fresh'] = float(e[fresh_te].mean() * std0 ** 2)
      res['refit'].append(cell)
      res['fixed_point'].append(fixed_point(mm, devB[mode]['x']))
    res['held_sensor0'] = held_sensor0(seed, d, m, models, mode)
    out['arms'][f'{kind}/{mode}'] = res
  return out


if __name__ == '__main__':
  res = run_parallel(per_seed, [(s,) for s in SEEDS])
  for r in res:
    print('seed', r['seed'], 'stale dev', r['n_dev_stale'], 'oracle', {k: round(v, 4) for k, v in r['window_oracle_mse'].items()},
          'input ceiling', {k: round(v, 3) for k, v in r['input_ceiling'].items()},
          'within stale', {k: round(v, 3) for k, v in r['input_ceiling_within_stale'].items()})
    for name, a in r['arms'].items():
      pr = {k: round(np.mean([p[k] for p in a['probe']]), 3) for k in a['probe'][0]}
      rf = {k: round(np.mean([c[k] for c in a['refit']]), 4) for k in a['refit'][0]}
      fp = {k: round(np.mean([f[k] for f in a['fixed_point']]), 4) for k in ['radius_median', 'radius_q90', 'final_step_norm', 'forecast_shift']}
      pw = {k: round(np.mean([p[k] for p in a['probe_within_stale']]), 3) for k in a['probe'][0]}
      print('  ', name, 'probe', pr, 'within stale', pw, 'ageR2', round(np.mean(a['age_r2_stale']), 3))
      print('     refit', rf)
      print('     fixed point', fp, 'k-decode R2', round(a['held_sensor0']['k_decode_r2'], 3))
      print('     held s0', [(c['k'], round(c['gru_sensitivity'], 2), round(c['kalman_sensitivity'], 2)) for c in a['held_sensor0']['curve']])
  save('e6_representation.json', res)
