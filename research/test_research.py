"""Checks for the research helpers. Run: python research/test_research.py"""
import common  # noqa: F401  (sets thread env vars first)
from common import (SEEDS, fill_age, load, make_split, masked, train_gru,
                    rmse)
import numpy as np
import json
from pathlib import Path
from starter import windows, impose
import oracle


def test_features_match_starter():
  for seed in SEEDS:
    d = load(seed)
    for kind in ['scatter', 'block']:
      raw, _ = impose(d['dev_observed'], kind, seed + 20)
      for use_age, mode in [(False, 'zero'), (True, 'age')]:
        x, y, pers, ends = windows(raw, d['dev_target'], d['mean'], d['std'], use_age)
        keep = ~d['dev_natural_missing'][ends + 6, 0]
        s = make_split(d, 'dev', kind, seed + 20, mode)
        assert np.allclose(s['x'], x[keep]) and np.allclose(s['y'], y[keep])
        assert np.allclose(s['persistence'], pers[keep])


def test_factorial_mask_counts():
  d = load(940)
  _, a = masked(d['dev_observed'], ('block',) + ('scatter',) * 5, 960)
  _, b = impose(d['dev_observed'], 'block', 960)
  _, c = impose(d['dev_observed'], 'scatter', 960)
  assert np.array_equal(a[:, 0], b[:, 0]) and np.array_equal(a[:, 1:], c[:, 1:])


def test_generator_recovery():
  for seed in SEEDS:
    d = load(seed)
    _, x = oracle.recover(seed)
    for split, lo in [('train', 0), ('dev', 2800)]:
      obs = d[split + '_observed']
      n = len(obs)
      ok = np.isfinite(obs)
      assert np.allclose(obs[ok], x[lo:lo + n][ok].astype('float32'))
      assert np.allclose(d[split + '_target'], x[lo:lo + n, 0].astype('float32'))


def test_kalman_beats_persistence_and_restart_matches():
  seed = 940
  d = load(seed)
  m, _ = oracle.recover(seed)
  s = make_split(d, 'dev', 'scatter', seed + 20, 'zero')
  mean, var = oracle.kalman_forecast(s['raw'], m, 'dev', s['ends'])
  y = d['dev_target'][s['ends'] + 6]
  kf = np.sqrt(np.mean((mean - y) ** 2))
  pers = rmse(s['persistence'], s['y'], d['std'][0])
  assert kf < pers, (kf, pers)
  # Predicted variance is calibrated on average (within 15%).
  assert abs(np.mean((mean - y) ** 2) / var.mean() - 1) < .15
  # Batched restart with the unchanged mask reproduces the full filter.
  means, covs, t, c = oracle.kalman_states(s['raw'], m, 'dev')
  ends = s['ends'][s['ends'] >= 30]
  starts = ends - 10
  use = np.ones((len(ends), 11, 6), bool)
  m2, v2 = oracle.batched_restart(means, covs, t, c, s['raw'], m, starts, ends, use)
  m1, v1 = oracle.kalman_forecast(s['raw'], m, 'dev', ends)
  assert np.allclose(m1, m2, atol=1e-5) and np.allclose(v1, v2, atol=1e-5)


def test_reproduces_validation():
  ref = json.loads((Path(common.ROOT) / 'VALIDATION.json').read_text())
  for run in ref['standard_runs']:
    seed = run['seed']
    d = load(seed)
    for r in run['result']['results']:
      mode = 'age' if r['age_input'] else 'zero'
      tx, ty = common.train_set(d, 'scatter', mode, seed)
      model = train_gru(tx, ty, seed, 0)
      s = make_split(d, 'dev', r['condition'], seed + 20, mode)
      got = rmse(model.predict(s['x']), s['y'], d['std'][0])
      assert abs(got - r['rmse']) < 1e-9, (seed, r, got)
      assert abs(rmse(s['persistence'], s['y'], d['std'][0]) - r['persistence_rmse']) < 1e-6


def test_no_test_partition_access():
  for f in Path(__file__).parent.glob('*.py'):
    if f.name == 'test_research.py':
      continue
    text = f.read_text()
    assert "'test_" not in text and "'test'" not in text, f
  assert not any(k.startswith('test_') for k in load(940))


def test_gru_step_matches_predict():
  import e6_representation as e6
  d = load(940)
  s = make_split(d, 'dev', 'block', 960, 'zero')
  model = train_gru(s['x'][:64], s['y'][:64], 940, 0, epochs=1, per_epoch=64)
  h = np.zeros((64, 8))
  for t in range(s['x'].shape[1]):
    h = e6.gru_step(model.p, s['x'][:64, t], h)
  assert np.allclose(h, model.predict(s['x'][:64], cache=True)[1])


def test_window_kf_split_default_and_block_default():
  import e2_block_position as e2
  import e3_window_oracle as e3
  d = load(941)
  m, _ = oracle.recover(941)
  s = make_split(d, 'dev', 'block', 961, 'zero')
  mean, var = e3.window_kf(s, m, return_var=True)
  assert np.array_equal(mean, e3.window_kf(s, m, split='dev')) and (var > 0).all()
  a = e2.intervene(s, d, 4, [0], 'zero')
  b = e2.intervene(s, d, 4, [0], 'zero', block=e2.BLOCK)
  assert all(np.array_equal(u, v) for u, v in zip(a, b))


def test_torch_models_shape_and_determinism():
  import torch_models as tm
  d = load(940)
  s = make_split(d, 'train', 'scatter', 950, 'zero')
  x, y = s['x'][:256], s['y'][:256]
  for arch in tm.ARCH:
    a = tm.train(arch, x, y, 940, 0, epochs=1, per_epoch=64)
    b = tm.train(arch, x, y, 940, 0, epochs=1, per_epoch=64)
    pa = a.predict(x[:50])
    assert pa.shape == (50,) and np.array_equal(pa, b.predict(x[:50]))
    assert a.hidden(x[:5]).ndim == 2
  t = tm.train('tf', x, y, 940, 0, epochs=1, per_epoch=64)
  assert np.allclose(t.attention(x[:5]).sum(1), 1, atol=1e-5)


def test_real_data_no_test_period():
  import real_data as rd
  d = rd.load_real(rd.TARGETS[0])
  assert not [k for k in d if k.startswith('test')]
  assert d['dates']['train'][1] < d['dates']['dev'][0] and d['dates']['dev'][1] < rd.DEV_END
  assert d['train_observed'].shape[1] == 6 and d['dev_target'].ndim == 1


if __name__ == '__main__':
  for name, fn in list(globals().items()):
    if name.startswith('test_'):
      fn()
      print('PASS', name)
