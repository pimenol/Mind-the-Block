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
  seed = 940
  d = load(seed)
  tx, ty = common.train_set(d, 'scatter', 'zero', seed)
  model = train_gru(tx, ty, seed, 0)
  s = make_split(d, 'dev', 'block', seed + 20, 'zero')
  got = rmse(model.predict(s['x']), s['y'], d['std'][0])
  want = [r for r in ref['standard_runs'][0]['result']['results']
          if not r['age_input'] and r['condition'] == 'block'][0]['rmse']
  assert abs(got - want) < 1e-9, (got, want)


def test_no_test_partition_access():
  for f in Path(__file__).parent.glob('*.py'):
    if f.name == 'test_research.py':
      continue
    text = f.read_text()
    assert "'test_" not in text and "'test'" not in text, f
  assert not any(k.startswith('test_') for k in load(940))


if __name__ == '__main__':
  for name, fn in list(globals().items()):
    if name.startswith('test_'):
      fn()
      print('PASS', name)
