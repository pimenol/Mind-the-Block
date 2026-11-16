"""Shared helpers for the Step-1 mechanism experiments.

Nothing here reads the reserved test partition. All evaluation is on dev.
The GRU math is the unchanged `starter.GRU`; feature building reproduces
`starter.windows` exactly (asserted in test_research.py) but also accepts an
explicit mask, so experiments can intervene on the mask per window.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
  os.environ.setdefault(_v, '1')

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from starter import GRU, impose  # noqa: E402

RESULTS = ROOT / 'results'
MODELS = RESULTS / 'models'
SEEDS = [940, 941, 942]
INITS = [0, 1, 2, 3, 4]
LOOKBACK, HORIZON = 24, 6
AGE_BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 10**6)]
AGE_LABELS = ['0', '1', '2', '3-4', '5-8', '9+']


def load(seed):
  d = np.load(ROOT / f'data/synthetic_{seed}.npz')
  # Only train and dev are exposed to experiment code.
  out = {k: d[k] for k in d.files if k.startswith(('train_', 'dev_'))}
  out['mean'] = np.nanmean(out['train_observed'], 0)
  out['std'] = np.nanstd(out['train_observed'], 0)
  return out


def masked(observed, kind, seed):
  """Artificial mask from starter.impose. kind may be 'scatter', 'block',
  or a 6-tuple of kinds per sensor (factorial designs)."""
  if isinstance(kind, str):
    return impose(observed, kind, seed)
  out = observed.copy()
  mask = np.zeros(observed.shape, bool)
  cache = {}
  for j, k in enumerate(kind):
    if k not in cache:
      cache[k] = impose(observed, k, seed)[1]
    mask[:, j] = cache[k][:, j]
  out[mask] = np.nan
  return out, mask


def fill_age(raw, mean):
  """Causal forward fill and age, as in starter.windows."""
  valid = np.isfinite(raw)
  filled = np.empty_like(raw, dtype=float)
  age = np.empty_like(raw, dtype=float)
  prev = mean.astype(float).copy()
  a = np.zeros(raw.shape[1])
  for t in range(len(raw)):
    prev = np.where(valid[t], raw[t], prev)
    a = np.where(valid[t], 0, a + 1)
    filled[t] = prev
    age[t] = a
  return filled, valid, age


def ends_for(n):
  return np.arange(LOOKBACK - 1, n - HORIZON)


def feature_series(filled, valid, age, mean, std, mode):
  """mode: 'age' (full), 'zero' (ages zeroed, starter arm), 'fill'
  (mask and age channels zeroed)."""
  f = (filled - mean) / std
  m = valid.astype(float)
  a = np.minimum(age, 48) / 48
  if mode == 'zero':
    a = np.zeros_like(a)
  if mode == 'fill':
    a = np.zeros_like(a)
    m = np.zeros_like(m)
  return np.concatenate([f, m, a], 1)


def stack_windows(series, ends):
  idx = ends[:, None] + np.arange(-LOOKBACK + 1, 1)[None]
  return series[idx]


def make_split(d, split, kind, mask_seed, mode):
  """Windows for a split under a mask. Returns dict with x, y (normalized),
  persistence (normalized), ends, and exposure arrays. Targets that are
  naturally missing are dropped, as in starter.run."""
  mean, std = d['mean'], d['std']
  raw, _ = masked(d[split + '_observed'], kind, mask_seed)
  filled, valid, age = fill_age(raw, mean)
  ends = ends_for(len(raw))
  keep = ~d[split + '_natural_missing'][ends + HORIZON, 0]
  ends = ends[keep]
  series = feature_series(filled, valid, age, mean, std, mode)
  x = stack_windows(series, ends)
  y = (d[split + '_target'][ends + HORIZON] - mean[0]) / std[0]
  pers = (filled[ends, 0] - mean[0]) / std[0]
  win_valid = stack_windows(valid.astype(float), ends)
  return dict(x=x, y=y, persistence=pers, ends=ends, raw=raw,
              filled=filled, valid=valid, age=age,
              age_end=age[ends], obs_count=win_valid.sum(1))


def train_gru(tx, ty, data_seed, init, epochs=12, per_epoch=1600, hidden=8):
  """Same schedule as starter.run. init=0 reproduces the starter exactly."""
  s = data_seed + 1000 * init
  model = GRU(tx.shape[2], s, hidden)
  rng = np.random.default_rng(s)
  for _ in range(epochs):
    ix = rng.permutation(len(tx))[:per_epoch]
    for start in range(0, len(ix), 32):
      j = ix[start:start + 32]
      model.step(tx[j], ty[j])
  return model


def train_set(d, train_kind, mode, data_seed):
  """Training windows. 'mixed' picks scatter or block per window (50/50)
  with a fixed rng, so the example count and budget are unchanged."""
  if train_kind == 'mixed':
    a = make_split(d, 'train', 'scatter', data_seed + 10, mode)
    b = make_split(d, 'train', 'block', data_seed + 10, mode)
    pick = np.random.default_rng(data_seed + 77).random(len(a['y'])) < .5
    x = np.where(pick[:, None, None], b['x'], a['x'])
    return x, a['y']
  s = make_split(d, 'train', train_kind, data_seed + 10, mode)
  return s['x'], s['y']


def get_model(d, data_seed, train_kind='scatter', mode='zero', init=0,
              epochs=12, hidden=8):
  """Train or load a cached GRU (cache lives in results/models, ignored)."""
  MODELS.mkdir(parents=True, exist_ok=True)
  tag = '' if (epochs, hidden) == (12, 8) else f'_e{epochs}_h{hidden}'
  path = MODELS / f'{data_seed}_{train_kind}_{mode}_{init}{tag}.npz'
  if path.exists():
    p = dict(np.load(path))
    model = GRU(18, 0)
    model.p = {k: (v if v.ndim else np.array(v)) for k, v in p.items()}
    return model
  tx, ty = train_set(d, train_kind, mode, data_seed)
  model = train_gru(tx, ty, data_seed, init, epochs=epochs, hidden=hidden)
  np.savez(path, **model.p)
  return model


def rmse(pred, y, std0):
  return float(np.sqrt(np.mean((pred - y) ** 2)) * std0)


def age_bin_index(age0):
  out = np.full(len(age0), -1)
  for i, (lo, hi) in enumerate(AGE_BINS):
    out[(age0 >= lo) & (age0 <= hi)] = i
  return out


def per_bin(sq_err, age0, std0):
  """Per age-bin MSE in raw units, with counts and mean age."""
  b = age_bin_index(age0)
  rows = []
  for i, label in enumerate(AGE_LABELS):
    sel = b == i
    rows.append(dict(bin=label, n=int(sel.sum()),
                     mean_age=float(age0[sel].mean()) if sel.any() else None,
                     mse=float(sq_err[sel].mean() * std0 ** 2) if sel.any() else None))
  return rows


def run_parallel(fn, args, procs=None):
  from multiprocessing import Pool
  with Pool(procs or min(len(args), 32)) as pool:
    return pool.starmap(fn, args)


def save(name, obj):
  RESULTS.mkdir(exist_ok=True)
  (RESULTS / name).write_text(json.dumps(obj, indent=1))
  print('wrote', RESULTS / name)


def summarize(values):
  v = np.asarray(values, float)
  return dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()),
              sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0)
