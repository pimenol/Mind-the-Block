"""Shared helpers. Nothing here reads the test partition.

The GRU is the unchanged `starter.GRU`. `make_split` builds the same features
as `starter.windows`, and returns the mask, fill and age arrays too.
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


def load(seed):
  d = np.load(ROOT / f'data/synthetic_{seed}.npz')
  # Only train and dev are exposed to experiment code.
  out = {k: d[k] for k in d.files if k.startswith(('train_', 'dev_'))}
  out['mean'] = np.nanmean(out['train_observed'], 0)
  out['std'] = np.nanstd(out['train_observed'], 0)
  return out


def masked(observed, kind, seed):
  """Artificial mask from starter.impose: 'scatter' or 'block'."""
  return impose(observed, kind, seed)


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
  """mode: 'age' (age features) or 'zero' (zeros in the age slots), as in starter.run."""
  f = (filled - mean) / std
  m = valid.astype(float)
  a = np.minimum(age, 48) / 48
  if mode == 'zero':
    a = np.zeros_like(a)
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
  """Training windows with mask seed data_seed + 10, as in starter.run."""
  s = make_split(d, 'train', train_kind, data_seed + 10, mode)
  return s['x'], s['y']


def get_model(d, data_seed, train_kind='scatter', mode='zero', init=0,
              epochs=12, hidden=8):
  """Train or load a cached GRU (cache lives in results/models, ignored)."""
  tag = '' if (epochs, hidden) == (12, 8) else f'_e{epochs}_h{hidden}'
  name = f'{data_seed}_{train_kind}_{mode}_{init}{tag}'

  def build():
    tx, ty = train_set(d, train_kind, mode, data_seed)
    return train_gru(tx, ty, data_seed, init, epochs=epochs, hidden=hidden)
  return cached_model(name, build)


def cached_model(name, build):
  """Load results/models/{name}.npz, or call build() and save the result."""
  MODELS.mkdir(parents=True, exist_ok=True)
  path = MODELS / f'{name}.npz'
  if path.exists():
    p = dict(np.load(path))
    model = GRU(18, 0)
    model.p = {k: (v if v.ndim else np.array(v)) for k, v in p.items()}
    return model
  model = build()
  np.savez(path, **model.p)
  return model


def rmse(pred, y, std0):
  return float(np.sqrt(np.mean((pred - y) ** 2)) * std0)



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
