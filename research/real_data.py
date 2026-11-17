"""UCI Beijing Multi-Site Air Quality data in the common.load format.

Six "sensors" are log(1 + PM2.5) at six central Beijing stations, hourly.
The natural NaNs stay as natural missingness. Sensor 0 is the target station.
There are 3 target stations; each keeps the other 5 of the 6 as inputs.

Split by time: train 2013-03-01 .. 2015-08-31, dev 2015-09-01 .. 2016-08-31.
The rest (2016-09-01 .. 2017-02-28) is the real test period. load_real never
returns it. Mean and scale come from train observations only.
"""
import csv
import hashlib
import io
import urllib.request
import zipfile

import numpy as np

import common

URL = 'https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip'
SHA256 = 'b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4'
DIR = common.ROOT / 'data' / 'real'
STATIONS = ['Dongsi', 'Guanyuan', 'Tiantan', 'Wanshouxigong', 'Aotizhongxin', 'Nongzhanguan']
TARGETS = STATIONS[:3]
SEED = {name: 500 + i for i, name in enumerate(TARGETS)}  # mask and init seeds per target station
TRAIN_END = (2015, 9, 1)  # first hour not in train
DEV_END = (2016, 9, 1)    # first hour not in dev


def fetch():
  """Download the archive once and check its hash."""
  DIR.mkdir(parents=True, exist_ok=True)
  path = DIR / 'beijing.zip'
  if not path.exists():
    urllib.request.urlretrieve(URL, path)
  data = path.read_bytes()
  assert hashlib.sha256(data).hexdigest() == SHA256, 'archive hash changed'
  return data


def read_all():
  """Hourly log1p(PM2.5) for STATIONS, with (year, month, day) per row."""
  outer = zipfile.ZipFile(io.BytesIO(fetch()))
  inner = zipfile.ZipFile(io.BytesIO(outer.read('PRSA2017_Data_20130301-20170228.zip')))
  cols, dates = [], None
  for st in STATIONS:
    name = [n for n in inner.namelist() if f'_{st}_' in n][0]
    rows = list(csv.DictReader(io.TextIOWrapper(inner.open(name), encoding='utf-8')))
    cols.append([np.log1p(float(r['PM2.5'])) if r['PM2.5'] not in ('', 'NA') else np.nan for r in rows])
    d = [(int(r['year']), int(r['month']), int(r['day'])) for r in rows]
    assert dates is None or d == dates
    dates = d
  return np.array(cols).T, dates


def load_real(target):
  """Same keys as common.load: {train,dev}_{observed,natural_missing,target}, mean, std."""
  x, dates = read_all()
  order = [STATIONS.index(target)] + [i for i, s in enumerate(STATIONS) if s != target]
  x = x[:, order]
  n_tr = next(i for i, d in enumerate(dates) if d >= TRAIN_END)
  n_dev = next(i for i, d in enumerate(dates) if d >= DEV_END)
  out = {}
  for split, sl in [('train', slice(0, n_tr)), ('dev', slice(n_tr, n_dev))]:
    obs = x[sl].copy()
    out[split + '_observed'] = obs
    out[split + '_natural_missing'] = np.isnan(obs)
    out[split + '_target'] = np.nan_to_num(obs[:, 0])  # sensor 0; missing targets are dropped by make_split
  out['mean'] = np.nanmean(out['train_observed'], 0)
  out['std'] = np.nanstd(out['train_observed'], 0)
  out['dates'] = dict(train=(dates[0], dates[n_tr - 1]), dev=(dates[n_tr], dates[n_dev - 1]))
  return out
