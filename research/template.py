"""Experiment template. Copy it: cp research/template.py research/e1_<topic>.py

Run from the repo root: python research/e1_<topic>.py
It writes results/e1_<topic>.json.
"""
from pathlib import Path

import numpy as np

from common import INITS, SEEDS, get_model, load, make_split, rmse, save

NAME = Path(__file__).stem  # results go to results/<NAME>.json


def per_seed(seed):
  d = load(seed)
  s = make_split(d, 'dev', 'block', seed + 20, 'zero')  # dev windows with 8-step outages
  pred = np.mean([get_model(d, seed, 'scatter', 'zero', i).predict(s['x']) for i in INITS], 0)
  return dict(seed=seed, rmse_block=rmse(pred, s['y'], d['std'][0]))


if __name__ == '__main__':
  res = [per_seed(seed) for seed in SEEDS]
  print(res)
  save(f'{NAME}.json', res)
