"""Personal experiment file. Copy it: cp template.py <your_name>.py

Only you edit your file, so git never has a conflict on it.
Run from the repo root: python research/scratch/<your_name>.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # makes research/ importable

import numpy as np

from common import INITS, SEEDS, get_model, load, make_split, rmse, save

NAME = Path(__file__).stem  # results go to results/scratch_<NAME>.json


def per_seed(seed):
  d = load(seed)
  s = make_split(d, 'dev', 'block', seed + 20, 'zero')  # dev windows with 8-step outages
  pred = np.mean([get_model(d, seed, 'scatter', 'zero', i).predict(s['x']) for i in INITS], 0)
  return dict(seed=seed, rmse_block=rmse(pred, s['y'], d['std'][0]))


if __name__ == '__main__':
  res = [per_seed(seed) for seed in SEEDS]
  print(res)
  save(f'scratch_{NAME}.json', res)
