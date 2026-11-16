"""Bayes-optimal reference forecaster for the synthetic benchmark.

generate_data.py is deterministic: a 5-dim AR(0.88) latent state s_t with
known seasonal offsets c_t, observed through a fixed 5x6 mixing matrix M plus
N(0, 0.1^2) sensor noise. We rebuild M by replaying the generator's RNG, then
run a Kalman filter that simply skips missing readings. Its forecast of
x0[e+6] is the minimum-MSE forecast given exactly the readings a model sees.
This is an analysis tool (it uses the true model), not a method.
"""
import numpy as np

PHI, Q, R = .88, .22 ** 2, .1 ** 2
OFFSET = {'train': 0, 'dev': 2800}
BIG = 1e8


def recover(seed):
  """Replay generate_data.generate(seed) and return (M, x)."""
  rng = np.random.default_rng(seed)
  t = np.arange(4800)
  a = np.zeros((4800, 5))
  for i in range(1, len(t)):
    a[i] = PHI * a[i - 1] + rng.normal(0, .22, 5)
  a[:, 0] += np.sin(2 * np.pi * t / 96)
  a[:, 1] += .6 * np.cos(2 * np.pi * t / 37)
  m = rng.normal(0, .65, (5, 6))
  x = a @ m + rng.normal(0, .1, (len(t), 6))
  return m, x


def season(t):
  c = np.zeros((len(t), 5))
  c[:, 0] = np.sin(2 * np.pi * t / 96)
  c[:, 1] = .6 * np.cos(2 * np.pi * t / 37)
  return c


def kalman_forecast(raw, m, split, ends, sensors=None, horizon=6):
  """Filter a partition (NaN = missing) from a stationary prior at the
  partition start. Returns forecast mean and variance of x0[e+horizon] for
  each window end e, in raw units. sensors limits which sensors are used."""
  means, covs, t, c = kalman_states(raw, m, split, sensors)
  return _forecast(means[ends], covs[ends], c, ends, m, horizon, len(raw), t)


def _forecast(mu, p, c, ends, m, horizon, n, t):
  f = PHI ** horizon
  extra = Q * sum(PHI ** (2 * j) for j in range(horizon))
  tt = t[ends] + horizon
  c_future = season(tt)
  w = m[:, 0]
  mean = (f * mu + c_future) @ w
  var = np.einsum('i,nij,j->n', w, f * f * p, w) + extra * (w @ w) + R
  return mean, var


def kalman_states(raw, m, split, sensors=None):
  """Filtered mean/cov at every step (used to restart per-window filters)."""
  n = len(raw)
  t = np.arange(n) + OFFSET[split]
  c = season(t)
  use = np.isfinite(raw)
  if sensors is not None:
    keep = np.zeros(6, bool)
    keep[list(sensors)] = True
    use &= keep[None]
  mu = np.zeros(5)
  p = np.eye(5) * Q / (1 - PHI ** 2)
  means = np.empty((n, 5))
  covs = np.empty((n, 5, 5))
  h = m.T
  for k in range(n):
    if k:
      mu = PHI * mu
      p = PHI ** 2 * p + Q * np.eye(5)
    o = use[k]
    if o.any():
      ho = h[o]
      s = ho @ p @ ho.T + R * np.eye(o.sum())
      gain = np.linalg.solve(s, ho @ p).T
      mu = mu + gain @ (np.where(o, raw[k], 0)[o] - ho @ (mu + c[k]))
      p = p - gain @ ho @ p
    means[k] = mu
    covs[k] = p
  return means, covs, t, c


def batched_restart(means, covs, t, c, raw, m, starts, stops, use, horizon=6,
                    bump=None):
  """Per-window filters run in a batch. Window w restarts from the filtered
  state at starts[w]-1 and assimilates raw[starts[w]..stops[w]] using the
  boolean matrix use (windows x steps x 6). Missing readings get a huge
  noise variance, which is numerically the same as skipping them."""
  nw = len(starts)
  mu = means[starts - 1].copy()
  p = covs[starts - 1].copy()
  h = m.T
  steps = use.shape[1]
  for k in range(steps):
    tk = starts + k
    live = tk <= stops
    if not live.any():
      break
    tkc = np.minimum(tk, len(raw) - 1)
    mu_p = PHI * mu
    p_p = PHI ** 2 * p + Q * np.eye(5)
    o = use[:, k] & np.isfinite(raw[tkc])
    rdiag = np.where(o, R, BIG)
    yk = np.where(o, np.nan_to_num(raw[tkc]), 0.)
    if bump is not None and k == 0:
      yk = yk + bump  # perturb the first assimilated reading (sensitivity)
    pred = (mu_p + c[tkc]) @ m
    s = np.einsum('ij,njk,kl->nil', h, p_p, h.T) + rdiag[:, :, None] * np.eye(6)
    phT = np.einsum('nij,jk->nik', p_p, h.T)
    gain = np.linalg.solve(s, np.transpose(phT, (0, 2, 1))).transpose(0, 2, 1)
    innov = np.where(o, yk - pred, 0.)
    mu_n = mu_p + np.einsum('nij,nj->ni', gain, innov)
    p_n = p_p - np.einsum('nij,jk,nkl->nil', gain, h, p_p)
    mu = np.where(live[:, None], mu_n, mu)
    p = np.where(live[:, None, None], p_n, p)
  ends = stops
  return _forecast(mu, p, c, ends, m, horizon, len(raw), t)
