"""PyTorch models for Step 1c: a tiny transformer and an 8-unit GRU.

Both wrap a torch module behind the same NumPy interface as starter.GRU
(`predict(x) -> (n,)`), so the model-agnostic helpers (e4.gru_sensitivity,
e2.intervene, the probes) work unchanged. Training uses the same minibatch
order as common.train_gru: rng seed data_seed + 1000 * init, per_epoch
windows per epoch, batch 32. The optimizer is Adam, not plain SGD.
CPU only, one thread per process.

Transformer size and regularization (width 16, ff 32, 2 layers, 2 heads,
dropout 0.1, weight decay 0.01) were picked once from 8 configurations by dev
scatter RMSE (init 0, scatter/zero). Width 32 without dropout overfits the
~2800 train windows. The picked model is still 5-8% worse than the GRU on dev
scatter.
"""
import numpy as np
import torch
from torch import nn

import common

torch.set_num_threads(1)
LR = 1e-3
WD = .01  # AdamW weight decay
CLIP = 5.


class Block(nn.Module):
  """Pre-norm encoder layer that also returns the attention weights."""

  def __init__(self, d, heads, ff, drop):
    super().__init__()
    self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
    self.att = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
    self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, d))

  def forward(self, z):
    q = self.n1(z)
    a, w = self.att(q, q, q, need_weights=True, average_attn_weights=True)
    z = z + a
    return z + self.ff(self.n2(z)), w


class Transformer(nn.Module):
  def __init__(self, n_in=18, d=16, heads=2, layers=2, ff=32, drop=.1, length=common.LOOKBACK):
    super().__init__()
    self.inp = nn.Linear(n_in, d)
    self.pos = nn.Parameter(torch.randn(length, d) * .02)
    self.blocks = nn.ModuleList([Block(d, heads, ff, drop) for _ in range(layers)])
    self.norm = nn.LayerNorm(d)
    self.out = nn.Linear(d, 1)

  def features(self, x):
    z = self.inp(x) + self.pos
    ws = []
    for b in self.blocks:
      z, w = b(z)
      ws.append(w)
    return self.norm(z[:, -1]), ws

  def forward(self, x):
    return self.out(self.features(x)[0])[:, 0]


class GRUNet(nn.Module):
  def __init__(self, n_in=18, hidden=8):
    super().__init__()
    self.gru = nn.GRU(n_in, hidden, batch_first=True)
    self.out = nn.Linear(hidden, 1)

  def features(self, x):
    return self.gru(x)[1][0], []

  def forward(self, x):
    return self.out(self.features(x)[0])[:, 0]


ARCH = {'tf': Transformer, 'tgru': GRUNet}


class Model:
  """NumPy interface around a torch module."""

  def __init__(self, net):
    self.net = net.eval()

  def _run(self, x, fn, bs=4096):
    with torch.no_grad():
      return [fn(torch.as_tensor(x[i:i + bs], dtype=torch.float32)) for i in range(0, len(x), bs)]

  def predict(self, x):
    return np.concatenate([p.numpy() for p in self._run(x, self.net)]).astype(float)

  def hidden(self, x):
    return np.concatenate([f[0].numpy() for f in self._run(x, self.net.features)]).astype(float)

  def attention(self, x, layer=-1):
    """Attention of the last token over the window, mean over heads: (n, L)."""
    return np.concatenate([f[1][layer][:, -1].numpy() for f in self._run(x, self.net.features)]).astype(float)


def train(arch, tx, ty, data_seed, init, epochs=12, per_epoch=1600, lr=LR, wd=WD, **kw):
  s = data_seed + 1000 * init
  torch.manual_seed(s)
  net = ARCH[arch](tx.shape[2], **kw)
  opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
  rng = np.random.default_rng(s)
  X, Y = torch.as_tensor(tx, dtype=torch.float32), torch.as_tensor(ty, dtype=torch.float32)
  net.train()
  for _ in range(epochs):
    ix = rng.permutation(len(tx))[:per_epoch]
    for start in range(0, len(ix), 32):
      j = torch.as_tensor(ix[start:start + 32])
      loss = ((net(X[j]) - Y[j]) ** 2).mean()
      opt.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(net.parameters(), CLIP)
      opt.step()
  return Model(net)


def cached(arch, name, build, n_in=18):
  """Load results/models/{name}.pt, or call build() and save the weights."""
  common.MODELS.mkdir(parents=True, exist_ok=True)
  path = common.MODELS / f'{name}.pt'
  if path.exists():
    net = ARCH[arch](n_in)
    net.load_state_dict(torch.load(path))
    return Model(net)
  model = build()
  torch.save(model.net.state_dict(), path)
  return model
