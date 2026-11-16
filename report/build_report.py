"""Build report/report.html from results/*.json. Every number is read from
the committed result files. Run: python report/build_report.py"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R = {p.stem: json.loads(p.read_text()) for p in (ROOT / 'results').glob('*.json')}
SEEDS = [940, 941, 942]


def by_seed(name):
  return {r['seed']: r for r in R[name]}


E0, E1, E2, E3 = by_seed('e0_baseline_exposure'), by_seed('e1_age_stratified'), by_seed('e2_block_position'), by_seed('e3_window_oracle')
E4, E4B, E5 = by_seed('e4_training_masks'), by_seed('e4b_budget_capacity'), by_seed('e5_cross_sensor')

COL = {'gru': 'var(--s1)', 'kalman': 'var(--s2)', 'persistence': 'var(--s3)', 'gru_block': 'var(--s4)', 'gru_big': 'var(--s5)'}
NAME = {'gru': 'GRU (learned model)', 'kalman': 'Kalman filter (best possible)', 'persistence': 'Persistence (repeat last value)',
        'gru_block': 'GRU, block-trained', 'gru_big': 'GRU, block-trained, 4x steps, 32 units'}


def f(x, d=3):
  return f'{x:.{d}f}'


def fs(x, d=3):
  return f'{x:+.{d}f}'


def cell_list(vals, d=3, signed=False):
  return ''.join(f'<td>{fs(v, d) if signed else f(v, d)}</td>' for v in vals)


def nice_ticks(lo, hi, n=5):
  """Round tick values (1, 2, 2.5 or 5 times a power of 10) that cover lo..hi."""
  raw = (hi - lo) / (n - 1)
  mag = 10 ** np.floor(np.log10(raw))
  step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw))
  t0 = np.floor(lo / step) * step
  return [round(t0 + i * step, 10) for i in range(int(np.ceil((hi - t0) / step)) + 1)]


def tfmt(t):
  return f'{t:g}'


# ---------------------------------------------------------------- charts
def line_chart(cid, title, xs, xlabels, series, ylab, ymin=None, ymax=None, w=560, h=300, zero=False, direct=True):
  """series: list of (key, label, values). Values may contain None."""
  pl, pr, pt, pb = 52, (200 if direct else 16), 16, 40
  allv = [v for _, _, vs in series for v in vs if v is not None]
  lo = min(allv + ([0] if zero else [])) if ymin is None else ymin
  hi = max(allv) if ymax is None else ymax
  pad = (hi - lo) * .08 or .01
  lo, hi = (lo - pad if ymin is None else lo), hi + pad
  X = lambda i: pl + (w - pl - pr) * (i / (len(xs) - 1))
  Y = lambda v: pt + (h - pt - pb) * (1 - (v - lo) / (hi - lo))
  ticks = nice_ticks(lo, hi)
  lo, hi = ticks[0], ticks[-1]
  out = [f'<figure class="chart" id="{cid}"><figcaption>{title}</figcaption>',
         f'<div class="svgwrap"><svg viewBox="0 0 {w} {h}" role="img" aria-label="{title}">']
  for t in ticks:
    out.append(f'<line class="grid" x1="{pl}" x2="{w - pr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
               f'<text class="tick" x="{pl - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{tfmt(t)}</text>')
  if zero and lo < 0 < hi:
    out.append(f'<line class="zero" x1="{pl}" x2="{w - pr}" y1="{Y(0):.1f}" y2="{Y(0):.1f}"/>')
  for i, lab in enumerate(xlabels):
    out.append(f'<text class="tick" x="{X(i):.1f}" y="{h - pb + 18}" text-anchor="middle">{lab}</text>')
  out.append(f'<text class="axis" x="{(pl + w - pr) / 2}" y="{h - 4}" text-anchor="middle">{ylab[0]}</text>')
  out.append(f'<text class="axis" x="12" y="{pt + (h - pt - pb) / 2}" text-anchor="middle" '
             f'transform="rotate(-90 12 {pt + (h - pt - pb) / 2})">{ylab[1]}</text>')
  label_y = []
  for key, label, vs in series:
    pts = [(X(i), Y(v), v, xlabels[i]) for i, v in enumerate(vs) if v is not None]
    d = ' '.join(('M' if j == 0 else 'L') + f'{x:.1f},{y:.1f}' for j, (x, y, _, _) in enumerate(pts))
    out.append(f'<path d="{d}" fill="none" stroke="{COL[key]}" stroke-width="2"/>')
    for x, y, v, xl in pts:
      out.append(f'<circle class="pt" cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{COL[key]}" '
                 f'data-tip="{label} | {xl}: {v:.3f}"/>')
    if not direct:
      continue
    ly = pts[-1][1]
    for prev in label_y:
      if abs(ly - prev) < 13:
        ly = prev + 13
    label_y.append(ly)
    out.append(f'<text class="dlabel" x="{pts[-1][0] + 8:.1f}" y="{ly + 4:.1f}">{label}</text>')
  out.append('</svg></div>')
  out.append('<div class="legend">' + ''.join(
      f'<span><i style="background:{COL[k]}"></i>{lab}</span>' for k, lab, _ in series) + '</div></figure>')
  return ''.join(out)


def stacked_bars(cid, title, rows, w=560, h=230):
  """rows: list of (label, oracle_part, excess_part)."""
  pl, pr, pt, pb = 70, 150, 10, 34
  ticks_ = nice_ticks(0, max(a + b for _, a, b in rows) * 1.1)
  hi = ticks_[-1]
  bh = (h - pt - pb) / len(rows) * .55
  X = lambda v: pl + (w - pl - pr) * v / hi
  out = [f'<figure class="chart" id="{cid}"><figcaption>{title}</figcaption>',
         f'<div class="svgwrap"><svg viewBox="0 0 {w} {h}" role="img" aria-label="{title}">']
  for t in ticks_:
    out.append(f'<line class="grid" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{pt}" y2="{h - pb}"/>'
               f'<text class="tick" x="{X(t):.1f}" y="{h - pb + 16}" text-anchor="middle">{tfmt(t)}</text>')
  for i, (lab, a, b) in enumerate(rows):
    yc = pt + (h - pt - pb) * (i + .5) / len(rows)
    y0 = yc - bh / 2
    out.append(f'<text class="tick" x="{pl - 8}" y="{yc + 4:.1f}" text-anchor="end">{lab}</text>')
    out.append(f'<rect x="{X(0):.1f}" y="{y0:.1f}" width="{max(X(a) - X(0) - 1, 1):.1f}" height="{bh:.1f}" rx="2" '
               f'fill="var(--s2)" class="pt" data-tip="{lab} | oracle gap (information lost): {a:.4f}"/>')
    out.append(f'<rect x="{X(a) + 1:.1f}" y="{y0:.1f}" width="{max(X(a + b) - X(a) - 1, 1):.1f}" height="{bh:.1f}" rx="2" '
               f'fill="var(--s1)" class="pt" data-tip="{lab} | excess (model failure): {b:.4f}"/>')
    out.append(f'<text class="dlabel" x="{X(a + b) + 6:.1f}" y="{yc + 4:.1f}">{100 * b / (a + b):.0f}% model failure</text>')
  out.append(f'<text class="axis" x="{(pl + w - pr) / 2}" y="{h - 2}" text-anchor="middle">'
             'block minus scatter, MSE (raw units squared)</text></svg></div>')
  out.append('<div class="legend"><span><i style="background:var(--s2)"></i>Information lost (Kalman filter gap)</span>'
             '<span><i style="background:var(--s1)"></i>Model failure (GRU gap above the Kalman gap)</span></div></figure>')
  return ''.join(out)


# ---------------------------------------------------------------- numbers
gap_rows = []
for s in SEEDS:
  g, k = E3[s]['mse_gap']['gru_scatter'], E3[s]['mse_gap']['kalman_window24']
  gap_rows.append((f'seed {s}', k, g - k))
excess_pct = [100 * b / (a + b) for _, a, b in gap_rows]

def mean_over(fn):
  return float(np.mean([fn(s) for s in SEEDS]))

offs = [0, 4, 8, 16]
def e2cell(s, sensors, off, key):
  return [c for c in E2[s]['cells'] if c['sensors'] == sensors and c['offset'] == off][0]['delta_mse'][key]

e2_charts = ''
for sensors, lab in [('s0', 'Outage on sensor 0 only'), ('others', 'Outage on the 5 other sensors'), ('all', 'Outage on all sensors')]:
  series = [(key, NAME[key], [mean_over(lambda s: e2cell(s, sensors, o, src)) for o in offs])
            for key, src in [('gru', 'gru_scatter'), ('kalman', 'kalman'), ('persistence', 'persistence')]]
  if sensors == 'others':
    series = series[:2]
  e2_charts += line_chart(f'e2-{sensors}', f'{lab} (mean of 3 seeds)', offs,
                          ['0', '4', '8', '16'], series,
                          ('steps between outage end and forecast origin', 'extra MSE'), zero=True, w=440, h=260, direct=False)

bins = [x['bin'] for x in E4[940]['kalman_sensitivity']]
def sens(s, arm):
  return [x['value'] for x in E4[s]['arms'][arm]['sensitivity_s0']]
def sens_big(s):
  a = [a for a in E4B[s]['arms'] if a['train'] == 'block' and a['epochs'] == 48 and a['hidden'] == 32][0]
  return [x['value'] for x in a['sensitivity_s0']]
sens_series = [
    ('kalman', NAME['kalman'], list(np.mean([[x['value'] for x in E4[s]['kalman_sensitivity']] for s in SEEDS], 0))),
    ('gru', 'GRU, scatter-trained', list(np.mean([sens(s, 'scatter/zero') for s in SEEDS], 0))),
    ('gru_block', 'GRU, block-trained', list(np.mean([sens(s, 'block/zero') for s in SEEDS], 0))),
    ('gru_big', 'GRU, block, 4x training, 32 units', list(np.mean([sens_big(s) for s in SEEDS], 0))),
]
sens_chart = line_chart('sens', 'How much the forecast trusts the last sensor-0 value, by its age (mean of 3 seeds)',
                        list(range(len(bins))), bins, sens_series,
                        ('age of the sensor-0 value at the forecast origin (steps)', 'trust (forecast change per unit)'), ymin=0, w=620, h=320)

def paired_bins(s, key):
  return [x['delta'] for x in E1[s]['paired'][key]['by_block_age']]
paired_series = [(k, NAME[k], list(np.mean([paired_bins(s, src) for s in SEEDS], 0)))
                 for k, src in [('gru', 'gru'), ('kalman', 'kalman'), ('persistence', 'persistence')]]
paired_chart = line_chart('paired', 'H1: extra error on the same window, by the age of the sensor-0 value (mean of 3 seeds)',
                          list(range(len(bins))), bins, paired_series,
                          ('sensor-0 age at the origin under the block mask (steps)', 'extra MSE'), zero=True, w=620, h=300)

# ---------------------------------------------------------------- new simple charts
def hbar_groups(cid, title, groups, keys, xlab, w=600, row=26, signed=False):
  """groups: list of (label, {key: value}). One bar per key inside each group."""
  pl, pr, pt, pb = 150, 70, 8, 38
  gh = row * len(keys) + 12
  h = pt + pb + gh * len(groups)
  vals = [v for _, d in groups for v in d.values()]
  ticks_ = nice_ticks(min(0, min(vals)) if min(vals) < -.05 * max(vals) else 0, max(vals) * 1.08)
  lo, hi = ticks_[0], ticks_[-1]
  X = lambda v: pl + (w - pl - pr) * (v - lo) / (hi - lo)
  out = [f'<figure class="chart" id="{cid}"><figcaption>{title}</figcaption>',
         f'<div class="svgwrap"><svg viewBox="0 0 {w} {h}" role="img" aria-label="{title}">']
  for t in ticks_:
    out.append(f'<line class="grid" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{pt}" y2="{h - pb}"/>'
               f'<text class="tick" x="{X(t):.1f}" y="{h - pb + 16}" text-anchor="middle">{tfmt(t)}</text>')
  if lo < 0:
    out.append(f'<line class="zero" x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{pt}" y2="{h - pb}"/>')
  for gi, (lab, d) in enumerate(groups):
    y0 = pt + gi * gh
    out.append(f'<text class="glabel" x="{pl - 10}" y="{y0 + gh / 2 + 4:.1f}" text-anchor="end">{lab}</text>')
    for ki, k in enumerate(keys):
      v = d[k]
      y = y0 + 6 + ki * row
      x0, x1 = sorted([X(0), X(max(v, lo))])
      out.append(f'<rect class="pt" x="{x0:.1f}" y="{y:.1f}" width="{max(x1 - x0, 1.5):.1f}" height="{row - 6}" rx="3" '
                 f'fill="{COL[k]}" data-tip="{lab} | {NAME[k]}: {v:.4f}"/>')
      out.append(f'<text class="dlabel" x="{max(X(v), X(0)) + 6:.1f}" y="{y + row / 2 + 1:.1f}">{fs(v, 4 if abs(v) < .001 else 3) if signed else f(v, 3)}</text>')
  out.append(f'<text class="axis" x="{(pl + w - pr) / 2}" y="{h - 4}" text-anchor="middle">{xlab}</text></svg></div>')
  out.append('<div class="legend">' + ''.join(f'<span><i style="background:{COL[k]}"></i>{NAME[k]}</span>' for k in keys)
             + '</div></figure>')
  return ''.join(out)


def mask_diagram():
  """Illustration of the two mask kinds on one 24-step window. Drawn, not real data."""
  rng_ = np.random.default_rng(3)
  sc = rng_.random((6, 24)) < .25
  bl = np.zeros((6, 24), bool)
  for j, start in enumerate([16, 3, 12, 0, 9, 14]):
    bl[j, start:start + 8] = True
  bl[0, 16:24] = True
  cw, ch, gap = 13, 15, 2
  def panel(m, x0, title):
    o = [f'<text class="axis" x="{x0}" y="14">{title}</text>']
    for j in range(6):
      y = 24 + j * (ch + gap)
      o.append(f'<text class="tick" x="{x0 - 6}" y="{y + 11}" text-anchor="end">{"sensor 0" if j == 0 else j}</text>')
      for t in range(24):
        cls = 'cell-miss' if m[j, t] else 'cell-obs'
        o.append(f'<rect class="{cls}" x="{x0 + t * cw}" y="{y}" width="{cw - gap}" height="{ch}" rx="2"/>')
    xo = x0 + 24 * cw + 2
    o.append(f'<line class="origin" x1="{xo}" x2="{xo}" y1="20" y2="{24 + 6 * (ch + gap) + 4}"/>'
             f'<text class="tick" x="{xo}" y="{24 + 6 * (ch + gap) + 18}" text-anchor="end">forecast origin</text>')
    return ''.join(o)
  w = 70 + 24 * cw + 110 + 24 * cw + 20
  svg = (f'<svg viewBox="0 0 {w} 150" role="img" aria-label="Scatter and block masks on one window">'
         + panel(sc, 70, 'Scatter mask: single gaps') + panel(bl, 70 + 24 * cw + 110, 'Block mask: 8-step outages') + '</svg>')
  return ('<figure class="chart"><figcaption>The two kinds of missing data (drawing, not real data)</figcaption>'
          f'<div class="svgwrap">{svg}</div><div class="legend"><span><i class="k-obs"></i>reading present</span>'
          '<span><i class="k-miss"></i>reading removed</span></div></figure>')


def stale_diagram():
  """Illustration of forward fill during an outage. Drawn from a simple AR(0.88) series."""
  rng_ = np.random.default_rng(7)
  n, out_start = 30, 16
  x = np.zeros(n)
  for t in range(1, n):
    x[t] = .88 * x[t - 1] + .5 * rng_.standard_normal()
  x = x - x[:out_start].mean() + .3
  origin, target = 23, 29
  fill = x.copy(); fill[out_start:origin + 1] = x[out_start - 1]
  w, h, pl, pr, pt, pb = 600, 230, 20, 50, 20, 36
  lo, hi = x.min() - .4, x.max() + .4
  X = lambda t: pl + (w - pl - pr) * t / (n - 1)
  Y = lambda v: pt + (h - pt - pb) * (1 - (v - lo) / (hi - lo))
  o = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Forward fill makes an old value look current">']
  o.append(f'<rect class="outage" x="{X(out_start - .5):.1f}" y="{pt}" width="{X(origin + .5) - X(out_start - .5):.1f}" height="{h - pt - pb}"/>')
  o.append(f'<text class="tick" x="{(X(out_start) + X(origin)) / 2:.1f}" y="{pt + 12}" text-anchor="middle">outage: no readings</text>')
  o.append('<path class="truth" d="' + ' '.join(('M' if t == 0 else 'L') + f'{X(t):.1f},{Y(x[t]):.1f}' for t in range(n)) + '"/>')
  o.append('<path class="filled" d="' + ' '.join(('M' if i == 0 else 'L') + f'{X(t):.1f},{Y(fill[t]):.1f}' for i, t in enumerate(range(out_start - 1, origin + 1))) + '"/>')
  for t in range(out_start):
    o.append(f'<circle class="obs" cx="{X(t):.1f}" cy="{Y(x[t]):.1f}" r="3.5"/>')
  o.append(f'<line class="origin" x1="{X(origin):.1f}" x2="{X(origin):.1f}" y1="{pt}" y2="{h - pb}"/>')
  o.append(f'<text class="tick" x="{X(origin):.1f}" y="{h - pb + 16}" text-anchor="middle">forecast origin</text>')
  o.append(f'<circle class="target" cx="{X(target):.1f}" cy="{Y(x[target]):.1f}" r="5"/>'
           f'<text class="tick" x="{X(target):.1f}" y="{h - pb + 16}" text-anchor="middle">target (+6)</text>')
  o.append(f'<text class="dlabel" x="{X(origin) - 6:.1f}" y="{Y(fill[origin]) - 8:.1f}" text-anchor="end">filled value is 8 steps old</text>')
  o.append('</svg>')
  return ('<figure class="chart"><figcaption>A stale value: forward fill repeats the last reading through the outage (drawing)</figcaption>'
          f'<div class="svgwrap">{"".join(o)}</div><div class="legend">'
          '<span><i style="background:var(--muted)"></i>true signal</span><span><i class="k-obs"></i>reading the model sees</span>'
          '<span><i style="background:var(--s2)"></i>filled value the model sees</span></div></figure>')


NAME.update({'scatter_mask': 'Scatter mask', 'block_mask': 'Block mask',
             'tr_scatter': 'Trained on scatter', 'tr_block': 'Trained on block', 'tr_mixed': 'Trained on 50/50 mix'})
COL.update({'scatter_mask': 'var(--s3)', 'block_mask': 'var(--s2)',
            'tr_scatter': 'var(--s1)', 'tr_block': 'var(--s4)', 'tr_mixed': 'var(--s5)'})

exposure_chart = hbar_groups('expo', 'Share of windows where sensor 0 is 3 or more steps old at the forecast origin',
    [(f'seed {s}', {'scatter_mask': E0[s]['conditions']['scatter']['exposure']['age0_ge3'],
                    'block_mask': E0[s]['conditions']['block']['exposure']['age0_ge3']}) for s in SEEDS],
    ['scatter_mask', 'block_mask'], 'share of windows (0.20 = 1 in 5)')

baseline_chart = hbar_groups('base', 'The problem: error goes up under the block mask (RMSE, block minus scatter)',
    [(f'seed {s}', {'gru': E0[s]['conditions']['block']['gru']['zero']['mean'] - E0[s]['conditions']['scatter']['gru']['zero']['mean'],
                    'persistence': E0[s]['conditions']['block']['persistence'] - E0[s]['conditions']['scatter']['persistence'],
                    'kalman': E0[s]['conditions']['block']['kalman'] - E0[s]['conditions']['scatter']['kalman']}) for s in SEEDS],
    ['gru', 'persistence', 'kalman'], 'extra RMSE under the block mask (higher is worse)')

train_chart = hbar_groups('train', 'H3: does training on block masks remove the gap? (RMSE, block minus scatter)',
    [(f'seed {s}', {'tr_scatter': E4[s]['arms']['scatter/zero']['gap']['mean'], 'tr_block': E4[s]['arms']['block/zero']['gap']['mean'],
                    'tr_mixed': E4[s]['arms']['mixed/zero']['gap']['mean']}) for s in SEEDS],
    ['tr_scatter', 'tr_block', 'tr_mixed'], 'extra RMSE under the block mask (0 = no gap)', signed=True)

def fac(s, cell, k):
  return E5[s]['factorial'][cell][k] - E5[s]['factorial']['S0 scatter / others scatter'][k]
FAC_CELLS = [('S0 block / others scatter', 'only sensor 0'), ('S0 scatter / others block', 'only the 5 others'),
             ('S0 block / others block', 'all sensors')]
factorial_chart = hbar_groups('fac', 'H4: which outages cost error? (RMSE, against all-scatter)',
    [(f'{s} · {lab}', {'gru': fac(s, c, 'gru_scatter'), 'kalman': fac(s, c, 'kalman')}) for s in SEEDS for c, lab in FAC_CELLS],
    ['gru', 'kalman'], 'extra RMSE when these sensors get block outages', row=22, signed=True)


# ---------------------------------------------------------------- tables
def exposure_table():
  rows = []
  for s in SEEDS:
    for kind in ['scatter', 'block']:
      e = E0[s]['conditions'][kind]['exposure']
      ex = E3[s]['exposure']
      outages = f"{ex['age0>=5']['windows']} / {ex['age0>=5']['distinct_outages']}" if kind == 'block' else '-'
      rows.append(f'<tr><td>{s}</td><td>{kind}</td><td>{E0[s]["conditions"][kind]["n"]}</td>'
                  f'<td>{f(e["age0_mean"], 2)}</td><td>{100 * e["age0_ge3"]:.1f}%</td>'
                  f'<td>{f(e["age_all_mean"], 2)}</td><td>{f(e["n_stale_sensors_mean"], 2)}</td>'
                  f'<td>{f(e["obs_count_s0_mean"], 1)}</td><td>{e["obs_count_s0_min"]}</td>'
                  f'<td>{100 * e["min_age_across_sensors_gt0"]:.1f}%</td><td>{f(e["gap_run_mean_s0"], 1)}</td>'
                  f'<td>{outages}</td></tr>')
  return ''.join(rows)


def baseline_table():
  rows = []
  for s in SEEDS:
    c = E0[s]['conditions']
    for mode, lab in [('zero', 'GRU, ages zeroed'), ('age', 'GRU, age input'), ('fill', 'GRU, fills only')]:
      sc, bl = c['scatter']['gru'][mode], c['block']['gru'][mode]
      rows.append(f'<tr><td>{s}</td><td>{lab}</td><td>{f(sc["mean"], 4)}</td><td>{f(bl["mean"], 4)}</td>'
                  f'<td>{fs(bl["mean"] - sc["mean"], 4)}</td><td>{f(sc["min"], 3)} to {f(sc["max"], 3)}</td></tr>')
    for key, lab in [('persistence', 'Persistence'), ('kalman', 'Kalman oracle')]:
      rows.append(f'<tr><td>{s}</td><td>{lab}</td><td>{f(c["scatter"][key], 4)}</td><td>{f(c["block"][key], 4)}</td>'
                  f'<td>{fs(c["block"][key] - c["scatter"][key], 4)}</td><td>-</td></tr>')
  return ''.join(rows)


def fresh_stale_table():
  rows = []
  for s in SEEDS:
    st, fr = E3[s]['stale'], E3[s]['fresh']
    p = E1[s]['paired']['persistence']
    rows.append(f'<tr><td>{s}</td><td>{E5[s]["n_fresh"]}</td><td>{fs(fr["gru_scatter"])}</td><td>{fs(fr["kalman_window24"])}</td>'
                f'<td>{fs(p["both_fresh"])}</td><td>{E5[s]["n_stale"]}</td><td>{fs(st["gru_scatter"])}</td>'
                f'<td>{fs(st["kalman_window24"])}</td><td>{fs(E5[s]["paired_block_minus_scatter"]["kalman_s0"]["stale"])}</td></tr>')
  return ''.join(rows)


def e2_table():
  rows = []
  for s in SEEDS:
    for sensors in ['s0', 'others', 'all']:
      vals = [e2cell(s, sensors, 0, k) for k in ['gru_scatter', 'kalman', 'kalman_s0', 'persistence']]
      early = max(abs(e2cell(s, sensors, o, 'gru_scatter')) for o in [8, 16])
      rows.append(f'<tr><td>{s}</td><td>{sensors}</td>{cell_list(vals, 4, True)}<td>{f(early, 4)}</td></tr>')
  return ''.join(rows)


def training_table():
  rows = []
  for s in SEEDS:
    for arm in ['scatter/zero', 'block/zero', 'mixed/zero', 'block/age']:
      a = E4[s]['arms'][arm]
      st = [x for x in a['paired_delta_by_block_age'] if x['bin'] in ('5-8', '9+')]
      stale = sum(x['delta'] * x['n'] for x in st) / sum(x['n'] for x in st)
      rows.append(f'<tr><td>{s}</td><td>{arm.replace("/", ", ")}</td><td>{f(a["rmse_scatter"]["mean"], 4)}</td>'
                  f'<td>{f(a["rmse_block"]["mean"], 4)}</td><td>{fs(a["gap"]["mean"], 4)}</td>'
                  f'<td>{fs(a["gap"]["min"], 4)} to {fs(a["gap"]["max"], 4)}</td><td>{fs(stale)}</td></tr>')
  return ''.join(rows)


def e4b_table():
  rows = []
  for s in SEEDS:
    for a in E4B[s]['arms']:
      sv = [x['value'] for x in a['sensitivity_s0']]
      rows.append(f'<tr><td>{s}</td><td>{a["train"]}</td><td>{a["epochs"]}</td><td>{a["hidden"]}</td>'
                  f'<td>{f(a["rmse_scatter"]["mean"], 4)}</td><td>{f(a["rmse_block"]["mean"], 4)}</td>'
                  f'<td>{fs(a["stale_paired_delta"])}</td><td>{f(sv[0], 2)} / {f(sv[-1], 2)}</td></tr>')
  return ''.join(rows)


def factorial_table():
  rows = []
  for s in SEEDS:
    base = E5[s]['factorial']['S0 scatter / others scatter']
    for name, v in E5[s]['factorial'].items():
      cells = ''.join(f'<td>{f(v[k], 4)}<span class="d">{fs(v[k] - base[k], 4) if v is not base else ""}</span></td>'
                      for k in ['gru_scatter', 'gru_block', 'ridge_block', 'persistence', 'kalman'])
      rows.append(f'<tr><td>{s}</td><td>{name}</td>{cells}</tr>')
  return ''.join(rows)


# ---------------------------------------------------------------- summary numbers used in text
oracle_share = [100 - p for p in excess_pct]
others_gru = [e2cell(s, 'others', 0, 'gru_scatter') for s in SEEDS]
others_kf = [e2cell(s, 'others', 0, 'kalman') for s in SEEDS]
s0_gru = [e2cell(s, 's0', 0, 'gru_scatter') for s in SEEDS]
s0_kf = [e2cell(s, 's0', 0, 'kalman') for s in SEEDS]
age_vs_zero = max(abs(E4[s]['arms']['block/age']['rmse_block']['mean'] - E4[s]['arms']['block/zero']['rmse_block']['mean']) for s in SEEDS)
kf_sens = sens_series[0][2]
gru_sens = sens_series[1][2]

fac_fr = [fac(s_, c, 'gru_scatter') / fac(s_, 'S0 block / others block', 'gru_scatter') for s_ in SEEDS for c, _ in FAC_CELLS[:2]]
fac_share = f'{100 * min(fac_fr):.0f}% to {100 * max(fac_fr):.0f}%'
fac_kf_less = all(fac(s_, c, 'kalman') < fac(s_, c, 'gru_scatter') for s_ in SEEDS for c, _ in FAC_CELLS)
assert fac_kf_less


def rng(vals, d=3):
  return f'{min(vals):.{d}f} to {max(vals):.{d}f}'


html = (ROOT / 'report' / 'template.html').read_text()
subs = {
    'EXPOSURE_ROWS': exposure_table(), 'BASELINE_ROWS': baseline_table(), 'FRESH_STALE_ROWS': fresh_stale_table(),
    'E2_ROWS': e2_table(), 'TRAIN_ROWS': training_table(), 'E4B_ROWS': e4b_table(), 'FACTORIAL_ROWS': factorial_table(),
    'GAP_CHART': stacked_bars('gap', 'Main result: the GRU error gap, split into information lost and model failure', gap_rows),
    'E2_CHARTS': e2_charts, 'MASK_DIAGRAM': mask_diagram(), 'STALE_DIAGRAM': stale_diagram(),
    'EXPOSURE_CHART': exposure_chart, 'BASELINE_CHART': baseline_chart, 'TRAIN_CHART': train_chart, 'FACTORIAL_CHART': factorial_chart, 'SENS_CHART': sens_chart, 'PAIRED_CHART': paired_chart,
    'EXCESS_RANGE': f'{min(excess_pct):.0f}% to {max(excess_pct):.0f}%',
    'ORACLE_RANGE': f'{min(oracle_share):.0f}% to {max(oracle_share):.0f}%',
    'OTHERS_GRU': rng(others_gru), 'OTHERS_KF': rng(others_kf), 'S0_GRU': rng(s0_gru), 'S0_KF': rng(s0_kf),
    'AGE_VS_ZERO': f'{age_vs_zero:.4f}', 'FAC_SHARE': fac_share,
    'KF_SENS_0': f(kf_sens[0], 2), 'KF_SENS_9': f(kf_sens[-1], 2), 'GRU_SENS_0': f(gru_sens[0], 2), 'GRU_SENS_9': f(gru_sens[-1], 2),
}
for k, v in subs.items():
  html = html.replace('{{' + k + '}}', v)
assert '{{' not in html, html[html.index('{{'):html.index('{{') + 40]
(ROOT / 'report' / 'report.html').write_text(html)
print('wrote report/report.html', len(html), 'bytes')
for k in ['EXCESS_RANGE', 'ORACLE_RANGE', 'OTHERS_GRU', 'OTHERS_KF', 'S0_GRU', 'S0_KF', 'AGE_VS_ZERO']:
  print(k, subs[k])
print('sens kf', [round(x, 2) for x in kf_sens], 'gru', [round(x, 2) for x in gru_sens])
