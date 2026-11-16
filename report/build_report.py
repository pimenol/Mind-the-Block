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
NAME = {'gru': 'GRU, scatter-trained', 'kalman': 'Kalman oracle', 'persistence': 'Persistence',
        'gru_block': 'GRU, block-trained', 'gru_big': 'GRU, block-trained, 4x steps, 32 units'}


def f(x, d=3):
  return f'{x:.{d}f}'


def fs(x, d=3):
  return f'{x:+.{d}f}'


def cell_list(vals, d=3, signed=False):
  return ''.join(f'<td>{fs(v, d) if signed else f(v, d)}</td>' for v in vals)


# ---------------------------------------------------------------- charts
def line_chart(cid, title, xs, xlabels, series, ylab, ymin=None, ymax=None, w=560, h=300, zero=False):
  """series: list of (key, label, values). Values may contain None."""
  pl, pr, pt, pb = 52, 150, 16, 40
  allv = [v for _, _, vs in series for v in vs if v is not None]
  lo = min(allv + ([0] if zero else [])) if ymin is None else ymin
  hi = max(allv) if ymax is None else ymax
  pad = (hi - lo) * .08 or .01
  lo, hi = (lo - pad if ymin is None else lo), hi + pad
  X = lambda i: pl + (w - pl - pr) * (i / (len(xs) - 1))
  Y = lambda v: pt + (h - pt - pb) * (1 - (v - lo) / (hi - lo))
  ticks = np.linspace(lo, hi, 5)
  out = [f'<figure class="chart" id="{cid}"><figcaption>{title}</figcaption>',
         f'<div class="svgwrap"><svg viewBox="0 0 {w} {h}" role="img" aria-label="{title}">']
  for t in ticks:
    out.append(f'<line class="grid" x1="{pl}" x2="{w - pr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
               f'<text class="tick" x="{pl - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{t:.2f}</text>')
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
  pl, pr, pt, pb = 70, 120, 10, 34
  hi = max(a + b for _, a, b in rows) * 1.1
  bh = (h - pt - pb) / len(rows) * .55
  X = lambda v: pl + (w - pl - pr) * v / hi
  out = [f'<figure class="chart" id="{cid}"><figcaption>{title}</figcaption>',
         f'<div class="svgwrap"><svg viewBox="0 0 {w} {h}" role="img" aria-label="{title}">']
  for t in np.linspace(0, hi, 5):
    out.append(f'<line class="grid" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{pt}" y2="{h - pb}"/>'
               f'<text class="tick" x="{X(t):.1f}" y="{h - pb + 16}" text-anchor="middle">{t:.3f}</text>')
  for i, (lab, a, b) in enumerate(rows):
    yc = pt + (h - pt - pb) * (i + .5) / len(rows)
    y0 = yc - bh / 2
    out.append(f'<text class="tick" x="{pl - 8}" y="{yc + 4:.1f}" text-anchor="end">{lab}</text>')
    out.append(f'<rect x="{X(0):.1f}" y="{y0:.1f}" width="{max(X(a) - X(0) - 1, 1):.1f}" height="{bh:.1f}" rx="2" '
               f'fill="var(--s2)" class="pt" data-tip="{lab} | oracle gap (information lost): {a:.4f}"/>')
    out.append(f'<rect x="{X(a) + 1:.1f}" y="{y0:.1f}" width="{max(X(a + b) - X(a) - 1, 1):.1f}" height="{bh:.1f}" rx="2" '
               f'fill="var(--s1)" class="pt" data-tip="{lab} | excess (model failure): {b:.4f}"/>')
    out.append(f'<text class="dlabel" x="{X(a + b) + 6:.1f}" y="{yc + 4:.1f}">{100 * b / (a + b):.0f}% excess</text>')
  out.append(f'<text class="axis" x="{(pl + w - pr) / 2}" y="{h - 2}" text-anchor="middle">'
             'block minus scatter, MSE (raw units squared)</text></svg></div>')
  out.append('<div class="legend"><span><i style="background:var(--s2)"></i>Oracle gap: information lost</span>'
             '<span><i style="background:var(--s1)"></i>Excess: GRU gap above the oracle gap</span></div></figure>')
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
  e2_charts += line_chart(f'e2-{sensors}', f'{lab}: extra MSE, mean of 3 seeds', offs,
                          ['0', '4', '8', '16'], series,
                          ('steps between outage end and forecast origin', 'extra MSE'), zero=True, w=440, h=260)

bins = [x['bin'] for x in E4[940]['kalman_sensitivity']]
def sens(s, arm):
  return [x['value'] for x in E4[s]['arms'][arm]['sensitivity_s0']]
def sens_big(s):
  a = [a for a in E4B[s]['arms'] if a['train'] == 'block' and a['epochs'] == 48 and a['hidden'] == 32][0]
  return [x['value'] for x in a['sensitivity_s0']]
sens_series = [
    ('kalman', NAME['kalman'], list(np.mean([[x['value'] for x in E4[s]['kalman_sensitivity']] for s in SEEDS], 0))),
    ('gru', NAME['gru'], list(np.mean([sens(s, 'scatter/zero') for s in SEEDS], 0))),
    ('gru_block', NAME['gru_block'], list(np.mean([sens(s, 'block/zero') for s in SEEDS], 0))),
    ('gru_big', 'GRU, block, 4x steps, 32 units', list(np.mean([sens_big(s) for s in SEEDS], 0))),
]
sens_chart = line_chart('sens', 'How much the forecast moves when the last sensor-0 reading moves by 1, mean of 3 seeds',
                        list(range(len(bins))), bins, sens_series,
                        ('sensor-0 age at the forecast origin (steps)', 'd forecast / d last reading'), ymin=0, w=620, h=320)

def paired_bins(s, key):
  return [x['delta'] for x in E1[s]['paired'][key]['by_block_age']]
paired_series = [(k, NAME[k], list(np.mean([paired_bins(s, src) for s in SEEDS], 0)))
                 for k, src in [('gru', 'gru'), ('kalman', 'kalman'), ('persistence', 'persistence')]]
paired_chart = line_chart('paired', 'Same window, block minus scatter MSE, by sensor-0 age under the block mask (mean of 3 seeds)',
                          list(range(len(bins))), bins, paired_series,
                          ('sensor-0 age at the origin under the block mask (steps)', 'extra MSE'), zero=True, w=620, h=300)

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


def rng(vals, d=3):
  return f'{min(vals):.{d}f} to {max(vals):.{d}f}'


html = (ROOT / 'report' / 'template.html').read_text()
subs = {
    'EXPOSURE_ROWS': exposure_table(), 'BASELINE_ROWS': baseline_table(), 'FRESH_STALE_ROWS': fresh_stale_table(),
    'E2_ROWS': e2_table(), 'TRAIN_ROWS': training_table(), 'E4B_ROWS': e4b_table(), 'FACTORIAL_ROWS': factorial_table(),
    'GAP_CHART': stacked_bars('gap', 'Block minus scatter MSE for the scatter-trained GRU, split into oracle gap and excess', gap_rows),
    'E2_CHARTS': e2_charts, 'SENS_CHART': sens_chart, 'PAIRED_CHART': paired_chart,
    'EXCESS_RANGE': f'{min(excess_pct):.0f}% to {max(excess_pct):.0f}%',
    'ORACLE_RANGE': f'{min(oracle_share):.0f}% to {max(oracle_share):.0f}%',
    'OTHERS_GRU': rng(others_gru), 'OTHERS_KF': rng(others_kf), 'S0_GRU': rng(s0_gru), 'S0_KF': rng(s0_kf),
    'AGE_VS_ZERO': f'{age_vs_zero:.4f}',
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
