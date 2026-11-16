---
description: Publication-quality color scheme and plotting conventions. Applies to all matplotlib/seaborn graphics.
paths: ["**/*.py", "**/*.ipynb"]
---

# Visualization

## Color Palette

Use this palette for all plots and figures. The colors follow Nature/Science publication conventions.

### Primary Palette (ordered by usage priority)

```python
PALETTE = [
    '#EB6692',  # rose — primary accent, baselines, first category
    '#7BC67B',  # green — improvements, positive results, second category
    '#32ABA2',  # teal — third category
    '#9E66C4',  # purple — fourth category
    '#E8B830',  # gold — fifth category
    '#3B7DD8',  # blue — sixth category
    '#E87D3E',  # orange — seventh category
    '#888888',  # gray — reference, neutral, control
]

PALETTE_DICT = {
    'rose':   '#EB6692',
    'green':  '#7BC67B',
    'teal':   '#32ABA2',
    'purple': '#9E66C4',
    'gold':   '#E8B830',
    'blue':   '#3B7DD8',
    'orange': '#E87D3E',
    'gray':   '#888888',
}
```

### Semantic Color Assignments

| Semantic role           | Color    | Hex       |
|------------------------|----------|-----------|
| Baseline / before      | rose     | `#EB6692` |
| Improvement / after    | green    | `#7BC67B` |
| Secondary comparison   | teal     | `#32ABA2` |
| Categorical accent     | purple   | `#9E66C4` |
| Highlight / warning    | gold     | `#E8B830` |
| Neutral / reference    | gray     | `#888888` |

### Applying the Palette

```python
import matplotlib.pyplot as plt
import matplotlib as mpl

PALETTE = ['#EB6692', '#7BC67B', '#32ABA2', '#9E66C4', '#E8B830', '#3B7DD8', '#E87D3E', '#888888']

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=PALETTE)
```

With seaborn:

```python
import seaborn as sns
sns.set_palette(PALETTE)
```

## Plotting Defaults

Apply these matplotlib rcParams for publication-quality figures:

```python
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'figure.figsize': (4, 3),
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})
```

## Quick Reference

- **2 groups** (e.g., before/after): use rose + green
- **3 groups**: rose + green + teal
- **Scatter with 2 series**: rose + teal (highest contrast pair)
- **Sequential/gradient**: use a single palette color and vary lightness, or use `plt.cm.RdYlGn`
- **Bar charts**: use filled bars with 70% alpha, black edge lines (0.5pt)
- **Box/violin plots**: fill with palette colors at 50% alpha, show individual data points
- **Error bars/bands**: use lighter tint (alpha ~0.2) of the same color
