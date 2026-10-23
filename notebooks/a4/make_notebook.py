"""Builds A4_exchange_bias.ipynb (run: python make_notebook.py)."""

import json

cells = []


def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n')})


def code(text):
    cells.append(
        {
            'cell_type': 'code',
            'metadata': {},
            'execution_count': None,
            'outputs': [],
            'source': text.strip('\n'),
        }
    )


md(r"""
# A4 – Exchange bias before and after He-ion bombardment

This notebook evaluates your L-MOKE hysteresis loops with **evaluix** (arctan fit) and
compares the exchange bias field $H_\mathrm{EB}$ and the coercive field $H_\mathrm{C}$
of every grid point **before** and **after** the ion bombardment, as a function of the ion dose.

**How to use it**
1. Keep this notebook and `a4_eval.py` in the folder with your L-MOKE files (`*_LMOKE_*.txt`).
2. Check the settings in the next cell, then run everything: *Run → Run All Cells*.
3. Results (tables, figures) are written to the folder `A4_results`.

**What the files must contain** (set in the L-MOKE software when measuring)
- `State`: `after IB` for the measurements after the bombardment, anything else before
  (e.g. `after FC`, `as made`).
- `Comment`: the grid point, e.g. `B3` or `07/19`; after the bombardment optionally the dose,
  e.g. `07/19 AP 9e14` (ions/cm²). Files without a point label (e.g. `test`) are skipped.

All fields are in kA/m. $H_\mathrm{EB}$ is the loop centre, $H_\mathrm{C}$ half the loop width.
""")

code(r"""
# ---------------- Settings ----------------
FOLDER = '.'               # folder with the L-MOKE files ('.' = folder of this notebook)
AFTER_STATES = ['after IB']  # File 'State' values that mean: measured after bombardment
AFTER_SINCE = None         # alternative: e.g. '2026-08-19 12:00' -> everything later is 'after'
REPEATS = 'last'           # point measured several times: 'last' = newest loop, 'mean' = average
METHOD = 'arctan'          # 'arctan' (evaluix fit) or 'linear' (zero crossings)
R2_MIN = 0.99              # fits below this are listed for checking

# Ion dose per grid point in ions/cm^2. Only needed if the dose is NOT in the file comments
# (doses given here win over the comments). Example: DOSES = {'B1': 1e14, 'B2': 2e14}
DOSES = {}
""")

code(r"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import a4_eval as a4
pd.set_option('display.max_rows', 200)
%matplotlib inline
""")

md(r"""
## 1. Files

One row per file: phase (before/after), grid point and the dose found in the comment.
Check this table first: wrong phase or point means a wrong `State`/`Comment` in the file.
""")

code(r"""
files = a4.load_folder(FOLDER, after_states=AFTER_STATES, after_since=AFTER_SINCE)
print(len(files), 'files:', files['phase'].value_counts().to_dict())
skipped = files[files['point'].isna()]
if len(skipped):
    print('No grid point in the comment, skipped:', ', '.join(skipped['file']))
files[['file', 'time', 'phase', 'point', 'dose_in_file', 'control', 'comment']]
""")

md(r"""
## 2. Evaluate every loop

`hys_prep` removes outliers and the linear background and normalises the loop, then the
loop is fitted with an arctan function. `dHEB`/`dHC` include the fit uncertainty and half the field step.
""")

code(r"""
results, curves = a4.evaluate(files, method=METHOD)
results[['file', 'phase', 'point', 'HEB', 'dHEB', 'HC', 'dHC', 'r_squared', 'error']].round(3)
""")

md(r"""
**Loops to check.** A low $R^2$ often means a *double loop*: only part of the laser spot sits on the
bombarded area, or the point is in the middle of the transition. Look at these loops in section 4
and decide whether to re-measure or exclude them.
""")

code(r"""
a4.check_fits(results, r2_min=R2_MIN).round(3)
""")

md(r"""
## 3. Before vs. after per grid point

`dHEB_shift` = $H_\mathrm{EB}^\mathrm{after} - H_\mathrm{EB}^\mathrm{before}$,
`HEB_ratio` = $H_\mathrm{EB}^\mathrm{after} / H_\mathrm{EB}^\mathrm{before}$, `n_…` = number of loops of that point.
""")

code(r"""
comparison = a4.compare(results, doses=DOSES, repeats=REPEATS)
missing = comparison[comparison['HEB_before'].isna() | comparison['HEB_after'].isna()].index.tolist()
if missing:
    print('Points without a loop before AND after:', missing)
if comparison['dose'].isna().all():
    print('No doses found: fill in DOSES in the settings cell to plot against the dose.')
comparison[['dose', 'HEB_before', 'dHEB_before', 'HEB_after', 'dHEB_after', 'dHEB_shift',
            'd_dHEB_shift', 'HEB_ratio', 'HC_before', 'HC_after', 'n_before', 'n_after']].round(3)
""")

md("## 4. Plots")

code(r"""
figures = {}
has_dose = comparison['dose'].notna().any()
fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
if has_dose:
    a4.plot_vs_dose(comparison, 'HEB', ax=axes[0])
    a4.plot_shift_vs_dose(comparison, ax=axes[1])
    a4.plot_vs_dose(comparison, 'HC', ax=axes[2])
else:
    a4.plot_by_point(comparison, 'HEB', ax=axes[0])
    a4.plot_shift_by_point(comparison, ax=axes[1])
    a4.plot_by_point(comparison, 'HC', ax=axes[2])
axes[0].set_title('Exchange bias'); axes[1].set_title('Change of the exchange bias'); axes[2].set_title('Coercive field')
fig.tight_layout()
figures['HEB_HC_vs_dose' if has_dose else 'HEB_HC_per_point'] = fig
""")

md("All loops (blue: before, red: after; dotted line: $H_\\mathrm{EB}$ of the fit).")

code(r"""
fig = a4.plot_all_points(results, curves)
figures['loops'] = fig
""")

md("A single point in detail (change the label):")

code(r"""
a4.plot_point(results, curves, comparison.index[0]);
""")

md("## 5. Save the results")

code(r"""
try:
    written = a4.export(results, comparison, 'A4_results', figures)
except PermissionError:
    written = a4.export(results, comparison, '/home/jovyan/work/A4_results', figures)
written
""")

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}
for i, c in enumerate(nb['cells']):
    c['id'] = f'a4-{i:02d}'
with open('A4_exchange_bias.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print('cells', len(cells))
