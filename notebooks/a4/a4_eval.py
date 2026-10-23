"""Evaluation helpers for lab course A4 (exchange bias and He-ion bombardment).

Students measure L-MOKE hysteresis loops on a grid of points, bombard the
points with helium ions of different doses, and measure again. This module
reads the L-MOKE files, evaluates every loop with evaluix (arctan fit) and
pairs the loops before and after the bombardment per grid point.

All field values are in kA/m, as written by the L-MOKE setup.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evaluix.utils.hysteresis_functions import (
    arctan_hyseval,
    hys_prep,
    lin_hyseval,
)

# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

HEADER_KEYS = {
    'User': 'user',
    'Sample': 'sample',
    'State': 'state',
    'Profile': 'profile',
    'Hmax (kA/m)': 'hmax',
    'pts./branch': 'pts_per_branch',
    'nCycles': 'n_cycles',
}

# Point labels the groups used so far: "Ber.Ny. 07/19 AP 9e14" -> "07",
# "B1 HEB0" -> "B1". Tried in this order; the first match wins.
POINT_PATTERNS = (
    r'(\d{1,3})\s*/\s*\d{1,3}',  # 07/19
    r'\b([A-Z]\d{1,2})\b',  # B1, C5
    r'\b[Pp](?:oint)?\s*(\d{1,3})\b',  # P7, Point 7
)
# Ion dose in the comment: "AP 9e14", "dose 1.5e15", "1e16 ions/cm2"
DOSE_PATTERN = r'(?<![\w.])(\d+(?:\.\d+)?[eE][+-]?\d+)(?![\w.])'


@dataclass
class Loop:
    file: str
    sample: str
    state: str
    comment: str
    time: pd.Timestamp
    H: np.ndarray  # kA/m
    I: np.ndarray  # arb. u.
    hmax: float


def read_lmoke(path: str | Path) -> Loop:
    """Read one L-MOKE .txt file (header lines start with '#')."""
    path = Path(path)
    header, comment, rows = {}, [], []
    in_comment = False
    with open(path, encoding='utf-8', errors='replace') as f:
        for raw in f:
            line = raw.rstrip('\r\n')
            if line.startswith('#'):
                text = line[1:].strip()
                if text.startswith('Comment'):
                    in_comment = True
                    rest = text[len('Comment') :].strip()
                    if rest:
                        comment.append(rest)
                    continue
                if in_comment:
                    if not text or text.startswith('Meas. type'):
                        in_comment = False
                    else:
                        comment.append(text)
                        continue
                parts = [p for p in re.split(r'\t+', text) if p.strip()]
                if len(parts) >= 2 and parts[0].strip() in HEADER_KEYS:
                    header[HEADER_KEYS[parts[0].strip()]] = parts[1].strip()
            elif line.strip():
                rows.append(line.split())
    data = np.asarray(rows, dtype=float)
    stamp = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', path.name)
    time = (
        pd.to_datetime(stamp.group(1), format='%Y-%m-%d_%H-%M-%S')
        if stamp
        else pd.NaT
    )
    return Loop(
        file=path.name,
        sample=header.get('sample', ''),
        state=header.get('state', ''),
        comment=' '.join(comment).strip(),
        time=time,
        H=data[:, 0],
        I=data[:, 1],
        hmax=float(header.get('hmax', 'nan')),
    )


def find_point(comment: str, patterns=POINT_PATTERNS) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, comment)
        if match:
            return match.group(1)
    return None


def find_dose(comment: str) -> float | None:
    match = re.search(DOSE_PATTERN, comment)
    return float(match.group(1)) if match else None


def load_folder(
    folder: str | Path = '.',
    pattern: str = '*LMOKE*.txt',
    after_states=('after IB',),
    after_since: str | None = None,
) -> pd.DataFrame:
    """Read all L-MOKE files of a folder into a table (one row per file).

    phase is 'after' when the file's State is one of after_states
    (case-insensitive), otherwise 'before'. If the State was not set
    correctly, give after_since (e.g. '2026-08-19 12:00'): every file
    measured later counts as 'after'.
    """
    after = {s.lower() for s in after_states}
    since = pd.Timestamp(after_since) if after_since else None
    rows, loops = [], {}
    for path in sorted(Path(folder).glob(pattern)):
        try:
            loop = read_lmoke(path)
        except Exception as error:  # broken file: report, keep going
            warnings.warn(f'{path.name}: could not be read ({error})')
            continue
        loops[loop.file] = loop
        rows.append(
            {
                'file': loop.file,
                'time': loop.time,
                'sample': loop.sample,
                'state': loop.state,
                'phase': (
                    'after'
                    if (since is not None and loop.time >= since)
                    or (since is None and loop.state.lower() in after)
                    else 'before'
                ),
                'comment': loop.comment,
                'point': find_point(loop.comment),
                'dose_in_file': find_dose(loop.comment),
                'control': 'control' in loop.comment.lower(),
                'hmax': loop.hmax,
            }
        )
    table = pd.DataFrame(rows).sort_values('time').reset_index(drop=True)
    table['dose_in_file'] = table['dose_in_file'].astype(float)
    table.attrs['loops'] = loops
    return table


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

RESULT_COLUMNS = ['HEB', 'dHEB', 'HC', 'dHC', 'MR', 'MS', 'r_squared']


def _scalar(value):
    """evaluix returns some values as tuples (mean, branch 1, branch 2)."""
    if isinstance(value, (tuple, list, np.ndarray)):
        return float(value[0]) if len(value) else np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def evaluate_loop(loop: Loop, method: str = 'arctan', prep: bool = True) -> dict:
    """Evaluate one loop with evaluix. method: 'arctan' (fit) or 'linear'."""
    H, I = loop.H, loop.I
    if prep:
        H, I = hys_prep(H, I)
    if method == 'arctan':
        fitted, params, _ = arctan_hyseval(H, I)
    elif method == 'linear':
        fitted, params = None, lin_hyseval(H, I)
    else:
        raise ValueError("method must be 'arctan' or 'linear'")
    # arctan_hyseval names the values of its (single) loop HEB_1, HC_1, ...
    params = {
        (k[:-2] if k.endswith('_1') else k): v for k, v in params.items()
    }
    result = {key: _scalar(params.get(key, np.nan)) for key in RESULT_COLUMNS}
    if 'saturation_fields' in params:
        desc, asc = params['saturation_fields']
        result['Hsat_desc'], result['Hsat_asc'] = float(desc), float(asc)
    if 'rectangularity' in params:
        result['rectangularity'] = _scalar(params['rectangularity'])
    result['_H'], result['_I'] = np.asarray(H), np.asarray(I)
    if fitted is not None:
        result['_fit_H'] = np.asarray(fitted['xdata'])
        result['_fit_I'] = np.asarray(fitted['ydata'])
    return result


def evaluate(table: pd.DataFrame, method: str = 'arctan', prep: bool = True):
    """Evaluate every file of the table. Returns (results, curves).

    Loops that cannot be evaluated get an 'error' text instead of values.
    """
    loops = table.attrs['loops']
    rows, curves = [], {}
    for _, row in table.iterrows():
        out = {'file': row['file']}
        try:
            result = evaluate_loop(loops[row['file']], method=method, prep=prep)
            curves[row['file']] = {
                k[1:]: v for k, v in result.items() if k.startswith('_')
            }
            out.update({k: v for k, v in result.items() if not k.startswith('_')})
            out['error'] = ''
        except Exception as error:
            out['error'] = str(error)
        rows.append(out)
    results = table.drop(columns=[], errors='ignore').merge(
        pd.DataFrame(rows), on='file'
    )
    results.attrs = table.attrs
    return results, curves


def compare(
    results: pd.DataFrame,
    doses: dict | None = None,
    repeats: str = 'last',
    include_control: bool = False,
) -> pd.DataFrame:
    """One row per grid point with the values before and after bombardment.

    doses: {point: dose in ions/cm^2}. Overrides doses written in the file
    comments. repeats: 'last' uses the newest loop of a point and phase,
    'mean' averages repeated loops (dHEB becomes the larger of the standard
    deviation and the mean fit uncertainty).
    """
    usable = results[(results['error'] == '') & results['point'].notna()]
    if not include_control:
        usable = usable[~usable['control']]
    parts = {}
    for phase in ('before', 'after'):
        sub = usable[usable['phase'] == phase].sort_values('time')
        if repeats == 'last':
            agg = sub.groupby('point').last(numeric_only=False)
            agg['n'] = sub.groupby('point').size()
        elif repeats == 'mean':
            grouped = sub.groupby('point')
            agg = grouped[['HEB', 'HC', 'MR', 'MS', 'r_squared']].mean()
            spread = grouped['HEB'].std().fillna(0)
            agg['dHEB'] = np.maximum(spread, grouped['dHEB'].mean())
            agg['dHC'] = np.maximum(
                grouped['HC'].std().fillna(0), grouped['dHC'].mean()
            )
            agg['file'] = grouped['file'].apply(', '.join)
            agg['dose_in_file'] = grouped['dose_in_file'].last()
            agg['n'] = grouped.size()
        else:
            raise ValueError("repeats must be 'last' or 'mean'")
        parts[phase] = agg[
            ['HEB', 'dHEB', 'HC', 'dHC', 'MR', 'r_squared', 'file', 'n', 'dose_in_file']
        ]
    table = parts['before'].join(
        parts['after'], how='outer', lsuffix='_before', rsuffix='_after'
    )
    dose = table['dose_in_file_after'].combine_first(table['dose_in_file_before'])
    if doses:
        given = pd.Series({str(k): float(v) for k, v in doses.items()})
        dose = given.reindex(table.index).combine_first(dose)
    table.insert(0, 'dose', dose)
    table = table.drop(columns=['dose_in_file_before', 'dose_in_file_after'])
    table['dHEB_shift'] = table['HEB_after'] - table['HEB_before']
    table['d_dHEB_shift'] = np.hypot(table['dHEB_before'], table['dHEB_after'])
    table['HEB_ratio'] = table['HEB_after'] / table['HEB_before']
    table['dHC_shift'] = table['HC_after'] - table['HC_before']
    table.index.name = 'point'
    return table.sort_values(['dose', 'point'], na_position='last')


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def plot_point(results, curves, point, ax=None):
    """Normalised loops before/after of one grid point with their fits."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    sub = results[(results['point'] == point) & (results['error'] == '')]
    for _, row in sub.sort_values('time').iterrows():
        c = curves.get(row['file'])
        if not c:
            continue
        color = 'tab:blue' if row['phase'] == 'before' else 'tab:red'
        label = f"{row['phase']} {row['time']:%d.%m. %H:%M}"
        ax.plot(c['H'], c['I'], '.', ms=2, color=color, alpha=0.5)
        if 'fit_H' in c:
            ax.plot(c['fit_H'], c['fit_I'], '-', color=color, label=label)
        ax.axvline(row['HEB'], color=color, ls=':', lw=1)
    ax.set_xlabel('H (kA/m)')
    ax.set_ylabel('normalised Kerr signal')
    ax.set_title(f'Point {point}')
    ax.legend(fontsize=7)
    return ax


def plot_vs_dose(comparison, quantity='HEB', ax=None):
    """quantity before and after against ion dose (log axis)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    data = comparison.dropna(subset=['dose'])
    for phase, color, marker in (('before', 'tab:blue', 'o'), ('after', 'tab:red', 's')):
        ax.errorbar(
            data['dose'],
            data[f'{quantity}_{phase}'],
            yerr=data[f'd{quantity}_{phase}'],
            fmt=marker,
            color=color,
            capsize=3,
            label=f'{quantity} {phase}',
        )
    ax.set_xscale('log')
    ax.set_xlabel('He$^+$ dose (ions/cm$^2$)')
    ax.set_ylabel(f'{quantity} (kA/m)')
    ax.axhline(0, color='grey', lw=0.5)
    ax.legend()
    return ax


def plot_by_point(comparison, quantity='HEB', ax=None):
    """quantity before and after per grid point (when no doses are known)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 0.35 * len(comparison)), 4))
    x = np.arange(len(comparison))
    for shift, phase, color in ((-0.18, 'before', 'tab:blue'), (0.18, 'after', 'tab:red')):
        ax.errorbar(
            x + shift,
            comparison[f'{quantity}_{phase}'],
            yerr=comparison[f'd{quantity}_{phase}'],
            fmt='o',
            color=color,
            capsize=3,
            label=f'{quantity} {phase}',
        )
    ax.set_xticks(x, comparison.index, rotation=90)
    ax.set_xlabel('grid point')
    ax.set_ylabel(f'{quantity} (kA/m)')
    ax.axhline(0, color='grey', lw=0.5)
    ax.legend()
    return ax


def plot_shift_by_point(comparison, ax=None):
    """Change of HEB (after - before) per grid point (when no doses are known)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 0.35 * len(comparison)), 4))
    x = np.arange(len(comparison))
    ax.errorbar(x, comparison['dHEB_shift'], yerr=comparison['d_dHEB_shift'],
                fmt='o', color='k', capsize=3)
    ax.set_xticks(x, comparison.index, rotation=90)
    ax.set_xlabel('grid point')
    ax.set_ylabel(r'$\Delta H_\mathrm{EB}$ = after $-$ before (kA/m)')
    ax.axhline(0, color='grey', lw=0.5)
    return ax


def plot_all_points(results, curves, points=None, ncols=5):
    """Small loop plot (before/after) for every grid point."""
    import matplotlib.pyplot as plt

    if points is None:
        points = sorted(results['point'].dropna().unique())
    nrows = int(np.ceil(len(points) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows),
                             squeeze=False, sharex=True, sharey=True)
    for ax, point in zip(axes.flat, points):
        plot_point(results, curves, point, ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.legend().set_visible(False)
    for ax in list(axes.flat)[len(points):]:
        ax.set_visible(False)
    fig.supxlabel('H (kA/m)')
    fig.supylabel('normalised Kerr signal')
    fig.tight_layout()
    return fig


def check_fits(results, r2_min=0.99):
    """Loops whose single-arctan fit is poor (e.g. a double loop when only part
    of the measured spot was bombarded, or a noisy measurement)."""
    bad = results[(results['error'] != '') | (results['r_squared'] < r2_min)]
    return bad[['file', 'phase', 'point', 'comment', 'r_squared', 'HEB', 'error']]


def export(results, comparison, folder='A4_results', figures=None):
    """Write the tables (CSV + Excel) and figures into folder."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    per_file = results.drop(columns=[c for c in results.columns if c.startswith('_')])
    per_file.to_csv(folder / 'loops.csv', index=False)
    comparison.to_csv(folder / 'comparison.csv')
    with pd.ExcelWriter(folder / 'A4_results.xlsx') as writer:
        comparison.to_excel(writer, sheet_name='comparison')
        per_file.to_excel(writer, sheet_name='loops', index=False)
    for name, fig in (figures or {}).items():
        fig.savefig(folder / f'{name}.png', dpi=150, bbox_inches='tight')
        fig.savefig(folder / f'{name}.pdf', bbox_inches='tight')
    return sorted(p.name for p in folder.iterdir())


def plot_shift_vs_dose(comparison, ax=None):
    """Change of HEB (after - before) against ion dose."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    data = comparison.dropna(subset=['dose', 'dHEB_shift'])
    ax.errorbar(
        data['dose'],
        data['dHEB_shift'],
        yerr=data['d_dHEB_shift'],
        fmt='o',
        color='k',
        capsize=3,
    )
    for point, row in data.iterrows():
        ax.annotate(point, (row['dose'], row['dHEB_shift']), fontsize=7,
                    xytext=(3, 3), textcoords='offset points')
    ax.set_xscale('log')
    ax.set_xlabel('He$^+$ dose (ions/cm$^2$)')
    ax.set_ylabel(r'$\Delta H_\mathrm{EB}$ = after $-$ before (kA/m)')
    ax.axhline(0, color='grey', lw=0.5)
    return ax
