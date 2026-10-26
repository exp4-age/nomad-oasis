"""Builds particle_tracking.ipynb (run: python make_particle_tracking.py).

Port of code/tracking (main.py, filtering.py, velocities.py) for NORTH Jupyter.
Edit this file, not the notebook, then rebuild.
"""

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md("""# Particle tracking (trackpy)

Tracks particles in a video of an Oasis upload, filters the trajectories and
analyses their velocities. Same steps and defaults as `code/tracking`
(`main.py`, `filtering.py`, `velocities.py`).

**How to use**

1. Open NORTH Jupyter *from the upload* (upload page → *Jupyter*), so the upload
   is mounted under `~/uploads/`.
2. Run the cells from the top. Set the video and parameters in section 1.
3. Sections 2–3 are fast previews on a few frames: tune `DIAMETER`, `MINMASS`,
   `SEARCH_RANGE` there before the full run in section 4.
4. Results are always written to `~/work/tracking/` (kept between sessions).
   Only section 7 writes into the upload, and only if `SAVE_TO_UPLOAD = True`.

Copy this notebook to `~/work/` before editing it: changes outside `~/work`
are lost when the Jupyter server stops."""))

cells.append(code("""import json, os, re, time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trackpy as tp

tp.quiet()
UPLOADS = Path.home() / 'uploads'
OUT_DIR = Path.home() / 'work' / 'tracking'
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRACKS_SUFFIX, FILTERED_SUFFIX = '_trajs.h5', '_filtered_trajs.h5'


class H5Video:
    \"\"\"Frames of a rig video (.h5 with `frames`, `timestamps`, `frame_counters`).

    Reads in blocks of the HDF5 chunk size, so long videos never have to fit
    into memory.
    \"\"\"

    def __init__(self, path):
        self.path = Path(path)
        with h5py.File(self.path, 'r') as f:
            key = 'frames' if 'frames' in f else 'transport'
            if key not in f or f[key].ndim != 3:
                raise ValueError(f'{self.path.name}: no frames dataset')
            self.key, self.shape = key, f[key].shape
            self.block = (f[key].chunks or (32,))[0]
            self.attrs = dict(f.attrs)
            n = self.shape[0]
            self.timestamps = f['timestamps'][:] if 'timestamps' in f else None
            self.frame_counters = (
                f['frame_counters'][:] if 'frame_counters' in f else np.arange(n)
            )
        rate = float(self.attrs.get('AcquisitionFrameRate', 0) or 0)
        if self.timestamps is not None and n > 1:
            self.fps = 1 / np.median(np.diff(self.timestamps))
        else:
            self.fps = rate or 100.0
        if self.timestamps is None:
            self.timestamps = np.arange(n) / self.fps

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, i):
        with h5py.File(self.path, 'r') as f:
            return f[self.key][i]

    def blocks(self, start=0, stop=None):
        \"\"\"Yield (first frame number, array of frames).\"\"\"
        stop = len(self) if stop is None else min(stop, len(self))
        with h5py.File(self.path, 'r') as f:
            for a in range(start, stop, self.block):
                yield a, f[self.key][a : min(a + self.block, stop)]


def find_videos():
    videos = []
    for p in sorted(UPLOADS.glob('*/**/*.h5')):
        if p.name.endswith(TRACKS_SUFFIX):
            continue
        try:
            H5Video(p)
        except (OSError, ValueError, KeyError):
            continue
        videos.append(p)
    return videos


videos = find_videos()
for i, p in enumerate(videos):
    done = ' (has tracks)' if p.with_name(p.stem + TRACKS_SUFFIX).exists() else ''
    print(f'{i:3d}  {p.relative_to(UPLOADS)}{done}')
if not videos:
    print(f'No videos under {UPLOADS}. Start Jupyter from the upload page.')"""))

cells.append(md("""## 1. Video and parameters

Defaults are those of `code/tracking`. `INVERT = None` picks bright- or
darkfield automatically (whichever finds more particles in frame 0)."""))

cells.append(code("""VIDEO = videos[0]          # or Path('~/uploads/.../name.h5').expanduser()

# locate (tp.locate / tp.batch)
DIAMETER = 11              # px, odd; a bit larger than the particles
MINMASS = 1000             # minimum integrated brightness
INVERT = None              # None = auto, True = dark particles, False = bright

# link (tp.link_df)
SEARCH_RANGE = 10          # px a particle may move between frames
MEMORY = 5                 # frames a particle may disappear

# filter
MIN_FRAMES = 10            # tp.filter_stubs: drop tracks shorter than this
MIN_DISPLACEMENT_PX = 10   # drop tracks with |x_end - x_start| below this

# velocities
PX_TO_UM = 0.25            # µm per pixel (objective dependent!)
VELOCITY_WINDOW_S = 0.1    # averaging time of the velocity

# run
MAX_FRAMES = None          # None = whole video; e.g. 200 for a quick test
PROCESSES = 2              # CPU cores for locating; the server is shared

video = H5Video(VIDEO)
stem = VIDEO.name[: -len('.h5')]
print(f'{VIDEO.name}: {len(video)} frames of {video.shape[2]}x{video.shape[1]} px, '
      f'{video.fps:.1f} fps, {len(video) / video.fps:.1f} s')"""))

cells.append(md("## 2. Preview: locate in one frame"))

cells.append(code("""frame0 = video[0]
if INVERT is None:
    n_bright = tp.locate(frame0, DIAMETER, invert=False, minmass=MINMASS).x.nunique()
    n_dark = tp.locate(frame0, DIAMETER, invert=True, minmass=MINMASS).x.nunique()
    invert = n_dark >= n_bright
    print(f'auto: bright {n_bright}, dark {n_dark} -> invert={invert}')
else:
    invert = INVERT
f0 = tp.locate(frame0, DIAMETER, invert=invert, minmass=MINMASS)
print(f'{len(f0)} particles in frame 0')

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), gridspec_kw=dict(width_ratios=[2, 1, 1]))
ax[0].imshow(frame0, cmap='gray')
ax[0].scatter(f0.x, f0.y, s=40, edgecolor='red', facecolor='none', lw=0.8)
ax[0].set_title(f'frame 0: {len(f0)} particles')
ax[1].hist(f0.mass, bins=40)
ax[1].axvline(MINMASS, color='r', ls='--')
ax[1].set(title='mass (raise MINMASS to drop dim spots)', xlabel='mass')
ax[2].hist(np.modf(f0.x)[0], bins=20, alpha=0.6, label='x')
ax[2].hist(np.modf(f0.y)[0], bins=20, alpha=0.6, label='y')
ax[2].legend()
ax[2].set(title='subpixel bias (should be flat)', xlabel='decimal part of position')
plt.tight_layout()"""))

cells.append(md("""Zoom in to check single particles (change `x0, y0`):"""))

cells.append(code("""x0, y0, w = 900, 500, 200
sel = f0[(f0.x.between(x0, x0 + w)) & (f0.y.between(y0, y0 + w))]
plt.figure(figsize=(5, 5))
plt.imshow(frame0[y0:y0 + w, x0:x0 + w], cmap='gray', extent=(x0, x0 + w, y0 + w, y0))
plt.scatter(sel.x, sel.y, s=200, edgecolor='red', facecolor='none')
plt.title(f'{len(sel)} particles');"""))

cells.append(md("## 3. Preview: link the first frames"))

cells.append(code("""N_TEST = min(100, len(video))
t0 = time.time()
test = pd.concat(
    [tp.batch(list(b), DIAMETER, invert=invert, minmass=MINMASS).assign(
        frame=lambda d, a=a: d.frame + a)
     for a, b in video.blocks(0, N_TEST)],
    ignore_index=True)
test = tp.link_df(test, SEARCH_RANGE, memory=MEMORY)
dt = time.time() - t0
print(f'{N_TEST} frames in {dt:.0f} s -> full video ~{dt * len(video) / N_TEST / 60:.0f} min '
      f'with 1 process')
print(f'{test.particle.nunique()} tracks, median length '
      f'{test.groupby("particle").size().median():.0f} frames')
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(frame0, cmap='gray')
tp.plot_traj(test, ax=ax, label=False)
ax.set_title(f'tracks in the first {N_TEST} frames');"""))

cells.append(md("""## 4. Full run

Locates block by block, then links. Takes minutes for a 1000-frame video."""))

cells.append(code("""stop = len(video) if MAX_FRAMES is None else min(MAX_FRAMES, len(video))
parts, t0 = [], time.time()
for a, b in video.blocks(0, stop):
    f = tp.batch(list(b), DIAMETER, invert=invert, minmass=MINMASS, processes=PROCESSES)
    f['frame'] += a
    parts.append(f)
    done = a + len(b)
    if len(parts) % 5 == 0 or done == stop:
        el = time.time() - t0
        print(f'\\r{done}/{stop} frames, {el:.0f} s, ~{el / done * (stop - done):.0f} s left',
              end='', flush=True)
locs = pd.concat(parts, ignore_index=True)
print(f'\\nlocated {len(locs)} positions')

trajs = tp.link_df(locs, SEARCH_RANGE, memory=MEMORY)
trajs['timestamp'] = video.timestamps[trajs.frame.to_numpy()]
trajs['frame_counter'] = video.frame_counters[trajs.frame.to_numpy()]
print(f'{trajs.particle.nunique()} tracks')

fig, ax = plt.subplots(figsize=(10, 6))
tp.plot_traj(trajs, ax=ax, label=False)
ax.set_title(f'{stem}: all tracks')
fig.savefig(OUT_DIR / f'{stem}_trajs.png', dpi=150)"""))

cells.append(md("""## 5. Filter

As `filtering.py`: drop stubs shorter than `MIN_FRAMES`, then tracks whose
net x displacement is below `MIN_DISPLACEMENT_PX` (stuck particles)."""))

cells.append(code("""def filter_short(df, min_len):
    \"\"\"Tracks with |x_end - x_start| > min_len (evaluation.filter_short).\"\"\"
    g = df.groupby('particle').x
    keep = (g.last() - g.first()).abs() > min_len
    return df[df.particle.isin(keep[keep].index)]


stubs = tp.filter_stubs(trajs, MIN_FRAMES)
filtered = filter_short(stubs, MIN_DISPLACEMENT_PX)
for name, d in [('all', trajs), ('without stubs', stubs), ('moving', filtered)]:
    print(f'{name:14s} {d.particle.nunique():6d} tracks  {len(d):8d} positions')

fig, ax = plt.subplots(figsize=(10, 6))
tp.plot_traj(filtered, ax=ax, label=False)
ax.set_title(f'{stem}: filtered tracks')
fig.savefig(OUT_DIR / f'{stem}_filtered_trajs.png', dpi=150)"""))

cells.append(md("""## 6. Velocities

As `velocities.py`: x velocity over a window of `VELOCITY_WINDOW_S`, for the
longest track and averaged over all tracks per frame."""))

cells.append(code("""def add_velocity(df, fps, window_s=VELOCITY_WINDOW_S, px_to_um=PX_TO_UM):
    n = max(1, int(round(window_s * fps)))
    df = df.reset_index(drop=True).sort_values(['particle', 'frame']).reset_index(drop=True)
    g = df.groupby('particle')
    df['dx'] = g.x.diff(periods=n)
    df['dt'] = g.timestamp.diff(periods=n)
    df['velocity'] = df.dx * px_to_um / df.dt
    return df, n


def velocity_figure(df, fps, title, n):
    p = df.particle.value_counts().index[0]
    one = df[df.particle == p]
    mean_t = df.groupby('frame').velocity.mean()
    fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    ax[0].plot(one.frame / fps, one.x * PX_TO_UM, color='C0')
    ax[0].set(ylabel='x [µm]', title=f'x of particle {p} ({len(one)} frames)')
    for a, s, c, lab in [(ax[1], one.velocity, 'C1', f'particle {p}'),
                         (ax[2], mean_t, 'C2', f'mean of {df.particle.nunique()} tracks')]:
        x = (one.frame if a is ax[1] else mean_t.index) / fps
        a.plot(x, s.values, color=c, label=lab)
        a.axhline(s.mean(), color=c, ls='--', label=f'mean {s.mean():.2f} µm/s')
        for sign in (1, -1):
            a.axhline(s.mean() + sign * s.std(), color=c, ls=':', alpha=0.7,
                      label=f'±1σ {s.std():.2f} µm/s' if sign == 1 else None)
        a.set_ylabel('v [µm/s]')
        a.legend(fontsize=8)
    ax[1].set_title(f'velocity (averaged over {n / fps:.2f} s)')
    ax[2].set(xlabel='t [s]', title='mean velocity over all tracks')
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig, one.velocity, mean_t


vel, n = add_velocity(filtered, video.fps)
fig, v_one, v_mean = velocity_figure(vel, video.fps, f'{stem} (filtered)', n)
fig.savefig(OUT_DIR / f'{stem}_velocity_analysis.png', dpi=200)
summary = dict(recording=stem, tracks=int(vel.particle.nunique()),
               v_mean_um_s=float(v_mean.mean()), v_std_um_s=float(v_mean.std()),
               v_particle_um_s=float(v_one.mean()))
print(json.dumps(summary, indent=1))"""))

cells.append(md("""### Compare recordings (e.g. the frequency series)

Mean velocity of every recording in the upload folder that has filtered tracks
(`*_filtered_trajs.h5`) or tracks (`*_trajs.h5`, filtered here with the
parameters above). Names like `1to2`, `sqrt2to1` are also read as a ratio
for the x axis."""))

cells.append(code("""def ratio_from_name(name):
    m = re.fullmatch(r'(sqrt)?(\\d+)to(sqrt)?(\\d+)(?:_\\d+)?', name)
    if not m:
        return np.nan
    a = float(m[2]) ** (0.5 if m[1] else 1)
    b = float(m[4]) ** (0.5 if m[3] else 1)
    return a / b


rows = []
folder = VIDEO.parent
for tf in sorted(folder.glob('*' + TRACKS_SUFFIX)):
    if tf.name.endswith(FILTERED_SUFFIX):
        continue
    name = tf.name[: -len(TRACKS_SUFFIX)]
    ff = tf.with_name(name + FILTERED_SUFFIX)
    d = pd.read_hdf(ff if ff.exists() else tf, key='trajs')
    if not ff.exists():
        d = filter_short(tp.filter_stubs(d, MIN_FRAMES), MIN_DISPLACEMENT_PX)
    if 'timestamp' not in d:
        continue
    fps = 1 / np.median(np.diff(np.sort(d.timestamp.unique())))
    d, _ = add_velocity(d, fps)
    vt = d.groupby('frame').velocity.mean()
    rows.append(dict(recording=name, ratio=ratio_from_name(name), fps=round(fps, 1),
                     tracks=d.particle.nunique(), v_mean_um_s=vt.mean(), v_std_um_s=vt.std()))
    print(f'{name:20s} {rows[-1]["v_mean_um_s"]:8.2f} ± {rows[-1]["v_std_um_s"]:.2f} µm/s')

table = pd.DataFrame(rows).sort_values(['ratio', 'recording'])
table.to_csv(OUT_DIR / f'velocities_{folder.name}.csv', index=False)
if table.ratio.notna().any():
    t = table.dropna(subset=['ratio'])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(t.ratio, t.v_mean_um_s, t.v_std_um_s, fmt='o', capsize=3)
    for _, r in t.iterrows():
        ax.annotate(r.recording, (r.ratio, r.v_mean_um_s), fontsize=7,
                    xytext=(3, 3), textcoords='offset points')
    ax.set(xscale='log', xlabel='ratio from file name', ylabel='mean v [µm/s]')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f'velocities_{folder.name}.png', dpi=200)
table"""))

cells.append(md("""## 7. Save

Writes `{name}_trajs.h5` and `{name}_filtered_trajs.h5` (key `trajs`, the
video's attributes such as `recording_id` copied, tracking parameters in the
group `tracking_parameters`) to `~/work/tracking/`.

With `SAVE_TO_UPLOAD = True` they are also uploaded next to the video and
processed by the Oasis, so they show up as the recording's tracks entry.
Existing tracks are only replaced with `OVERWRITE = True`."""))

cells.append(code("""SAVE_TO_UPLOAD = False
OVERWRITE = False

params = dict(diameter=DIAMETER, minmass=MINMASS, invert=bool(invert),
              search_range=SEARCH_RANGE, memory=MEMORY, min_frames=MIN_FRAMES,
              min_displacement_px=MIN_DISPLACEMENT_PX, trackpy=tp.__version__,
              frames=int(stop), video=VIDEO.name,
              created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              user=os.environ.get('NOMAD_CLIENT_USER', ''))


def write_tracks(df, path):
    df.to_hdf(path, key='trajs', mode='w', complevel=4, complib='zlib')
    with h5py.File(path, 'a') as f:
        for k, v in video.attrs.items():
            f.attrs[k] = v
        g = f.create_group('tracking_parameters')
        for k, v in params.items():
            g.attrs[k] = v


local = {}
for suffix, df in [(TRACKS_SUFFIX, trajs), (FILTERED_SUFFIX, filtered)]:
    local[suffix] = OUT_DIR / f'{stem}{suffix}'
    write_tracks(df, local[suffix])
    print('saved', local[suffix])"""))

cells.append(code("""def upload_location(path):
    \"\"\"(upload_id, directory in the upload) of a file under ~/uploads.\"\"\"
    rel = path.relative_to(UPLOADS)
    folder = rel.parts[0]  # '{slugified name}-{upload_id}' or '{upload_id}'
    upload_id = folder[-22:] if len(folder) > 22 and folder[-23] == '-' else folder
    return upload_id, '/'.join(rel.parts[1:-1])


if SAVE_TO_UPLOAD:
    import requests

    upload_id, directory = upload_location(VIDEO)
    existing = [VIDEO.with_name(p.name) for p in local.values()
                if VIDEO.with_name(p.name).exists()]
    if existing and not OVERWRITE:
        raise SystemExit(f'Not saved: {[p.name for p in existing]} already in the upload. '
                         'Set OVERWRITE = True to replace them.')
    api = os.environ['NOMAD_CLIENT_URL'].rstrip('/') + '/v1'
    auth = {'Authorization': f'Bearer {os.environ["NOMAD_CLIENT_ACCESS_TOKEN"]}'}
    for p in local.values():
        with open(p, 'rb') as fh:
            r = requests.put(f'{api}/uploads/{upload_id}/raw/{directory}',
                             params=dict(file_name=p.name), data=fh, headers=auth)
        if r.status_code == 401:
            raise SystemExit('Login expired: restart Jupyter from the upload page '
                             '(stop it in the NORTH tool list) and run this cell again.')
        r.raise_for_status()
        print('uploaded', p.name, '-> processing in the Oasis')
    gui = re.sub(r'(/backend)?/api/v1$', '', api)
    print(f'Check the upload page: {gui}/gui/user/uploads/upload/id/{upload_id}')
else:
    print('SAVE_TO_UPLOAD is False: files are only in', OUT_DIR)"""))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, 'particle_tracking.ipynb')
print('wrote particle_tracking.ipynb')
