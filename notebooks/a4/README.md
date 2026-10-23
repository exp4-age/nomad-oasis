# A4 – exchange bias vs. He-ion dose

`A4_exchange_bias.ipynb` + `a4_eval.py` evaluate L-MOKE loops with evaluix (arctan fit) and compare
HEB/HC per grid point before and after ion bombardment. Main result: HEB/HEB0 (after/before) vs. dose.

Use on the Oasis: open the upload with the L-MOKE files in NORTH (Jupyter tool), copy both files into
the upload folder, run all cells. Results go to `A4_results/` (CSV, Excel, PNG/PDF).

File conventions: `State` = `after IB` after bombardment; `Comment` = grid point (`B3`, `07/19`).
Doses: the first run writes `doses.csv` (one line per grid point found); students fill it from their
notes (0 = not bombarded, shown as reference band) and rerun. A dose in the comment (`9e14`) is used
when `doses.csv` has none.

`make_notebook.py` regenerates the notebook. Tested with the jupyter image (evaluix 0.9.4.3) on the
2025 (47/50 files, doses in comments) and 2026 (47 files, B1–F4 grid) course data.
