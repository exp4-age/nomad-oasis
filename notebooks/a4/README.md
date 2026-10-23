# A4 – exchange bias vs. He-ion dose

`A4_exchange_bias.ipynb` + `a4_eval.py` evaluate L-MOKE loops with evaluix (arctan fit) and compare
HEB/HC per grid point before and after ion bombardment.

Use on the Oasis: open the upload with the L-MOKE files in NORTH (Jupyter tool), copy both files into
the upload folder, run all cells. Results go to `A4_results/` (CSV, Excel, PNG/PDF).

File conventions: `State` = `after IB` after bombardment; `Comment` = grid point (`B3`, `07/19`),
optionally the dose (`9e14`). Doses can also be entered in the notebook (`DOSES`).

`make_notebook.py` regenerates the notebook. Tested with the jupyter image (evaluix 0.9.4.3) on the
2025 (47/50 files, doses in comments) and 2026 (47 files, B1–F4 grid) course data.
