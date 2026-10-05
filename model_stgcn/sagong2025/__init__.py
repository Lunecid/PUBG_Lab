"""External baseline: Sagong, Woo & Kim (2025, Entertainment Computing 55, 101040) re-implemented on our data.

Exploratory, added after the primary BigComp 2027 analysis (design doc section 4.3).  Not part of the PTR
framework: nothing here edits ``model_stgcn/ptr``.  Stages (``python -m model_stgcn.sagong2025 <stage>``):

    plan      dumps (torch) -> unit / phase table with the exact PTR rows and phase windows (no DB)
    extract   DB (read-only SELECT) -> the paper's 8 in-game features per unit and phase, causal to phase_end - 10 s
    fit       types, Eq. 4 survival rate, XGBoost / RandomForest / LogisticRegression, CSV + metrics.json

Run with an interpreter that has xgboost and pyarrow (the pyenv 3.11 of the repo has neither; ``/usr/local/bin/python3``
does).  Defaults: dumps ``model_stgcn/dumps/bigcomp2027_phase_v1``, cache ``model_stgcn/dumps/bigcomp2027_sagong_v1``, runs
``model_stgcn/checkpoints/bigcomp2027_ptr_v1``.  ``extract`` only reads the DB (read-only session); ``--workers`` <= 2 and
``--n-jobs`` <= 2 are enforced.  Sensitivity runs (``fit --scope per_phase`` / ``--sr-mode label``) need another ``--out-dir``.
"""
