"""External baseline: Sagong, Woo & Kim (2025, Entertainment Computing 55, 101040) re-implemented on our data.

Exploratory, added after the primary BigComp 2027 analysis (design doc section 4.3).  Not part of the PTR
framework: nothing here edits ``model_stgcn/ptr``.  Stages (``python -m model_stgcn.sagong2025 <stage>``):

    plan      dumps (torch) -> unit / phase table with the exact PTR rows and phase windows (no DB)
    extract   DB (read-only SELECT) -> the paper's 8 in-game features per unit and phase, causal to the cut a_k = phase_end - 10 s
              (v2 dump: window [s_k, a_k), records strictly before the cut, clock computed by the database)
    habcols   dumps (torch) -> the 44 zone / habitat columns (what noHab removes) per risk-set row, for ``fit --models hgb-hab``
    fit       types, Eq. 4 survival rate, XGBoost / RandomForest / LogisticRegression (+ the HGB controls), CSV + metrics.json

Run with an interpreter that has xgboost and pyarrow (the pyenv 3.11 of the repo has neither; ``/usr/local/bin/python3``
does).  Defaults: dumps ``model_stgcn/dumps/bigcomp2027_phase_v2``, cache ``model_stgcn/dumps/bigcomp2027_sagong_v2``, runs
``model_stgcn/checkpoints/bigcomp2027_ptr_v2``.  ``extract`` only reads the DB (read-only session); ``--workers`` <= 2 and
``--n-jobs`` <= 2 are enforced.  Sensitivity runs (``fit --scope per_phase`` / ``--sr-mode label``) need another ``--out-dir``.

The dump version (``meta['version']``) is recorded in ``plan_audit.json`` / ``features_meta.json`` / ``hab44_audit.json`` and
every stage refuses to combine artefacts of different versions (a v1 cache next to a v2 plan raises ``DumpVersionError``).
v2 semantics (task rules 1-4): the PTR rows, labels and censoring come from the dump (no winner override, no re-derived death
times); "alive" = the dump's causal ``node_mask`` at the cut, the comeback rows (risk set, causally dead) are units too; every
indicator reads records strictly before ``a_k``; the types and the Eq. 4 survival-rate table are fitted on the TRAIN split only.
"""
