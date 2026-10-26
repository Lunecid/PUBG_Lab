"""Leak-free phase-tier inputs and labels (dump v2).

``source``    raw-telemetry container on the LogMatchStart clock W + read-only DB fetch (``cut_w`` truncates INSIDE the SQL)
``phases``    phase boundaries ``b_k`` from LogPhaseChange events, cut ``a_k = b_k - 10 s``
``alive``     causal alive status (kill events + revive evidence), final team elimination time T, revive audit
``grid``      10 s team grid on W (a grid point is admissible iff every sample it may read precedes the cut)
``events``    team-attributed combat events, combat edges, loot counters
``features``  all 150 features of one cut from records strictly before the cut (``compute_cut_features``)
``labels``    landmark risk set, hazard label, exclusions of truncated logs
``assemble``  one match -> the dump record
``replay``    the prefix-replay test (full source vs database-truncated source)

Writer: ``python -m model_stgcn.dump_phase_tier_v2``; record contract: ``model_stgcn/dumps/bigcomp2027_phase_v2/CONTRACT.md``.
"""
