# V3 Pilot Results 2026-06-15

These are sanity/pilot experiments for the final factorized experiment stack,
not final paper numbers.

## Protocol

| Item | Value |
| --- | --- |
| Dataset | `squad`, first 120 eligible matches |
| Split | match-level 96 train / 12 val / 12 test |
| Training | 2 epochs, batch 4 |
| Test hazard cells | 712 at-risk team-phase cells |
| Test rank/top-k teams | 317 teams |
| Hazard positive rate | 0.2921 |
| Input | V3 factorized `X_node`, `G`, `A`, optional snapshot MR tier |

The main and control pilot directories use the same split IDs. Their split
files differ only in metadata/creation time.

## Main Architecture Pilot

Sorted by hazard AUC.

| Model | Hazard AUC | AP | Brier | ECE | Rank Spearman | Top10 AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mr_farnet` | 0.6943 | 0.4969 | 0.1880 | 0.0411 | 0.9326 | 0.9796 |
| `mr_rgcn` | 0.6942 | 0.4962 | 0.1886 | 0.0414 | 0.9345 | 0.9788 |
| `mr_rgcn_film` | 0.6935 | 0.4814 | 0.1868 | 0.0299 | 0.9432 | 0.9823 |
| `mr_gcn` | 0.6933 | 0.4871 | 0.1877 | 0.0456 | 0.9044 | 0.9819 |
| `mr_rgcn_attn_rank` | 0.6928 | 0.5012 | 0.1864 | 0.0350 | 0.9136 | 0.9771 |
| `mr_gat` | 0.6868 | 0.4885 | 0.1878 | 0.0395 | 0.9060 | 0.9836 |
| `mr_tcn` | 0.6844 | 0.4809 | 0.1884 | 0.0255 | 0.8715 | 0.9383 |
| `mr_sage` | 0.6757 | 0.4675 | 0.1918 | 0.0367 | 0.9114 | 0.9813 |

## Resolution Pilot

| Model | Hazard AUC | AP | Rank Spearman | Top10 AUC |
| --- | ---: | ---: | ---: | ---: |
| `phase_rgcn` | 0.6979 | 0.4973 | 0.8999 | 0.9694 |
| `mr_rgcn` | 0.6941 | 0.4938 | 0.9342 | 0.9788 |
| `phase_tcn` | 0.6906 | 0.4817 | 0.7388 | 0.9070 |
| `mr_tcn` | 0.6844 | 0.4809 | 0.8715 | 0.9383 |

## Decoder Pilot

| Model | Hazard AUC | AP | Brier | Rank Spearman |
| --- | ---: | ---: | ---: | ---: |
| `mr_rgcn_hazard_rank` | 0.6942 | 0.4943 | 0.1886 | 0.9342 |
| `mr_rgcn_pairwise_rank` | 0.6940 | 0.4959 | 0.1883 | 0.9140 |
| `mr_rgcn_attn_rank` | 0.6928 | 0.5012 | 0.1864 | 0.9136 |

## Graph-Control Pilot

| Graph | Hazard AUC | AP | Rank Spearman | Top10 AUC |
| --- | ---: | ---: | ---: | ---: |
| full observed | 0.6941 | 0.4937 | 0.9343 | 0.9788 |
| copressure only | 0.6931 | 0.4896 | 0.9341 | 0.9738 |
| combat only | 0.6894 | 0.4687 | 0.9004 | 0.9715 |
| proximity only | 0.6863 | 0.4791 | 0.9324 | 0.9831 |
| no graph | 0.6844 | 0.4801 | 0.8705 | 0.9382 |

## Interpretation

The pilot supports keeping graph representation in the final experiment:
`full observed` improves hazard AUC over no-graph by about 0.0096 on the same
test split. The gain is modest but consistent with the research claim that the
graph preserves additional relation signal.

R-GCN is the safest neural graph backbone from this pilot. `mr_farnet` is
essentially tied with plain `mr_rgcn` on hazard AUC, while `mr_rgcn_film` gives
the best rank Spearman and calibration among the R-GCN variants. The attention
rank decoder improves hazard AP and Brier but does not improve hazard AUC in
this short run.

MR input is not automatically better for short-run hazard AUC: `phase_rgcn`
beats `mr_rgcn` in the 120-match/2-epoch pilot. However, MR improves final
ranking and top-k structure relative to phase-only in this run, so it should
remain in the full experiment but should be evaluated task-wise rather than
claimed as universally better.

## Next Full Runs

Use `scripts/run_v3_full_main_matrix.sh` for the main architecture table and
`scripts/run_v3_full_controls_matrix.sh` for resolution, decoder, and graph
controls. For a memory-safe full run, leave `V3_CACHED` unset. For faster runs
on a machine with enough RAM, use `V3_CACHED=1`.

The pilot is not enough for a paper claim. Full DB runs need seed repeats,
strong tabular baselines under the same split, and shuffled/random graph
controls.
