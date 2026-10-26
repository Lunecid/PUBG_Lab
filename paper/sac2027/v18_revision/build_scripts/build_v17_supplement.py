"""Assemble the v17 supplementary notes from v16 text moved out of the main paper, plus new tables.

Moved blocks are copied verbatim from v16 main.tex by line range (checked by first words); only cross-references to
main-paper labels are rewritten as plain text.
"""
from pathlib import Path

import pandas as pd

REPO = Path('/Users/lunecid/project/PUBG')
V16 = REPO / 'paper/sac2027/v16_revision/main.tex'
V17 = REPO / 'paper/sac2027/v17_revision'
CE = REPO / 'reports/sac2027_patch_v14/control_evaluation'
L = V16.read_text(encoding='utf-8').split('\n')


def line(n, first):
    s = L[n - 1]
    assert s.startswith(first), (n, s[:70])
    return s


def lines(a, b, first, last):
    seg = L[a - 1:b]
    assert seg[0].startswith(first) and seg[-1].startswith(last), (a, b, seg[0][:50], seg[-1][:50])
    return '\n'.join(seg)


def fix_refs(s):
    reps = {
        r'Sec.~\ref{sec:method:rq2}': 'main paper, Section 4.2',
        r'(Fig.~\ref{fig:rq1})': '(Fig.~\\ref{fig:rq1})',
        r'Table~\ref{tab:steps}': 'Table~\\ref{tab:steps}',
        r'Table~\ref{tab:tuning}': 'Table~\\ref{tab:tuning}',
    }
    for a, b in reps.items():
        s = s.replace(a, b)
    assert r'\ref{sec:' not in s and r'\ref{tab:main}' not in s and r'\ref{fig:contrast}' not in s, s[:200]
    return s


# --- S1 habitat hazard model (v16 Sec. 4.2 method, 6.1 results, 7.2 interpretation)
rq1_method = lines(192, 202, 'For RQ1 we fit', 'An exploratory lite model')
rq1_method = rq1_method.replace('For RQ1 we fit, per cohort,', 'We fit, per cohort,')
rq1_fig = lines(290, 296, '\\begin{figure}[t]', '\\end{figure}')
rq1_results = lines(297, 306, 'Competition and remaining members', 'The lite model')
rq1_interp = line(373, 'Competition is strongest early')

# --- S2 path decomposition (v16 Sec. 4.4 and 6.3)
path_method = lines(244, 248, 'RQ3 asks where', 'Step (g) adds')
path_method = path_method.replace('RQ3 asks where performance differences come from, so we change',
                                  'The stepwise path asks where performance differences come from: we change')
path_results = lines(332, 334, 'Table~\\ref{tab:steps} pools', 'On the KRAFTON path')
path_method = path_method.replace('The additional matched controls below address these confounds', 'The main paper\'s matched controls address these confounds')
assert 'main paper\'s matched controls address' in path_method
path_results = path_results.replace('so matched controls follow.', 'so the main paper\'s matched controls separate them.')
assert 'separate them' in path_results

# --- S3 objective x mapping factorial (v16 Sec. 5.3 and 6.3)
factorial_method = ('An explicit $2\\times2$ analysis in the two largest cohorts crosses objective (full-follow-up '
                    'weighted survival/BCE) with probability mapping (global/phase-specific); the objective factor '
                    'bundles loss and horizon. Its Shapley contributions average the two possible orders~\\cite{Shorrocks13} '
                    'and sum to the joint change in every bootstrap draw.')
factorial_results = lines(341, 341, 'The objective--mapping factorial', 'The objective--mapping factorial')

# --- S4 equal-trial tuning (v16 Sec. 6.3)
tuning_results = lines(353, 353, 'With common inputs and BCE', 'With common inputs and BCE')
tuning_results = tuning_results.replace(' (Table~\\ref{tab:tuning} note)', ' (table note)')

# --- S5 population, history, graph controls (v16 Sec. 5.3, 6.3, 7.1)
pop_method = line(272, 'Human-team evaluations retain')
graph_method = line(274, 'An exploratory GATv2')
controls_results = line(357, 'Population and history controls move')
graph_interp = 'GATv2 detects small alignment effects and scores close to KRAFTON; existing relation summaries may limit the value of explicit graphs.'

steps_table = (V17 / 'tables/step_results.tex').read_text(encoding='utf-8')
tuning_table = (V17 / 'tables/tuning_results.tex').read_text(encoding='utf-8').replace(
    'Reference recipes (Table~\\ref{tab:main};', 'Reference recipes (main paper, Table~5;')
assert '\\ref{tab:main}' not in tuning_table

# --- S6 combined objective: all contrasts with 95% and 90% intervals
c = pd.read_csv(CE / 'objective_match_offset/contrasts.csv')
c = c[c.population == 'all']
rows = []
names = {'cox_truncated_bce_minus_cox_truncated': '(Cox+BCE) $-$ next-phase Cox',
         'bce_minus_cox_truncated_bce': 'BCE $-$ (Cox+BCE)',
         'bce_minus_cox_truncated': 'BCE $-$ next-phase Cox'}
metrics = {'original_within_phase': 'Within phase', 'centred_within_phase': 'Within phase, centered',
           'match_mean_between_match': 'Match--phase means only', 'within_match': 'Within match'}
for contrast, cname in names.items():
    rows.append(r'\multicolumn{4}{l}{\emph{' + cname + r'}} \\')
    for metric, mname in metrics.items():
        cells = []
        for cohort in ('erangel_tpp_squad', 'taego_tpp_squad'):
            r = c[(c.cohort == cohort) & (c.contrast == contrast) & (c.metric == metric)].iloc[0]
            cells.append(f'{r.estimate:+.4f} [{r.ci90_low:+.4f}, {r.ci90_high:+.4f}]')
        rows.append(f'\\quad {mname} & ' + ' & '.join(cells) + r' \\')
coxbce_table = '\n'.join([
    r'\begin{table*}[t]', r'\centering\footnotesize',
    r'\caption{Combined-objective test: AUC differences with 90\% intervals (equivalence bounds $\pm0.005$), three seeds, two cohorts.}',
    r'\label{tab:coxbce}', r'\begin{tabular}{lll}', r'\toprule',
    r'Contrast and evaluation unit & Erangel TPP squad & Taego TPP squad \\', r'\midrule',
    *[x.replace(' & ', ' & ', 2) for x in rows],
    r'\bottomrule', r'\end{tabular}', r'\end{table*}'])
coxbce_table = coxbce_table.replace(r'\multicolumn{4}{l}', r'\multicolumn{3}{l}')

# --- S7 pair shares
ps = pd.read_csv(CE / 'pair_share/pair_share.csv')
label = {'erangel_fpp_squad': 'Erangel FPP squad', 'erangel_tpp_duo': 'Erangel TPP duo', 'erangel_tpp_solo': 'Erangel TPP solo',
         'erangel_tpp_squad': 'Erangel TPP squad', 'miramar_desert_tpp_squad': 'Miramar TPP squad',
         'paramo_tpp_squad': 'Paramo TPP squad', 'rondo_tpp_squad': 'Rondo TPP squad', 'sanhok_tpp_squad': 'Sanhok TPP squad',
         'taego_fpp_squad': 'Taego FPP squad', 'taego_tpp_duo': 'Taego TPP duo', 'taego_tpp_squad': 'Taego TPP squad'}
share_rows = [f'{label[r.cohort]} & {r.n_matches} & {100 * r.share_mean_over_phases:.2f} & {100 * r.inverse_matches:.2f} \\\\'
              for r in ps.itertuples()]
share_table = '\n'.join([
    r'\begin{table}[t]', r'\centering\footnotesize',
    r'\caption{Share of same-match pairs among within-phase AUC pairs (patch 41.2).}', r'\label{tab:pairshare}',
    r'\begin{tabular}{lrrr}', r'\toprule', r'Cohort & Test matches & Same-match pairs (\%) & $1/M$ (\%) \\', r'\midrule',
    *share_rows, r'\bottomrule', r'\end{tabular}', r'\smallskip',
    r'\parbox{\columnwidth}{\scriptsize Per-phase shares averaged over eligible phases, as within-phase AUC averages per-phase AUCs; pair composition is shared by all designs.}',
    r'\end{table}'])

plan = r'''Before training the combined objective, we recorded its definition and decision rule in a dated plan. The loss is next-phase Cox (Breslow ties, administrative censoring at the next-phase boundary) plus the next-phase BCE with weight one, fixed in advance and not tuned; everything else matches the objective controls (backbone, configuration, seeds 42/13/2026, 12-epoch cap, patience three, stopping on validation within-phase AUC, common validation calibration with phase indicators). If match-level offsets explain BCE's within-phase advantage, then in both cohorts (1) the combined objective's within-phase AUC exceeds next-phase Cox's (95\% CI above zero), (2) its within-phase AUC is practically equivalent to BCE's (90\% CI within $\pm0.005$), and (3) its within-match AUC is practically equivalent to next-phase Cox's (90\% CI within $\pm0.005$). All three held (Table~\ref{tab:coxbce}). The combined objective closes 58\% (Erangel) and 71\% (Taego) of the within-phase gap; the remaining difference from BCE is small but its 95\% CI excludes zero in both cohorts. The recovered gain lies in the match--phase means, as the explanation predicts, and the combined objective also improves log loss over next-phase Cox by 0.0031 and 0.0034.'''

doc = r'''% !TEX program = xelatex
\documentclass[sigconf,anonymous,nonacm]{acmart}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs,tabularx}
\graphicspath{{figures/}}
\settopmatter{printacmref=false,printfolios=true}
\setcopyright{none}
\acmConference[SAC '27]{The 42nd ACM/SIGAPP Symposium on Applied Computing}{April 5--9, 2027}{Gwangju, South Korea}
\acmDOI{}
\acmISBN{}
\hypersetup{hidelinks}
\setlength{\emergencystretch}{1.5em}
\raggedbottom
\begin{document}
\title[Supplementary Notes]{Supplementary Notes: Ranking Teams or Matches? Evaluation Units for Next-Phase Elimination Prediction in Battle Royale}
\maketitle
These notes hold analyses summarized in the main paper. Section and table numbers without a prefix refer to these notes; ``main paper'' marks references to the submission.

\section{Combined-Objective Test}
''' + plan + '\n' + coxbce_table + r'''

\section{Same-Match Pair Shares}
Table~\ref{tab:pairshare} lists, for each primary cohort, the share of same-match pairs among the (eliminated, surviving) pairs that within-phase AUC compares. The share tracks $1/M$ for $M$ test matches.
''' + share_table + r'''

\section{Habitat Hazard Model}
''' + rq1_method + '\n\n' + rq1_fig + '\n' + rq1_results + '\n\n' + rq1_interp + r'''

\section{Stepwise Path Decomposition}
''' + path_method + '\n\n' + steps_table + '\n' + path_results + r'''

\section{Objective--Mapping Factorial}
''' + factorial_method + ' ' + factorial_results + r'''

\section{Equal-Trial Tuning}
''' + tuning_table + '\n' + tuning_results + r'''

\section{Population, History, and Graph Controls}
''' + pop_method + '\n\n' + graph_method + '\n\n' + controls_results + ' ' + graph_interp + r'''

\bibliographystyle{ACM-Reference-Format-citation-order}
\bibliography{refs}
\end{document}
'''
doc = fix_refs(doc)
(V17 / 'supplement.tex').write_text(doc, encoding='utf-8')
print('supplement written', len(doc))
