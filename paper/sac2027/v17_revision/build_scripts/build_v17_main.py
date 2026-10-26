"""Assemble v17 main.tex (evaluation-unit reframing) from the v16 manuscript plus new sections.

Unchanged v16 blocks are copied by line range and checked by their first/last lines.
"""
from pathlib import Path

V16 = Path('/Users/lunecid/project/PUBG/paper/sac2027/v16_revision/main.tex')
OUT = Path('/Users/lunecid/project/PUBG/paper/sac2027/v17_revision/main.tex')
L = V16.read_text(encoding='utf-8').split('\n')


def block(a, b, first, last):
    seg = L[a - 1:b]
    assert seg[0].startswith(first), (a, seg[0][:60])
    assert seg[-1].startswith(last), (b, seg[-1][:60])
    return '\n'.join(seg)


preamble = block(1, 16, '% !TEX program', '\\begin{document}')
sec3_body = block(84, 147, '\\subsection{Data Collection}', 'This is an interval event probability')
tab_blocks = block(159, 188, '\\begin{table}[t]', '\\end{table}')
designs = block(206, 240, 'Our design learns', '\\end{table*}')
designs = designs.replace('All matched objective--mapping controls fit their mapping on validation data.',
                          'All matched controls fit their mapping on validation data.')
assert 'All matched controls fit their mapping' in designs
tail = block(388, 393, '\\section*{AI Assistance Disclosure}', '\\end{document}')

title = r'''\title[Evaluation Units for Next-Phase Elimination Prediction]{Ranking Teams or Matches? Evaluation Units for Next-Phase Elimination Prediction in Battle Royale}'''

abstract = r'''\begin{abstract}
Battle-royale match review asks which teams alive at a phase boundary are most likely to be eliminated before the next phase ends. Discrimination is usually measured on samples pooled across matches, but even within phases fewer than 3\% of AUC pairs in our data share a match, so the metric mainly ranks teams across matches. We adapt established designs to a common landmark task on 13,948 PUBG matches split by game patch and isolate the training objective with matched controls. Truncating Cox learning at the next phase leaves cross-entropy no within-match advantage (below 0.001 AUC), but a within-phase advantage of 0.006--0.008 remains. Centering scores within each match--phase reduces it to at most 0.002, and adding a pointwise cross-entropy term to the Cox loss closes 58--71\% of it without lowering within-match AUC, as predicted before training: objectives normalized within a match's risk set receive no training signal for match-level risk offsets. Most within-phase gains of our designs over a phase-adjusted survival design likewise shrink after centering, and their top-3 capture of eliminations differs by at most 0.6 percentage points. Landmark predictors with group-level risk should therefore be evaluated within groups as well as within time strata.
\end{abstract}'''

ccs = r'''\begin{CCSXML}
<ccs2012>
<concept><concept_id>10010147.10010257</concept_id><concept_desc>Computing methodologies~Machine learning</concept_desc><concept_significance>500</concept_significance></concept>
<concept><concept_id>10002951.10003227.10003351</concept_id><concept_desc>Information systems~Data mining</concept_desc><concept_significance>300</concept_significance></concept>
</ccs2012>
\end{CCSXML}
\ccsdesc[500]{Computing methodologies~Machine learning}
\ccsdesc[300]{Information systems~Data mining}
\keywords{landmark prediction, evaluation metrics, survival analysis, spatio-temporal telemetry, esports analytics}
\maketitle'''

intro = r'''\section{Introduction}
\label{sec:introduction}
In a battle-royale game, teams must survive their opponents while the playable area shrinks. In PUBG: BATTLEGROUNDS (PUBG), up to 100 players compete individually or in teams of two or four. The safe zone shrinks in scheduled phases, players outside it take continuous damage, and a new safe zone is revealed at each phase boundary. We study \emph{next-phase elimination risk}: the probability that a team alive at a boundary is eliminated before the next phase ends.

The intended use is retrospective match review: from spectator-view telemetry, which joins all teams' trajectories with event and environment records~\cite{PUBGAPI,Zheng15}, an analyst compares the teams alive at one boundary of one match. Risk also differs between matches, because zones, team counts, and fights differ, yet designs are evaluated on samples pooled over matches. Overall AUC rewards predicting each phase's elimination rate; within-phase AUC removes that advantage but still pools matches, and with $M$ test matches only about $1/M$ of its pairs share a match. A model can thus raise within-phase AUC by ranking matches by their risk level, which a review of one match does not use.

The evaluation unit matters because objectives differ in what they can learn. KRAFTON's survival design~\cite{KRAFTON} learns elimination order from risk sets normalized within a match, as Cox partial likelihood does~\cite{Cox72}. Such objectives are unchanged when a constant is added to every score of a match, so they receive no training signal for match-level risk, whereas cross-entropy on next-phase labels does. A within-phase AUC gap between the two can therefore reflect match-level offsets rather than better rankings of teams that meet. Prior PUBG designs also differ in target, inputs, prediction unit, and metric~\cite{Sagong25a,KRAFTON}, so their reported gaps do not identify the responsible design choice.

We adapt these designs to a common landmark discrete-time task with a shared information cutoff and patch-separated evaluation, and compare them on three evaluation units: within-phase AUC, within-match AUC, and within-phase AUC after centering scores within each match--phase. A habitat-loss view~\cite{Fahrig03} organizes the inputs without asserting an ecological mechanism.

\paragraph{Research questions.}
\begin{itemize}
\item \textbf{RQ1:} With architecture and inputs fixed, how much of the difference between survival and cross-entropy objectives reflects the prediction horizon, and how much reflects match-level risk offsets?
\item \textbf{RQ2:} How does a design that learns next-phase elimination probability compare with prior designs adapted to the same task, and on which evaluation unit do the differences appear?
\item \textbf{RQ3:} With target and evaluation unit aligned, how much do inputs and model structure change performance?
\end{itemize}

\paragraph{Contributions.}
First, we show that within-phase AUC, the usual phase-adjusted comparison, mostly ranks teams across matches: same-match pairs are 0.15--2.8\% of its pairs in our 11 cohorts, so it credits match-level risk offsets that a within-match review cannot use. We report within-match AUC and a match--phase centering diagnostic as separate evaluation units (Section~\ref{sec:method:evaluation}). Second, matched controls show that this unit decides which objective looks better. Aligning the Cox horizon with the label leaves cross-entropy no within-match advantage (below 0.001 AUC), while its within-phase advantage of 0.006--0.008 shrinks to at most 0.002 after centering, and adding a pointwise cross-entropy term to the Cox loss closes 58--71\% of that gap without lowering within-match AUC, as predicted before training (Section~\ref{sec:results:rq1}). Most within-phase gains of our designs over phase-adjusted KRAFTON shrink the same way (Section~\ref{sec:results:rq2}). Third, we define common-cutoff, patch-separated samples, adapt two prior designs, and release the code (Sections~\ref{sec:method}, \ref{sec:method:design}). The lesson extends to landmark prediction wherever risk varies by group (Section~\ref{sec:discussion:implications}).'''

related = r'''\section{Related Work}
\label{sec:related}

\subsection{In-Game Outcome and Risk Prediction}
\label{sec:related:esports}
PUBG outcome prediction began with final placement from post-match statistics~\cite{KagglePUBG,LeeLee25}; other esports predict wins or short-horizon deaths from in-game information~\cite{Hodge21,Katona19,Marshall22}.
We adapt the PUBG activity-score~\cite{Sagong25a} and survival-learning~\cite{KRAFTON} designs to shared samples, preserving their intended inputs and training where telemetry permits (Table~\ref{tab:compare}).

\subsection{Survival Models, Evaluation, and Multi-Agent Structure}
\label{sec:related:survival}
The Cox model estimates relative risks without specifying a baseline hazard, the risk shared by all at a given time~\cite{Cox72}, whereas discrete-time survival models estimate interval event probabilities with interval baselines~\cite{SingerWillett93}, also with neural networks~\cite{Gensheimer19}; landmarking estimates subsequent risk for those alive at a prediction time~\cite{vanHouwelingen07}.
DeepHit combines event-time likelihood and ranking~\cite{DeepHit18}, and Dynamic-DeepHit learns from longitudinal measurements~\cite{DynamicDeepHit20}. Because a baseline turns relative-risk scores into interval probabilities~\cite{PrenticeGloeckler78}, we compare objectives after attaching one.
Time-dependent ROC analysis compares subjects within risk sets~\cite{HeagertyZheng05}, and discrimination in clustered clinical data has been assessed within clusters, apart from risk differences between clusters~\cite{vanKlaveren14}; we show that this choice decides which training objective looks better.
In multi-agent scenes, attention is combined across agent and time axes~\cite{Ngiam22} or graph structure is injected as attention bias~\cite{Ying21}, and player-relation graphs have been used in sports~\cite{Xenopoulos21}; we measure such structure with inputs and objective fixed.'''

sec3_head = r'''%======================================================================
\section{Data and Prediction Task}
\label{sec:method}

We construct one sample per team alive at a phase boundary and predict elimination before the next phase ends. Figure~\ref{fig:pipeline} connects data collection and sample construction to the three research questions and their evaluation units.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{fig_pipeline_clean.pdf}
\caption{Data collection, prediction-time sample construction, comparisons, and evaluation. Panels A--B define the data and task; panels C--D organize the three research questions and their evaluation units.}
\label{fig:pipeline}
\Description{Telemetry is converted to samples using a common cutoff, then used for matched objective controls, design comparisons, and input and structure analyses, evaluated within phases, within matches, and after match-phase centering.}
\end{figure*}
'''

sec4_head = r'''\section{Inputs and Model Designs}
\label{sec:method:design}
\subsection{Habitat Context and Other Inputs}
\label{sec:method:representation}
Habitat-loss ecology relates population persistence to the amount and arrangement of habitat~\cite{Fahrig03}; we borrow it only to organize a team's state into four constructs of position and crowding.
\emph{Boundary exposure} is how far a team is outside the current zone (outside depth) or inside it (inside depth).
\emph{Movement residual} is the retrospective speed difference $(d-\ell)/\tau_k$~\cite{Nathan08}, where $d$ is the distance from the last observed team centroid to the boundary of the target zone already visible at the cutoff, $\ell$ sums horizontal centroid displacements over alive grid points from phase start to the cutoff, and $\tau_k$ is the scheduled phase duration estimated on training matches; it is an empirical proxy, and inside-zone teams that move get a negative residual.
\emph{Local competition density} counts other teams within 200\,m (3-D centroid distance) at the focal team's last alive grid point, and \emph{remaining members} counts members alive at the cutoff.
Our inputs comprise a habitat block with the first three constructs (remaining members are in team status), other team-feature blocks (Table~\ref{tab:blocks}), and, for the Transformer of Section~\ref{sec:method:rq2}, the team centroid and three same-phase relations: proximity (dropped for pairs that fought), combat, and shared zone pressure (two teams within 200\,m, both outside or within 150\,m of the zone edge).
A per-cohort logistic hazard model with phase intercepts, the four constructs, and an AI-team control (supplementary notes) shows that competition and remaining members keep their directions in every evaluable cohort ($+0.284$ and $-0.594$ log-odds per SD, pooled over 11 and 10 cohorts), that depth associations change across phases, and that the movement residual's coefficient depends on AI adjustment; these associations are observational and guide inputs, not causal claims.
'''

designs_head = r'''\subsection{Prediction Designs}
\label{sec:method:rq2}'''

controls = r'''\subsection{Matched Controls}
\label{sec:method:decomposition}
\emph{Objective controls (RQ1)} hold observations, our inputs, the KRAFTON-derived backbone, and a configuration fixed before tuning, and vary only the loss: the upstream weighted survival objective and standard Cox partial likelihood with Breslow ties, each with full follow-up or truncated at the next-phase boundary; BCE on next-phase labels; and next-phase Cox plus that BCE with weight one. For weighted truncation only events beyond the horizon are excluded, and standard Cox uses administrative horizon censoring. The combined objective tests the offset explanation of Section~\ref{sec:discussion:drivers}: before training it, we recorded its weight and three predictions, namely higher within-phase AUC than next-phase Cox, within-phase AUC practically equivalent to BCE's, and within-match AUC practically equivalent to next-phase Cox's. All six conditions use three seeds, validation within-phase AUC for stopping, and a common validation logistic calibration with phase indicators.

\emph{Path and structure controls (RQ3).} A stepwise path changes each prior design toward ours (supplementary notes): on the Sagong path, with GBDT fixed, we add habitat context to Sagong's inputs and then replace the rest with ours; on the KRAFTON path we replace inputs, attach a phase baseline, switch to next-phase cross-entropy, and replace the architecture; a last step adds KRAFTON's inputs to GBDT. Five primary comparisons were fixed before the patch split was defined: a 2$\times$2 on/off design of PTR-Former's phase and team axes (AUC for both axes and, for the team axis, Spearman rank correlation within matches), a control with relations shuffled within matches, and a zone token (the safe zone as an extra attention token) tested without habitat and zone features. Equal-trial tuning~\cite{Bergstra12,Cawley10} compares GBDT, the KRAFTON-derived backbone, and PTR without ranking loss in the two largest cohorts, each with the same 153 columns (150 features, centroid coordinates, AI indicator), BCE, and 16 configurations selected by validation NLL; neural models share the search list, a 12-epoch cap, three-epoch patience, and eight-match batches, and selected settings are rerun with three seeds. Human-only training, history, and graph (GATv2~\cite{Brody22}) controls are in the supplementary notes.
'''

evaluation = r'''\section{Evaluation Units and Statistics}
\label{sec:method:evaluation}
All designs are scored on identical patch-41.2 test samples (Figure~\ref{fig:pipeline}D).

\subsection{Evaluation Units}
\label{sec:method:units}
AUC is the probability that an eliminated team receives a higher risk than a surviving one. Because elimination rates vary by phase, predicting the phase alone raises AUC, so we also report within-phase AUC, the unweighted mean of per-phase AUCs over phases with at least 30 rows and both outcome classes. Within one phase, a pair of an eliminated and a surviving team either shares a match or comes from two matches, so each per-phase AUC equals $\pi A_W+(1-\pi)A_B$, where $A_W$ and $A_B$ are AUCs over same-match and different-match pairs and $\pi$ is the share of same-match pairs. With $M$ test matches, $\pi\approx1/M$: 0.15--2.8\% in our cohorts and 0.15\%/0.20\% in the two largest. Within-phase AUC is therefore almost entirely a ranking across matches. Within-match AUC compares only teams alive in the same match at the same boundary~\cite{HeagertyZheng05,vanKlaveren14}, pooling pairs over risk sets that contain both classes; it is the comparison a match review makes.

To locate a within-phase difference, we recompute within-phase AUC after subtracting each model's mean logit per match--phase (centering), which removes any match-level offset, and compute $A_B$ from the match--phase mean logits alone. This diagnostic was added after the objective results were examined and is not an additive decomposition of AUC into causes; the combined objective of Section~\ref{sec:method:decomposition} tests the explanation it suggests.

For review use, top-3 capture is the share of next-phase eliminations among the three highest-risk teams at each boundary of a match (ties split at random; boundaries without eliminations excluded), compared with random selection and a fewest-remaining-members rule. Probability accuracy uses log loss, Brier score, and phase calibration error $\sum_k(n_k/n)|\bar p_k-\bar y_k|$, where $n_k$ is phase $k$'s sample count, $n$ the cohort total, and bars denote mean prediction and outcome. Calibration slope and calibration-in-the-large are the coefficient of a logistic regression of outcomes on predicted log-odds and its intercept with that coefficient fixed at one~\cite{VanCalster16}.

\subsection{Statistical Comparisons}
Per-cohort confidence intervals come from a match-level paired bootstrap with 2,000 test-match resamples.
Cohorts are pooled with a DerSimonian--Laird random-effects model~\cite{DerSimonian86}, checked with REML and Hartung--Knapp intervals~\cite{HartungKnapp01}.
Decision rules, the SESOI, and the five primary comparisons were fixed before the patch-split models were trained. A difference is a gain if its one-sided $p<0.05$ and its estimate is at least the smallest effect size of interest (SESOI), 0.005 (0.01 for rank correlation; Section~\ref{sec:results:rq3} relates it to top-3 capture), practically equivalent if its 90\% CI lies within $\pm$SESOI~\cite{Schuirmann87}, and inconclusive otherwise.
Holm correction applies to the five one-sided superiority tests against zero~\cite{Holm79}; the 90\% equivalence intervals are unadjusted, and a gain is evidence against zero, not evidence that the true effect exceeds the SESOI (Table~\ref{tab:primary}). The patch protocol was recorded before training, and equal-trial tuning selects settings on patch 41.1 alone.

\subsection{Training}
Reference GBDT uses 200 trees (learning rate 0.06); PTR (width 128, four heads, two blocks) and KRAFTON (width 64, eight heads/layers) use AdamW ($10^{-3}$/$10^{-4}$) with validation stopping, and the artifact lists all settings. Human-team evaluations retain opponents as observed context. Test data select neither settings, checkpoints, nor mappings.
'''

results = r'''%======================================================================
\section{Results}
\label{sec:results}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{fig_eval_units.pdf}
\caption{(a) Matched objective controls in the two largest cohorts: BCE minus full-follow-up Cox, next-phase Cox, and next-phase Cox with an added BCE term (Cox+BCE). (b) Within-phase AUC differences before and after match--phase centering, with within-match differences; KRAFTON+phase is phase-adjusted KRAFTON. Bars are 95\% confidence intervals; pooled rows use random-effects estimates across 11 cohorts.}
\label{fig:contrast}
\Description{Dot plots of AUC differences. Against full-follow-up Cox, cross-entropy gains within matches and within phases; against next-phase Cox only the within-phase gain remains, and it shrinks against Cox with an added cross-entropy term. Within-phase gains of cross-entropy and of our designs over phase-adjusted KRAFTON shrink after match-phase centering toward the within-match differences.}
\end{figure*}
Rebuilding inputs from telemetry truncated at the cutoff reproduced all 453 sampled cutoffs (72 matches, stratified by cohort and split), whereas disabling truncation changed 450 of them. Only two of the 817,766 retained rows, both in training, lacked an observed living member at the cutoff.
Unless stated otherwise, synthesis covers the 11 cohorts with at least 50 training matches; differences use random-effects pooling with 95\% CIs, and counts in parentheses give the number of evaluable cohorts with the pooled sign. Matched controls use the two largest cohorts, Erangel and Taego TPP squad. An AUC difference of 0.01 means a 1-percentage-point (pp) higher probability of ranking an eliminated team above a survivor.

\subsection{RQ1: Horizon Alignment and Match-Level Offsets}
\label{sec:results:rq1}
\input{tables/objective_results}
With architecture, inputs, optimization, and stopping aligned, truncating standard Cox at the next-phase boundary improves within-match AUC by $+0.0123$ $[0.0088,\allowbreak 0.0160]$ in Erangel and $+0.0147$ $[0.0112,\allowbreak 0.0180]$ in Taego (Table~\ref{tab:objective}, Fig.~\ref{fig:contrast}a). BCE minus truncated Cox is then only $+0.0008$ $[-0.0013,\allowbreak 0.0028]$ and $-0.0008$ $[-0.0028,\allowbreak 0.0012]$, so cross-entropy's within-match advantage over full-follow-up objectives reflects horizon alignment.

BCE nevertheless retains a within-phase advantage over truncated Cox, $+0.0060$ $[0.0038,\allowbreak 0.0082]$ and $+0.0076$ $[0.0055,\allowbreak 0.0097]$, and the evaluation units place it at the match level (Fig.~\ref{fig:contrast}b). Centering logits within each match--phase shrinks it to $+0.0015$ $[-0.0003,\allowbreak 0.0033]$ and $+0.0008$ $[-0.0009,\allowbreak 0.0024]$, whereas match--phase mean logits alone rank matches better under BCE by $+0.016$ $[0.013,\allowbreak 0.020]$ and $+0.019$ $[0.015,\allowbreak 0.022]$. Truncated Cox's means still reach AUC 0.586 and 0.597, about 84\% of BCE's above-chance match-level discrimination, because team inputs carry match state; truncation leaves this component unchanged ($+0.002$ and $-0.004$).

The combined objective met all three predictions recorded before training. Adding the BCE term to truncated Cox raises within-phase AUC by $+0.0035$ $[0.0015,\allowbreak 0.0056]$ in Erangel and $+0.0054$ $[0.0040,\allowbreak 0.0068]$ in Taego, closing 58\% and 71\% of the gap to BCE. The remaining differences from BCE, $+0.0025$ and $+0.0022$, have 90\% CIs within $\pm0.005$ (upper limits 0.0040 and 0.0037), although their 95\% CIs exclude zero. Within-match AUC stays practically equivalent to truncated Cox ($+0.0015$ and $+0.0007$; 90\% CIs within $[-0.0005,\allowbreak 0.0030]$). The recovered gain is match-level: match--phase mean logits alone improve by $+0.011$ $[0.008,\allowbreak 0.014]$ and $+0.012$ $[0.010,\allowbreak 0.015]$, while centered within-phase AUC changes by only $+0.0009$ and $+0.0010$.

\subsection{RQ2: Designs on Three Evaluation Units}
\label{sec:results:rq2}
\input{tables/main_results}
With our inputs, mean AUC is 0.768 (GBDT) and 0.769 (PTR-Former), versus 0.720 for the KRAFTON adaptation, 0.747 after phase-baseline adjustment, and 0.675 for the Sagong adaptation (Table~\ref{tab:main}). The pooled advantages over phase-adjusted KRAFTON are $+0.0197$ $[0.0143,\allowbreak 0.0250]$ for PTR ($I^2=0.85$) and $+0.0191$ $[0.0138,\allowbreak 0.0244]$ for GBDT, and PTR exceeds Sagong by $+0.0922$ $[0.0830,\allowbreak 0.1014]$. The PTR advantage persists among human teams ($+0.0204$ $[0.0141,\allowbreak 0.0266]$) and in normal and ranked matches ($+0.0166$ and $+0.0246$; GBDT $+0.0157$ and $+0.0315$; ten and six cohorts with at least 20 test matches of that type, all intervals excluding zero). The combined-input GBDT reaches 0.786 but uses additional information. The regularized logistic baseline (0.759) exceeds phase-adjusted KRAFTON by $0.0103$ $[0.0009,\allowbreak 0.0196]$ overall but not within matches ($-0.0088$ $[-0.0203,\allowbreak 0.0028]$).

The evaluation unit changes the size of these gains (Fig.~\ref{fig:contrast}b). Relative to phase-adjusted KRAFTON, within-phase AUC is higher by $+0.0271$ $[0.0180,\allowbreak 0.0361]$ for PTR and $+0.0187$ $[0.0099,\allowbreak 0.0276]$ for GBDT, but within-match AUC only by $+0.0071$ $[0.0030,\allowbreak 0.0113]$ and $+0.0100$ $[0.0071,\allowbreak 0.0128]$. Centering cuts the within-phase gains to $+0.0093$ $[0.0044,\allowbreak 0.0143]$ and $+0.0073$ $[0.0020,\allowbreak 0.0126]$, close to the within-match gains, whereas match--phase mean logits alone favor PTR and GBDT by $+0.048$ $[0.036,\allowbreak 0.060]$ and $+0.043$ $[0.032,\allowbreak 0.054]$. PTR exceeds GBDT by $+0.0058$ $[0.0035,\allowbreak 0.0080]$ within phases, but their within-match difference is uncertain ($-0.001$ $[-0.006,\allowbreak 0.004]$), and centering leaves $+0.0011$ $[-0.0022,\allowbreak 0.0044]$.

For review, the three highest-risk teams at a boundary include 51.7\% of next-phase eliminations under GBDT, 51.6\% under PTR-Former, 51.1\% under phase-adjusted KRAFTON, and 53.3\% with combined inputs, versus 34.7\% at random and 45.1\% for the fewest-remaining-members rule (pooled GBDT gain over this rule $+0.068$ $[0.062,\allowbreak 0.073]$, 11/11). The three designs lie within 0.6\,pp of one another, about a tenth of their gain over the rule, in line with their small within-match differences.

Phase-baseline adjustment lowers KRAFTON's phase calibration error from 0.068 to 0.012 (GBDT 0.020, PTR 0.025) while preserving rankings within phases and matches. Mean calibration slopes are 0.86 (GBDT), 0.95 (PTR-Former), and 0.95 (phase-adjusted KRAFTON), where values below one indicate overly spread predictions; GBDT's calibration-in-the-large of $-0.09$ indicates mild overprediction on patch 41.2. The survival recipe predicts winners better: KRAFTON exceeds PTR by $0.0175$ $[0.0020,\allowbreak 0.0324]$ in winner accuracy, while their C-index difference is uncertain ($0.0024$ $[-0.0005,\allowbreak 0.0053]$); with our inputs and backbone fixed, weighted survival exceeds BCE by $0.0252$ in winner accuracy and $0.0065$ in C-index (both intervals exclude zero).

\subsection{RQ3: Inputs and Structure}
\label{sec:results:rq3}
Inputs change performance more than structure. On the Sagong path, with GBDT fixed, adding habitat context to Sagong's inputs raises AUC by $+0.046$ $[0.041,\allowbreak 0.052]$ and replacing the rest with our inputs by $+0.028$ $[0.024,\allowbreak 0.033]$ (both 11/11); removing the habitat block from our GBDT costs $0.0127$ $[0.0110,\allowbreak 0.0144]$ overall and $0.0151$ $[0.0119,\allowbreak 0.0182]$ within-phase AUC (11/11). Our inputs score slightly below the reconstructed KRAFTON inputs ($-0.0035$ $[-0.0066,\allowbreak -0.0004]$), more so within matches ($-0.0124$ $[-0.0170,\allowbreak -0.0078]$), and combining both sets improves AUC over ours by $+0.0170$ $[0.0141,\allowbreak 0.0198]$ (11/11), so the sets are complementary. On the KRAFTON path, the phase baseline ($+0.020$, zero within phases by monotonicity) and the next-phase cross-entropy recipe ($+0.016$) contribute most (supplementary notes).

\input{tables/primary_decisions}
Structural effects are small. Three primary comparisons meet practical equivalence and both team-axis comparisons are inconclusive (Table~\ref{tab:primary}); REML/Hartung--Knapp retains these verdicts. Team-axis AUC is positive (Holm $p<0.001$), but its 90\% interval crosses 0.005; the verdict would be a gain for $\delta\le0.0043$ and equivalence for $\delta\ge0.0055$. Team attention raises within-phase AUC by $+0.0108$ $[0.0086,\allowbreak 0.0130]$ (11/11), of which $+0.0049$ $[0.0022,\allowbreak 0.0076]$ is match-level and $+0.0043$ $[0.0020,\allowbreak 0.0067]$ remains after centering. Across nine design contrasts, top-3 capture changes by 0.66 $[0.54,\allowbreak 0.78]$ per unit of within-match AUC, so a within-match difference of 0.005 corresponds to about 0.33\,pp, or one more captured elimination per about 90 boundaries; the primary structure contrasts change capture by at most 0.36\,pp.

Under equal-trial tuning with common inputs and BCE, PTR minus the KRAFTON backbone is $-0.0011$ (Erangel) and $+0.0006$ (Taego) in AUC, both with intervals including zero, and within-match differences from GBDT are $-0.0062$ $[-0.0097,\allowbreak -0.0029]$ and $+0.0026$ $[-0.0009,\allowbreak 0.0059]$. Tuning changes reference GBDT and PTR AUC by at most 0.002, so no architecture dominates both cohorts and metrics under this search budget. Human-only training, history, and GATv2 controls move AUC by at most 0.0034 (supplementary notes).
'''

discussion = r'''%======================================================================
\section{Discussion}
\label{sec:discussion}
\subsection{Why the Evaluation Unit Decides the Objective Comparison}
\label{sec:discussion:drivers}
Next-phase truncation leaves no clear within-match difference between Cox and BCE, so horizon alignment explains cross-entropy's within-match advantage. The remaining within-phase advantage follows from what each objective can learn. Cox-type objectives normalize scores within each match's risk set: adding a constant to every score in a match--phase leaves the loss unchanged, so they give no training signal for a match-level risk offset, and match information enters Cox scores only as a by-product of team-level fitting. BCE's derivative with respect to a common offset $c_{m,k}$ is $\sum_{i\in\mathcal{R}_{m,k}}(\hat p_{m,i,k}-Y_{m,i,k})$, the gap between expected and observed eliminations, so it provides that signal wherever the model can represent such offsets. Because within-phase AUC mostly compares teams from different matches, the advantage appears within phases but not within matches; centering removes most of it, and adding the pointwise term to Cox recovers most of it at the match level, as predicted. Conditional likelihoods in matched and fixed-effects designs share this property: group-level effects drop out~\cite{Chamberlain80}. Learning to rank faces the same invariance: without ties, the Cox partial likelihood is the Plackett--Luce likelihood that ListMLE maximizes~\cite{Xia08}, such listwise scores are uncalibrated across queries, and adding a pointwise sigmoid cross-entropy is one remedy~\cite{Bai23}; our combined objective shows that the remedy carries over to landmark survival prediction.

\subsection{Implications for Evaluation}
\label{sec:discussion:implications}
Which unit to report depends on the use. Reviewing one match needs within-match ranking, where horizon-aligned Cox, cross-entropy, and our designs differ little. Comparing or pooling probabilities across matches needs match-level risk, which within-phase AUC rewards and risk-set objectives do not learn, and predicting winners favors the survival recipe. Reporting within-phase AUC alone therefore overstates design differences for review: our designs' within-phase gains of 0.019--0.027 over phase-adjusted KRAFTON correspond to within-match gains of 0.007--0.010 and at most 0.6\,pp of top-3 capture. Nothing in this pair composition is specific to games. Whenever landmark risk varies by group, as for patients assessed on the same ward-day or agents in the same traffic scene, AUC pooled within a time stratum mostly compares groups once there are many of them. We therefore recommend reporting within-group AUC next to within-stratum AUC, checking a within-stratum gap with group centering, and adding a pointwise term to risk-set objectives when calibration across groups matters.

\subsection{Uses and Limitations}
\label{sec:discussion:limits}
The supported application is retrospective spectator-view match review; top-3 capture is only a proxy, analyst benefit and latency are unmeasured, and live use would require an online eligibility audit and real-time feeds. The learned designs beat the fewest-remaining-members rule by about 6\,pp in top-3 capture but differ by at most 0.6\,pp among themselves, so the study informs evaluation practice more than the choice of a review system.

Eligibility uses final elimination times, which future returns can affect, so online risk-set validity is unverified; the cutoff checks cover retained rows only. The data span ten weeks and three patches with one solo cohort, Karakin has six training matches, and there is no prospective or new-rule test; cohorts share a collection window, and bootstrap intervals condition on fitted models. The centering diagnostic is post hoc, and the matched controls, including the combined objective with one untuned weight, cover two cohorts and one backbone. Baselines use proxy inputs without original checkpoints and change Sagong's unit and interval, so adaptation quality can affect their comparisons; without redistributed telemetry, third parties can inspect the code but not rerun the exact samples. Habitat associations are observational: teams choose positions, and zone and combat damage are not separated. Tuning covers two cohorts, a limited search, and three seeds without equalizing computation; a standalone MLP baseline remains untested, and ranked-match strata cover six cohorts with 26--108 test matches each.

\paragraph{Code availability.}
Anonymized training, evaluation, and figure-generation code, feature definitions, configurations, verification notes, and supplementary notes (habitat hazard model, path decomposition, tuning table, and further controls) are available at \artifacturl{}; raw and processed telemetry are not redistributed.

\section{Conclusion}
\label{sec:conclusion}
Within-phase AUC, the usual phase-adjusted metric, ranks teams almost entirely across matches. In patch-separated next-phase elimination prediction, this unit decides which objective looks better: aligning the Cox horizon removes cross-entropy's within-match advantage, the remaining within-phase advantage shrinks after match--phase centering, and a pointwise cross-entropy term added to Cox recovers most of it without lowering within-match AUC, consistent with match-level risk offsets for which risk-set objectives receive no training signal. Most within-phase gains of our designs over phase-adjusted KRAFTON shrink the same way, inputs matter more than structure, and equal-trial comparisons in two cohorts yield no consistent architecture winner. Landmark predictors with group-level risk should be evaluated within groups as well as within strata. Future work should test the combined objective across cohorts and weights, private-observation inputs, newly collected patches including the zone-rule revamp~\cite{PUBGBlueZone2026}, and other grouped landmark settings.
'''

doc = '\n'.join([
    preamble.replace('\\begin{document}', '\\begin{document}'),
    title, abstract, ccs, intro, '', related, '', sec3_head, sec3_body, '',
    sec4_head, tab_blocks, '', designs_head, designs, '', controls, evaluation, '', results, '', discussion, tail, ''])
OUT.write_text(doc, encoding='utf-8')
print('written', len(doc))
