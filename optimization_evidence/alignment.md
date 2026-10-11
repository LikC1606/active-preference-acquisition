# Evidence for the frozen protocol

The sole optimization surface remains preference-pair selection. Baseline is
random selection; Reference is ARM-FI's D-optimal rule. No algorithm, seed,
metric or protocol will be selected according to whether Reference is positive.

| Item | Chosen setting | Evidence and scope |
| --- | --- | --- |
| Preference labels | Bernoulli(sigmoid(raw reward difference)) | Paper p.2 Eq.(1); ARM-FI bradleyterry.py:simulate_label. Experiment-specific rescaling is unpublished. |
| Annotation budget | 20000, in batches of 500 | Paper pp.7-8 Figures 3-4 and sections 5.1-5.3. |
| Candidate pool | 500 prompts, all 45 within-prompt response pairs, 22500 pairs | Paper p.7 section 5.1; ARM-FI random_candidate_withinprompt. Appendix A.5 mentions 20000 candidates; the main-text within-prompt construction is chosen. |
| Model | Three width-64 GELU layers, no output bias | Paper section 5.1, Appendix A.5, public MLP implementation. Width64 is an evaluated choice. |
| Acquisition | Last-layer Fisher, first-order gradient, top500 | Paper sections 3.1-3.3; frozen upstream dopt_rule with 1024-row accumulation. |
| Trainer | Adam lr0.001, 1000 full-batch updates | Public ARM-FI train_BT defaults. The exact experiment setting is unavailable, so this is not claimed as a paper-reported number. |
| Initialization | Import model module first; fresh seeded model every round | Shared task execution choice consistent with retraining the entire model. Paper does not specify restart versus warm start or its seed numbers. |
| Statistical repetition | Five fixed seeds | Paper Figure3 states five runs. Seed IDs are task choices. |
| Data and dev | Original quickstart embeddings; 33500 train,4188 dev,10 responses | User's existing 10GB data constraint. Independent 500-response author test embeddings are unavailable. |

Sources read: local ../paper.pdf; ARM-FI/model/bradleyterry.py and
model/rules_pre_paired.py at commit4634e168d46cf5109cd0dfe1ebaf6b34b6b3f3ba;
author infrastructure utils.py and README at
a4f1a4ab549b9b1a7fe701bcbdeefc843ddc8ad7. Infrastructure pair_annotate
does normalize rewards and defaults to annotation_quality0.1; its related
paper's step5_train_rms.py defaults to quality10 and 30 epochs with a different
network and batching. Neither establishes these active-paper experiment
parameters. Hence the earlier std-normalized quality10/30-update setting is
retained as historical evidence only, not carried into this protocol.

Changing labels, training and annotation budget invalidates all older science
caches. Their files remain intact. New cached runs are keyed by both protocol
and selector source hashes. Only new-protocol completed seeds are reusable.
Round checkpoints permit an interrupted new-protocol run to continue without
repeating completed rounds; fresh Adam is still created for each new round.

The frozen score stays (B - mean_error) / B, with B recalibrated on the new
random selector over all five seeds. Baseline scores zero. Until calibration
finishes, no historical numerical B is used and no candidate score is emitted.
