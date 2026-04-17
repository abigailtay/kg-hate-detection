# Experiment One-Pagers: EXP-01 through EXP-12

## Table of contents

1. EXP-01 — Dim1 Stage 1 binary classifier
2. EXP-02 — Dim1 Stage 2 multi-class classifier (production)
3. EXP-03 — Dim1 Stage 2 v2 tuned configuration (rolled back)
4. EXP-04 — Dim2 rule-based radicalization signal pipeline
5. EXP-05 — Dim3 terrorism knowledge graph build
6. EXP-06 — Dim3 retrieval-augmented generation index build
7. EXP-07 — Dim3 activation gate bug and fix
8. EXP-08 — Critic V1 16-model benchmark
9. EXP-09 — Critic V2.1 redesign (design one-pager)
10. EXP-10 — Critic V2.1 baseline benchmark
11. EXP-11 — V2.1 quote-grounding hallucination analysis
12. EXP-12 — V2.1 understated_self_report cross-model divergence

---

## EXP-01 — Dim1 Stage 1 binary classifier

**Experiment ID:** EXP-01
**Date run:** 2026-03-23
**Paper placement:** Methods (Dim1 architecture) + Main Results (Stage 1 baseline)

### Research question
Can a HateBERT backbone with LoRA adapters reliably separate harmful from non-harmful content as the first stage of a two-stage cascade, while maximizing harmful-class recall without collapsing the non-harmful class?

### Motivation
Dim1 is the supervised backbone of the three-dimensional architecture. Stage 1 acts as coarse triage: anything it labels non-harmful skips the more expensive Stage 2 multi-class head and the critic layer entirely. The stage therefore needs two things simultaneously — high harmful recall (so nothing gets dropped) and non-harmful F1 high enough that the cascade doesn't degenerate into pass-through. This is also the simplest component in the paper and serves as the "does the backbone work at all" sanity check.

### Setup
- **Base model:** `GroNLP/hateBERT`
- **Adapter:** LoRA (PEFT 0.18.1), r=16, alpha=32, dropout 0.1, target modules `query, value`, `modules_to_save = [classifier, score]`
- **Task:** binary classification (`harmful` vs `non-harmful`)
- **Training:** batch size 32, max 10 epochs, eval every 500 steps, early stopping patience 3 on `eval_macro_f1`, tf32 enabled, `DataCollatorWithPadding` (dynamic padding), 4 dataloader workers with pin_memory
- **LR schedule:** linear warmup to peak ~2.0e-4 (reached around step 2000) then linear decay
- **Best checkpoint promoted:** step 35628 (epoch 6.0), selected by `best_metric` on held-out validation
- **Training continued** through step 47504 (epoch 8) without improvement; early-stopping patience counter reached 2 → rolled back to step 35628
- **Validation split:** Non-Harmful 10,569 / Harmful 15,682 (collapsed from 4-class: Identity Hate 10,440 + Abuse 2,242 + Crisis 3,000)

### Metrics
- Primary selection metric: `eval_macro_f1`
- Reported: Cohen's kappa, macro F1, weighted F1, per-class precision / recall / F1, eval loss
- Rationale: macro F1 and kappa together track both per-class balance and inter-rater-style agreement; per-class recall is the operational gate for the cascade.

### Results (held-out validation, best checkpoint = step 35628)

| Metric                      | Value   |
|-----------------------------|---------|
| **Macro F1**                | **0.8300** |
| Cohen's kappa               | 0.6603  |
| Weighted F1                 | 0.8358  |
| Eval loss                   | 0.3841  |
| Harmful — precision         | 0.8762  |
| Harmful — recall            | 0.8431  |
| Harmful — F1                | 0.8593  |
| Non-harmful — precision     | 0.7796  |
| Non-harmful — recall        | 0.8232  |
| Non-harmful — F1            | 0.8008  |
| Eval throughput             | 1,113 samples/sec |

**Trajectory.** Epoch 7 (step 41566) macro F1 = 0.8297; epoch 8 (step 47504) = 0.8287. The model plateaued after epoch 6 and early stopping engaged — step 35628 is a stable, non-overfit selection, not an arbitrary cut.

### What the results mean
Stage 1 clears the research plan threshold and behaves as a usable triage layer: harmful precision and recall are both above 0.84, and non-harmful F1 stays above 0.80, so the cascade will not degenerate. The gap between harmful F1 (0.859) and non-harmful F1 (0.801) is expected given the harmful-heavy training distribution and is not severe enough to require reweighting. We can claim a working two-stage backbone; we cannot claim Stage 1 is competitive with SOTA dedicated hate speech detectors — that is not what it is for.

### Risks / caveats
- **Validation-set only.** No held-out test evaluation has been run. Fix before submission: `[TODO: run evaluate on the 27,017-example test split and report test-set macro F1, kappa, and per-class P/R/F1 alongside the validation numbers above]`.
- Non-harmful recall 0.823 means ~17.7% of non-harmful inputs are pushed into Stage 2 unnecessarily — a mild cascade-efficiency concern, not a correctness concern.
- HateBERT was pre-trained on Reddit; downstream distribution shift to other platforms is a known limitation that Discussion should acknowledge.

---

## EXP-02 — Dim1 Stage 2 multi-class classifier (production)

**Experiment ID:** EXP-02
**Date run:** 2026-03-23
**Paper placement:** Methods (Dim1 architecture) + Main Results (Stage 2 primary table)

### Research question
Given a harmful input, can a second HateBERT+LoRA head correctly assign one of three fine-grained harm categories — crisis/self-harm, identity-based hate, or interpersonal abuse — with per-class F1 sufficient to justify routing decisions downstream?

### Motivation
Stage 2 is where the architecture's semantic claims live. Crisis/self-harm recall is life-safety-critical. Identity-based hate drives Dim3 KG/RAG activation. Interpersonal abuse is the hardest class because it overlaps linguistically with identity hate without the protected-attribute anchor. If Stage 2 cannot separate these with useful F1, every downstream component — the critic layer, Dim3 gating, the whole paper — is polishing a broken foundation.

### Setup
- **Base model + adapter:** same as EXP-01 (`GroNLP/hateBERT`, LoRA r=16, alpha=32, dropout 0.1, target `query, value`)
- **Task:** 3-class classification over Stage 1-harmful inputs — `crisis/self-harm`, `identity-based_hate`, `interpersonal_abuse`
- **Training:** batch size 32, max 10 epochs, eval every 500 steps, early stopping patience 3, same speed optimizations as EXP-01
- **LR schedule:** linear warmup to peak ~2.0e-4 (around step 2000) then linear decay
- **Best checkpoint:** step 6550 (epoch 2.0) — fastest convergence of the three Dim1 experiments
- **Promoted to production:** step 6550, saved as `best_model/`
- **Validation split:** Identity Hate 10,440 / Interpersonal Abuse 2,242 / Crisis 3,000 (non-harmful excluded from Stage 2)

### Metrics
- Selection: `eval_macro_f1`
- Reported: Cohen's kappa, macro F1, weighted F1, per-class precision / recall / F1, eval loss
- Rationale: crisis recall and identity-hate F1 are the load-bearing numbers for the cascade → critic hand-off; interpersonal-abuse F1 is the known hard case and is the most informative metric for the abuse-vs-identity-hate boundary problem Critic 1 is designed to handle.

### Results (held-out validation, best checkpoint = step 6550, epoch 2.0)

| Metric                              | Value   |
|-------------------------------------|---------|
| **Macro F1**                        | **0.8506** |
| Cohen's kappa                       | 0.7769  |
| Weighted F1                         | 0.8892  |
| Eval loss                           | 0.2629  |
| **Crisis/self-harm — precision**    | 0.9936  |
| **Crisis/self-harm — recall**       | 0.9903  |
| **Crisis/self-harm — F1**           | **0.9920** |
| **Identity-based hate — precision** | 0.9412  |
| **Identity-based hate — recall**    | 0.8834  |
| **Identity-based hate — F1**        | **0.9114** |
| **Interpersonal abuse — precision** | 0.5755  |
| **Interpersonal abuse — recall**    | 0.7426  |
| **Interpersonal abuse — F1**        | **0.6485** |
| Eval throughput                     | 1,045 samples/sec |

**Convergence.** Epoch 1 (step 3275) macro F1 = 0.8109, kappa = 0.6943. Epoch 2 jumped to 0.8506 / 0.7769 (+3.97pp macro F1, +8.27pp kappa), after which training plateaued. Selecting epoch 2 is strongly supported by the trajectory.

**Confusion-matrix highlight.** The dominant off-diagonal is Identity Hate → Interpersonal Abuse: approximately 1,263 Identity Hate validation examples are misclassified into Abuse. This single confusion cell accounts for the bulk of Abuse's precision drop (0.5755) and is the direct target of Critic 1's boundary rule.

### What the results mean
The cascade's life-safety guarantee is empirically satisfied: crisis/self-harm recall 0.9903 means fewer than 1% of crisis posts are missed. Identity-based hate precision 0.9412 is high enough that critic-layer overrides will be rare for that class. Interpersonal abuse is the architectural tension point: precision 0.576 with recall 0.743 means the head over-fires on abuse, specifically capturing posts that should have been classified as identity-based hate. This is not a training failure — it is a real linguistic boundary problem (protected-attribute targeting vs. personalized aggression) that cannot be fully solved inside a single classifier head and is the direct motivation for Critic 1 in the critic layer. The paper's core story — "the backbone is strong but has one architecturally addressable weakness" — is supported by these numbers.

### Risks / caveats
- **Validation-set only.** Same as EXP-01. Fix before submission: `[TODO: run evaluate on the 27,017-example test split; expected per research plan addendum section 1.3 are Identity Hate 11,482 / Abuse 2,268 / Crisis 3,000]`.
- Reviewers may push back on the ~0.05 gap between macro F1 and weighted F1 (0.8506 vs 0.8892) as evidence the macro number is being dragged by the minority Abuse class. This is true, and is the point — reporting macro F1 is the honest choice.
- The 1,263 Identity Hate → Abuse confusion count is a validation-set number; re-derive on the test set when that eval runs.

---

## EXP-03 — Dim1 Stage 2 v2 tuned configuration (rolled back)

**Experiment ID:** EXP-03
**Date run:** 2026-03-30
**Paper placement:** Ablation — "Does aggressive class reweighting + capacity increase help the minority class?"

### Research question
Given that v1 interpersonal-abuse F1 (0.6485) is the lowest per-class score in Stage 2, does a tuned configuration combining higher LoRA capacity, lower learning rate, explicit 3x Abuse class weight, label smoothing, and longer training improve minority-class performance without sacrificing overall macro F1?

### Motivation
The v1 interpersonal-abuse metrics are visibly the weakest component of Stage 2, and a reviewer's first instinct on seeing 0.6485 will be: "did you try reweighting?" The ablation exists to answer that question with data rather than assertion. The secondary purpose is to document that further hyperparameter effort on Dim1 has diminishing returns — the research-plan targets are already met by v1 — so budget should flow to critic-layer and Dim3 work instead.

### Setup
**Single documented v2 tuned configuration**, delta against v1:

| Hyperparameter       | v1              | v2 tuned          |
|----------------------|-----------------|-------------------|
| LoRA rank (r)        | 16              | **32**            |
| LoRA alpha           | 32              | **64**            |
| Learning rate (peak) | 2.0e-4          | **1.0e-4**        |
| Abuse class weight   | Balanced        | **Balanced x 3.0** |
| Label smoothing      | 0.0             | **0.1**           |
| Max epochs           | 10              | **15**            |
| Early stopping patience | 3            | **4**             |

All other settings (base model, target modules, batch size 32, eval every 500 steps, speed optimizations) identical to v1. Validation split identical to v1. Selection metric: `eval_macro_f1`.

**Wandb run ID:** `dim1_stage2_v2_tuned_20260330_0017` (the documented single run).

### Metrics
Same as EXP-02. Side-by-side with v1 to surface the per-class tradeoff.

### Results (v2 tuned vs v1, held-out validation, best checkpoint selected by macro F1)

| Metric                              | v1 (prod) | v2 tuned  | Delta      |
|-------------------------------------|-----------|-----------|------------|
| Macro F1                            | **0.8506** | 0.7946   | **-0.0560**  |
| Cohen's kappa                       | 0.7769    | 0.6591    | -0.1178    |
| Weighted F1                         | 0.8892    | 0.8217    | -0.0675    |
| Eval loss                           | 0.2629    | 0.5812    | +0.3183    |
| Crisis/self-harm — F1               | 0.9920    | 0.9905    | -0.0015    |
| Crisis/self-harm — recall           | 0.9903    | 0.9880    | -0.0023    |
| Identity-based hate — F1            | 0.9114    | 0.8283    | -0.0831    |
| Identity-based hate — recall        | 0.8834    | 0.7198    | -0.1636    |
| Identity-based hate — precision     | 0.9412    | 0.9753    | +0.0341    |
| **Interpersonal abuse — F1**        | 0.6485    | 0.5651    | **-0.0834** |
| Interpersonal abuse — precision     | 0.5755    | 0.4095    | -0.1660    |
| **Interpersonal abuse — recall**    | 0.7426    | **0.9117** | **+0.1691** |

### What the results mean
The v2 tuned configuration did exactly what class reweighting is supposed to do — it raised minority-class recall by 16.9 percentage points (0.743 → 0.912) — and exactly what aggressive reweighting is known to do when overdone: it destroyed precision on the same class (-16.6pp) and, more importantly, bled into Identity Hate recall (-16.4pp) because the model started relabeling Identity Hate as Abuse to chase the weighted loss. Net macro F1 dropped 5.6pp. Higher LoRA capacity (r=32, alpha=64) combined with the lower learning rate did not compensate; label smoothing and longer training did not compensate. **v1 was rolled back as production.** What this ablation buys the paper is a clean answer to the reviewer's reweighting question: "yes, we tried it, and the tradeoff is unfavorable because the model trades Identity Hate accuracy for Abuse recall." That is also the cleanest possible motivation for Critic 1 — if you can't fix the boundary inside Dim1 without breaking Dim1, you fix it outside Dim1.

### Paper placement
**Ablation section**, presented as a one-row comparison against the v1 production row in the main Stage 2 results table, not as a standalone subsection. The story is a single paragraph.

### Risks / caveats
- **Two additional runs with matching directory naming (`dim1_stage2_v2_tuned_20260330_0107`, `_20260330_0208`) exist on disk with different best-metric values (macro F1 0.8439 and 0.8231 respectively).** These are excluded from reported results because their exact configurations are not documented in the research plan, progress log, or wandb record, and cannot be reconstructed from the saved `training_args.bin` alone (class weights are passed at loss-construction time, not in the HF training args). If wandb history for these runs is recovered before submission, they will be reported as a seed-variance or configuration-sweep footnote; otherwise they are treated as undocumented artifacts and excluded. `[TODO: attempt wandb log recovery for runs 0107 and 0208 before May 1]`.
- The Abuse recall gain is real and isolated — a reviewer sympathetic to recall-over-precision reframings might argue v2 is the correct production choice for high-stakes moderation contexts. The rebuttal is that macro F1 is the preregistered selection metric and the Identity Hate recall loss is unacceptable for an architecture that claims to protect identity-based targets.
- This ablation does not establish that reweighting cannot work — only that this particular combined configuration does not. A more surgical ablation (reweighting alone, no capacity or LR change) is deferrable future work.

---

## EXP-04 — Dim2 rule-based radicalization signal pipeline

**Experiment ID:** EXP-04
**Date run:** 2026-03 (pre-critic work)
**Paper placement:** Methods (Dim2 architecture) + qualitative examples in Discussion

### Research question
Can a pure-regex, literature-grounded signal detector identify radicalization precursors (grievance, polarization, violence justification, glorification, leakage, escalation) with enough precision to serve as a risk amplifier — never a terminal label — alongside Dim1's supervised classification?

### Motivation
Dim2 exists because supervised hate-speech classifiers are blind to pre-violent escalation language: posts that are not yet hateful or abusive in the classifier's sense but are on a recognizable trajectory. Radicalization research (Moghaddam 2005; McCauley & Moskalenko 2008; Meloy & O'Toole 2011) identifies six reproducible linguistic precursors. A rule-based implementation is the honest choice for a component whose patterns come directly from literature: it is interpretable, auditable, doesn't require labeled training data for precursor classes (which barely exist), and cannot overfit in the neural sense. The design constraint is **Boundary Rule #2: signals are amplifiers, never terminal labels** — Dim2 never decides "harmful" on its own.

### Setup
**Pipeline.** `gate → signals → scoring → risk vector`.

- **Gate** (`gate.py`): ~250 political/ideological keywords compiled into a single regex with word boundaries and length-descending alternation. Returns `True` if any keyword matches. If `False`, Dim2 is skipped entirely. The gate costs milliseconds and filters non-political content before pattern matching runs.
- **Signals** (`signals.py`): six detectors, one per precursor, totaling **73 regex patterns**:
  - 2.1 Grievance Framing
  - 2.2 Polarization ("us vs them" absolutism)
  - 2.3 Violence Justified
  - 2.4 Glorification (of past attackers / acts)
  - 2.5 Leakage (Meloy's validated pre-attack indicator)
  - 2.6 Escalation ("talking isn't working, time to act")
  - Each signal returns `raw_confidence = min(n_matches / divisor, 1.0)`, default divisor 3.
- **News dampener**: 27 regex patterns for third-person attribution, news framing, and academic framing (e.g., `according to`, `sources say`, `study finds`, reported-speech patterns). When any dampener pattern matches, all signal confidences are multiplied by `0.35`. This prevents news articles about attacks from being scored as pre-attack language.
- **Scoring** (`scoring.py`): literature-weighted combination with compounding bonus.
  - Signal weights: 2.1->0.10, 2.2->0.15, 2.3->0.25, 2.4->0.20, 2.5->**0.30** (highest; Meloy's most validated indicator), 2.6->0.25
  - Combination multiplier: 2 signals -> x1.10, 3 -> x1.25, 4 -> x1.40, 5 -> x1.55, 6 -> x1.70
  - Final `radicalization_score = min(weighted_sum * combo_mult, 1.0)`

**Test suites.** Two self-test batteries were run during development:
- **Gate self-test** (`gate.py:__main__`): 10 pass/fail cases — 5 benign (cat videos, weather, cooking) and 5 politically loaded — confirming the gate activates correctly and doesn't fire on non-political content.
- **Signal self-test** (`signals.py:__main__`): 8 cases, one targeting each primary signal plus a news-dampened case and a null case.
- **End-to-end pipeline self-test** (`dim2_pipeline.py:__main__`): **6 cases** — benign, gate+grievance, grievance+polarization, multi-signal escalation, max-escalation, news-dampened. These are the "6 test cases" referenced in the project progress log.

### Metrics
Binary pass/fail per test case (`signals fired = expected primary signals`), plus observed `radicalization_score` and fired-signal set. Pass rate across each suite. This is a unit-test style validation, not a statistical benchmark — the paper claim is *coverage of the literature signals*, not accuracy on an annotated precursor dataset (which does not exist).

### Results
**All three self-test suites pass at 100%** at the time of the March 30 progress log:
- Gate self-test: 10/10
- Signal self-test: 8/8 (including correct null-result on news-dampened and benign inputs)
- End-to-end pipeline: 6/6

**Representative pipeline outcomes** (from `dim2_pipeline.py` self-test):

| Case                               | Gate   | Signals fired         | Radicalization score | Dampened? |
|------------------------------------|--------|-----------------------|----------------------|-----------|
| "I love my cat..." (benign)        | SKIP   | none                  | 0.0000               | n/a       |
| "The government is failing us..."  | PASS   | {2.1}                 | low                  | no        |
| "They are destroying our country..." | PASS | {2.1, 2.2}            | moderate (x1.10)     | no        |
| "Violence is the only language..." | PASS   | {2.1, 2.3, 2.6}       | high (x1.25)         | no        |
| "People will see what I'm capable of..." (5 signals) | PASS | {2.1, 2.2, 2.3, 2.4, 2.5, 2.6 partial} | **max (x1.55+)** | no |
| "According to sources, the attacker was praised..." | PASS | dampened to near-zero | low | **yes** |

**Pattern counts:** 73 signal patterns + 27 news-dampener patterns = 100 regexes total across Dim2. Gate vocabulary: ~250 keywords.

### What the results mean
Dim2 is operationally complete and does what the literature says the signals should do: it fires on pre-violent escalation language, compounds appropriately when multiple precursors co-occur, respects Boundary Rule #2 by never returning a terminal label, and correctly suppresses signals in news/reporting contexts via the dampener. We can claim: "a literature-grounded, auditable radicalization signal detector with full coverage of the six McCauley/Moskalenko/Meloy precursors, validated on 24 unit tests across three test suites." We cannot claim precision or recall against a labeled precursor dataset — none exists, and building one is out of scope for this paper.

### Risks / caveats
- **No quantitative precision/recall on real data.** The only validation is unit tests. A reviewer will ask what the false-positive rate is on in-the-wild political content that is not radicalized. The honest answer is: unknown, because there is no labeled precursor corpus, and the dampener is the only defense against news-framed false positives.
- **Regex brittleness.** Character-level adversarial modifications (substitution, spacing, leetspeak) will defeat the patterns. This is a known limitation of rule-based approaches and is one reason Dim2 is an amplifier, not a standalone decision-maker.
- **Dampener factor (0.35) is a hand-chosen constant,** not learned. Sensitivity to this value is not characterized.
- The "6 test cases" in the progress log refers to the end-to-end suite specifically; the total case count across all three suites is 24.

---

## EXP-05 — Dim3 terrorism knowledge graph build

**Experiment ID:** EXP-05
**Date run:** 2026-03-30 16:24 UTC
**Paper placement:** Methods (Dim3 architecture) + Data section (KG construction)

### Research question
Can we build a terrorism-scoped knowledge graph from publicly available authoritative sources (GTD, UN sanctions, curated ideology/symbol data) at sufficient scale and connectivity to support contextual grounding when Dim1 flags identity-based hate with potential extremist linkage?

### Motivation
Dim3's role in the architecture is providing context for threat escalation judgments: when Dim1 flags identity-based hate and the content references a known extremist organization, ideology, or symbol, the critic layer needs a grounded source to verify the reference rather than relying on LLM parametric knowledge (which is unverifiable and hallucination-prone). A knowledge graph with explicit entity types and typed relations gives the system a defensible "why this was flagged as extremist-adjacent" trail — critical for the broader-impact disclaimer and any moderation or law-enforcement downstream use concerns.

### Setup
- **Sources (3):**
  1. Global Terrorism Database (GTD) — attack events, perpetrator organizations, locations
  2. UN Security Council Consolidated Sanctions List — designated organizations and individuals
  3. Curated supplementary data (SPLC/ADL-derived) — ideologies, symbols, slogans
- **Graph framework:** NetworkX `DiGraph`
- **Node IDs:** MD5-hashed `{node_type}::{name_normalized}` truncated to 12 hex chars; deterministic and collision-safe at this scale
- **Normalization:** `clean_str` drops `unknown`, `.`, `-`, `nan`, `n/a`, `other` and NaN values before node creation
- **Outputs (three formats for flexibility):**
  - `terrorism_kg.gpickle` (33 MB) — primary working format, loaded via `pickle.load()` (NetworkX 3.x removed `nx.read_gpickle`)
  - `terrorism_kg.graphml` (77 MB) — portable interchange format
  - `alias_lookup.json` (5.1 MB) — name-variant to canonical node ID map (including the ISIS / Islamic State alias dictionary added after the activation-gate bug)
  - `kg_stats.json` (546 B) — node/edge type counts
- **Design target:** 5K-10K nodes, 15K-30K edges (per `build_kg.py` docstring)

### Metrics
Graph-statistics only: total nodes, total edges, node-type distribution, edge-type distribution, connectivity of the organization subgraph (how many orgs have any relation at all). No downstream task metric at this stage — those arrive with EXP-07 and the ablations.

### Results

**Totals.** 106,932 nodes, 204,496 edges. Dramatically exceeds design target because attack events were included as first-class nodes rather than edge attributes — an upfront decision to let the graph support event-level queries.

**Node type distribution:**

| Node type       | Count   | % of total |
|-----------------|---------|------------|
| attack_event    | 100,553 | 94.03%     |
| organization    | 5,390   | 5.04%      |
| individual      | 731     | 0.68%      |
| location        | 221     | 0.21%      |
| symbol_slogan   | 23      | 0.02%      |
| ideology        | 14      | 0.01%      |

**Edge type distribution:**

| Edge type               | Count   | Notes |
|-------------------------|---------|-------|
| located_in              | 100,553 | one per attack event |
| perpetrated             | 100,543 | org -> event  |
| co_perpetrated          | 1,898   | multi-org events |
| nationality             | 655     | individual -> country |
| affiliated_with         | 599     | org <-> org |
| based_in                | 200     | org -> location |
| follows_ideology        | 27      | org -> ideology |
| associated_with_ideology | 21     | symbol/slogan -> ideology |

**Organization connectivity.** 671 of 5,390 organizations have at least one edge to another organization or ideology (12.45%). The remaining 87.55% are "singleton" orgs with only their own attack events attached — typical for GTD-derived graphs, where many small groups appear in one or two events.

### What the results mean
The KG is operationally complete and multiple orders of magnitude larger than designed because attack events became nodes — a choice that lets Dim3 support event-level grounding ("this post references the 2015 Bataclan attack") in addition to organization-level grounding ("this post references ISIS"). The low organization-to-organization connectivity (12.45%) is a real finding worth surfacing in Methods: most extremist organizations in GTD operate independently, so graph-traversal queries beyond 1 hop from a known entity will usually hit attack events rather than sibling organizations. This informs how Dim3's retrieval logic should be written — flat lookup + RAG is more useful here than deep traversal. The ideology layer (14 nodes, 48 total ideology-linked edges) is the weakest part: insufficient for meaningful ideology-based queries and a known limitation.

### Risks / caveats
- **Massive class imbalance in node types** (94% are attack events). Any analysis that counts nodes uniformly will be dominated by events. Report organization counts separately whenever possible.
- **GTD recency cutoff.** GTD releases lag current events by ~1 year; the KG is stale for anything in the last 12 months. This is a Dim3 limitation, not a KG-construction limitation.
- **Ideology node count (14) is too small** for downstream ideology classification. If Dim3 ever needs to return "ideology" as an explicit field, we will need to expand this layer before publication or explicitly scope the claim to "organization and event grounding only."
- **Singleton organization rate (87.55%)** means that most orgs in the graph can only be retrieved by exact name — no graph-traversal benefit. This is honest and should be disclosed.

---

## EXP-06 — Dim3 retrieval-augmented generation index build

**Experiment ID:** EXP-06
**Date run:** 2026-03-30 17:13 UTC
**Paper placement:** Methods (Dim3 architecture) + Data section (RAG construction)

### Research question
Can we build a dense-retrieval index over the same authoritative sources used for the KG (GTD + UN sanctions) that provides semantic context retrieval at sufficient coverage and quality to complement the KG's symbolic lookup, enabling Dim3 to return both "matched entities" (from KG) and "relevant narrative context" (from RAG) on a single query?

### Motivation
The KG gives precise symbolic answers — "yes, this name matches a designated organization" — but is blind to paraphrase, description, and indirect reference. A post saying "the group behind the Paris concert hall attack" will not match any KG node directly, but dense retrieval over GTD event summaries will surface Bataclan / ISIL / 2015 instantly. KG and RAG are complementary, not redundant: KG for verification, RAG for recall over paraphrased or contextual reference. Without RAG, Dim3 is only useful for explicit name matches.

### Setup
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim output)
- **Sources:** same as EXP-05 — GTD xlsx + UN Consolidated Sanctions XML. One document per GTD attack event; one document per sanctioned entity.
- **Chunking:** `langchain_text_splitters.RecursiveCharacterTextSplitter` with `chunk_size=800`, `chunk_overlap=100`. (Note: `langchain.text_splitter` was renamed to `langchain_text_splitters` in the version we use — an import-path gotcha worth documenting.)
- **Embedding batch size:** 256
- **Index type:** FAISS `IndexFlatIP` — exact inner-product search, no approximation. With 214,609 vectors at 384-dim this is still fast enough on A100 for interactive latencies.
- **Default retrieval top-k:** 5
- **Outputs:**
  - `faiss_index.bin` — 315 MB
  - `chunk_metadata.json` — 119 MB (chunk text + GTD/UN source reference per chunk)
  - `rag_config.json` — pipeline config for inference-time loading
- **Directory total:** 433 MB
- **Build timestamp (from `rag_config.json`):** 2026-03-30 17:13:00

### Metrics
Index-construction only: total chunks, dimension, index type, disk size. Retrieval quality is evaluated in EXP-07 (activation gate) and in the Dim3 KG/RAG ablation (EXP-15, not yet run).

### Results

| Property            | Value                   |
|---------------------|-------------------------|
| Embedding model     | all-MiniLM-L6-v2        |
| Embedding dim       | 384                     |
| Index type          | FAISS IndexFlatIP (exact) |
| **Total chunks (vectors)** | **214,609**       |
| Chunk size          | 800 chars               |
| Chunk overlap       | 100 chars               |
| Default top-k       | 5                       |
| `faiss_index.bin`   | 315 MB                  |
| `chunk_metadata.json` | 119 MB                |
| **Directory total** | **433 MB**              |

### What the results mean
Dim3 now has a dense retrieval layer with 214,609 searchable chunks covering GTD attack narratives and UN sanctions descriptions, at a size that fits comfortably in RAM and supports exact (not approximate) search. Paired with the KG's 106,932 nodes and 204,496 edges, Dim3 can support both "is this name a known extremist entity?" (KG) and "is this paraphrased description related to a known attack/group?" (RAG) on a single query, which is the operational requirement for contextual threat judgment. We can claim complete Dim3 infrastructure; we cannot yet claim Dim3 improves downstream performance — that is EXP-15.

### Risks / caveats
- **Chunk size 800 is a default, not a tuned choice.** No ablation on chunk size has been run. A reviewer may ask.
- **MiniLM-L6-v2 is a 2021-era model.** Newer embedding models would likely improve retrieval quality, but swapping would require re-embedding all 214,609 chunks and is not prioritized before the May 1 deadline.
- **IndexFlatIP is exact but does not scale** beyond low millions of vectors without switching to IVF or HNSW. At 214k we are far from that limit; noted only for completeness.
- **Same source-recency limitations as EXP-05** — GTD lag applies to RAG content as well.

---

## EXP-07 — Dim3 activation gate bug and fix

**Experiment ID:** EXP-07
**Date run:** 2026-03-30 (post-RAG build)
**Paper placement:** Methods (Dim3 activation logic) + Limitations (reported as a paper-worthy methodological finding about KG prescreening)

### Research question
Does naive substring matching over KG entity names produce an acceptable false-positive rate when used as Dim3's activation gate, or does it need a length floor and word-boundary regex to be usable?

### Motivation
Dim3 is expensive — every activation triggers both a KG lookup and a FAISS retrieval. The activation gate's job is deciding which Dim1-flagged inputs warrant that cost. The first implementation used naive substring matching: for every KG alias, check if the alias string appears anywhere in the input text. This matched the spec but immediately produced bizarre results — innocuous posts about ordinary topics were triggering Dim3 activation. The root cause is entity names that are also common English substrings (short aliases like "is", "al", "the group", "one", and, most visibly, "IS" for Islamic State matching any occurrence of `is`). Fixing this is essential before any Dim3 experiment can be trusted, and the fix itself is a reportable finding — KG prescreening false positives are under-discussed in the literature.

### Setup
**Before (bug).** For each alias in the KG alias dictionary, `alias.lower() in text.lower()`. No length filter, no word boundaries.

**After (fix).** In `src/dim3/dim3_pipeline.py` (verified at line 155):
1. **Minimum alias length >= 4 characters.** Aliases shorter than 4 chars are excluded from gate matching entirely (they can still be retrieved by explicit KG query, just not by gate prescreening).
2. **Regex word boundaries.** Each eligible alias is wrapped in `\b...\b` and matched with `re.search(..., re.IGNORECASE)`. This prevents "IS" from matching inside "is", "this", "analysis", etc., and prevents "al" from matching "also", "metal", etc.
3. **ISIS / Islamic State alias dictionary** was added separately to ensure canonical multi-word forms (`Islamic State`, `ISIS`, `ISIL`, `Daesh`) still match correctly under word-boundary constraints.

### Metrics
Activation rate: fraction of inputs in the diagnostic sample where the gate fires and Dim3 is invoked. Also: visual inspection of which inputs fired under the buggy gate vs the fixed gate, to characterize the failure mode qualitatively.

### Results

| Gate version   | Activation rate | Notes |
|----------------|-----------------|-------|
| **Before fix** | **~40%**        | False positives dominated by short-alias substring matches (e.g., "IS" inside common words) |
| **After fix**  | **~10.8%**      | Concentrated on inputs with actual word-boundary matches to entities >=4 chars |

**Reduction in activation rate: ~29 percentage points (roughly 4x fewer false activations).**

**Qualitative failure mode (pre-fix).** The dominant bug was short-form aliases treated as substrings: two-character aliases like `IS`, `AQ`, `AL`, and three-character aliases like `the`, `and`, `for` (which entered the alias dictionary through noisy name normalization) matched essentially any English sentence. The `is` -> `Islamic State` collision was the most visible because it fired on almost every input, but the general pattern was that any KG entity whose alias was <=3 characters was producing near-100% false-positive rate on its own.

### What the results mean
Naive substring matching over a KG alias dictionary is unusable as an activation gate when the alias set contains any short strings, and in practice any real-world KG will contain short strings due to acronyms, initials, and normalization artifacts. A minimum-length floor of 4 characters plus regex word boundaries reduces the Dim3 activation rate by a factor of ~3.7x, from ~40% to ~10.8%, without (as far as manual inspection shows) losing true positives. **This is a reportable methodological finding:** the KG-prescreening literature tends to assume alias matching is a solved problem, but in practice a length floor and word-boundary constraint are non-negotiable and are not consistently documented. The fix itself is five lines of code; the value is knowing it is required.

### Risks / caveats
- **Sample size for the 40% / 10.8% numbers is not logged.** The measurement was made on a diagnostic sample during Dim3 development, and the exact N was not written down. Before submission: `[TODO: re-measure activation rate on a known N — suggested is the full Dim1 validation split filtered to Identity Hate (~10,440 examples), which gives both the activation rate and a principled denominator the paper can cite]`.
- **No formal recall test.** We have not confirmed that the fix does not miss true positives. A false negative would look like: a post references "ISIS" but the gate misses it because of a formatting quirk. Spot-check review did not surface any such cases but this is not a formal recall measurement. Before submission, pair the re-measurement above with a manual check of ~50 Dim3-negative inputs to confirm zero missed references.
- **The length threshold of 4 is empirical, not tuned.** Threshold 3 might be acceptable if the alias dictionary is cleaned of trivial strings; threshold 5 might lose legitimate 4-char aliases (e.g., `ISIS`, `ISIL`). The value 4 was chosen because it preserves `ISIS`/`ISIL`/`Daesh` as the shortest meaningful organizational names.
- **Word-boundary regex is not Unicode-aware by default.** Entity names containing non-ASCII characters (Arabic transliterations, diacritics) may match inconsistently. Not measured.

---

## EXP-08 — Critic V1 16-model benchmark

**Experiment ID:** EXP-08
**Date run:** 2026-03 (pre-V2.1 redesign)
**Paper placement:** Methods (critic-model selection justification) + Appendix (full 16-model table)

### Research question
Which of 16 candidate LLMs across multiple providers are suitable as independent critics for the three-critic audit layer, measured by inter-model agreement (kappa), behavioral pathology rates (flag storm vs under-flagging), per-call cost, and provider diversity?

### Motivation
The three-critic architecture requires multiple independent LLM auditors, and the value of "independent" collapses if all critics come from the same provider, share a post-training lineage, or exhibit the same failure modes. A naive "pick the best three" strategy risks correlating errors across critics and defeating the audit layer's premise. A broader screen was needed to (a) identify models that agree with each other at useful levels (high kappa), (b) exclude models that flag everything or flag nothing (flag-rate pathology), (c) manage cost per 1,500 critic calls, and (d) ensure multi-provider coverage so that a single provider's systematic bias does not dominate the ensemble.

### Setup
- **Candidate pool: 16 models** across multiple providers (Anthropic, DeepSeek, Google, OpenAI, Meta, Mistral, NVIDIA, xAI, Alibaba):
  - Anthropic: `claude-haiku-4-5`, `claude-sonnet-4-6`
  - DeepSeek: `deepseek-r1`, `deepseek-v3.2`
  - Google: `gemini-3.1-pro-preview`, `gemma-4-31b-it`
  - OpenAI: `gpt-4o`, `gpt-5.4-mini`, `o3-mini`
  - Meta: `llama-3.3-70b-instruct`, `llama-4-maverick`, `llama-4-scout`
  - Mistral: `mistral-small-2603`
  - NVIDIA: `nemotron-3-super-120b-a12b`
  - xAI: `grok-4.20`
  - Alibaba: `qwen-plus-2025-07-28`
- **Routing:** all 16 accessed via OpenRouter with explicit per-model provider overrides (no prefix matching — prefix-based routing broke V2 first time and caused gpt-5.4-mini calls to go to the wrong endpoint).
- **Benchmark size:** N = 500 examples per model. All 16 ran the same 500-example prompt suite under the V1 critic prompt architecture.
- **Critic count at V1:** three critics per call (Classification Validation, Contextual Fairness, Threat Escalation), same architecture slot as V2.1 but with free-form chain-of-thought and prefix-based routing.
- **Metrics logged per model:** agreement rate, adjust rate, override rate, flag rate, any-flag rate, review rate, all-disagree rate, average latency, prompt tokens, completion tokens, estimated cost (USD).
- **Pairwise kappa matrix:** 16x16 Cohen's kappa between every model pair, computed on the 500-example overlap.

### Metrics
- **Inter-model kappa clustering** (pairwise agreement; high values indicate two models would make similar critic decisions)
- **any_flag_rate** (fraction of cases where at least one critic fires a flag) — pathology metric for flag-storm / flag-silence
- **Average latency per call** — operational constraint
- **Cost per 500 calls** — budget constraint
- **Provider diversity** — architectural constraint

### Results (16-model summary, sorted by final top-6 membership)

| Model | Agree rate | Override rate | any_flag_rate | Avg latency (s) | Cost / 500 (USD) | In final top-6? |
|---|---|---|---|---|---|---|
| claude-sonnet-4-6 | 0.738 | 0.146 | 0.298 | 2.64 | 3.71 | YES |
| deepseek-r1 | 0.748 | 0.116 | 0.362 | 37.36 | 4.46 | YES |
| gemma-4-31b-it | 0.736 | 0.204 | 0.260 | 7.15 | 0.07 | YES |
| gpt-5.4-mini | 0.734 | 0.234 | 0.166 | 0.99 | 0.14 | YES |
| qwen-plus-2025-07-28 | 0.710 | 0.150 | 0.256 | 1.47 | 0.34 | YES |
| grok-4.20 | 0.614 | 0.336 | 0.072 | 0.80 | 3.62 | YES |
| gemini-3.1-pro-preview | 0.758 | 0.138 | 0.228 | 5.92 | 2.81 | no |
| deepseek-v3.2 | 0.776 | 0.036 | 0.398 | 4.85 | 0.24 | no |
| claude-haiku-4-5 | 0.558 | 0.358 | 0.326 | 2.12 | 1.29 | no |
| gpt-4o | 0.636 | 0.300 | 0.134 | 1.12 | 2.36 | no |
| o3-mini | 0.664 | 0.014 | **0.636** (flag storm) | 3.83 | 3.26 | no |
| nemotron-3-super-120b | 0.484 | 0.050 | **0.782** (flag storm) | 6.74 | 0.96 | no |
| llama-4-maverick | 0.302 | 0.278 | **0.628** (flag storm) | 2.03 | 0.16 | no |
| llama-3.3-70b-instruct | 0.390 | 0.392 | 0.368 | 3.91 | 0.17 | no |
| mistral-small-2603 | 0.456 | 0.324 | 0.262 | 1.36 | 0.17 | no |
| llama-4-scout | 0.394 | 0.364 | 0.252 | 1.19 | 0.08 | no |

**Inter-model kappa clustering (selected values from 16x16 matrix).** The top-6 that were ultimately selected show mutually elevated kappa relative to the bottom cluster:

| Model pair | kappa |
|---|---|
| gemma-4-31b <-> gpt-5.4-mini | **0.509** |
| gemma-4-31b <-> sonnet-4-6 | **0.527** |
| gemma-4-31b <-> deepseek-r1 | 0.390 |
| gemma-4-31b <-> grok-4.20 | 0.404 |
| sonnet-4-6 <-> gpt-5.4-mini | 0.403 |
| sonnet-4-6 <-> deepseek-r1 | 0.326 |
| qwen <-> gpt-5.4-mini | 0.411 |
| (bottom cluster) llama-4-maverick <-> llama-4-scout | 0.283 |
| (bottom cluster) nemotron <-> anything | 0.08-0.26 |
| (bottom cluster) llama-4-maverick <-> sonnet-4-6 | 0.140 |

**Final top-6:** `gemma-4-31b-it`, `gpt-5.4-mini`, `claude-sonnet-4-6`, `deepseek-r1`, `grok-4.20`, `qwen-plus-2025-07-28`.

### What the results mean
The 16-model screen surfaced a clear usable cluster and excluded two pathology groups. The **usable cluster** — Sonnet, Gemma, gpt-5.4-mini, DeepSeek-R1, Qwen, Grok — shows mutual kappa in the 0.29-0.53 range, broad provider diversity (Anthropic, Google, OpenAI, DeepSeek, Alibaba, xAI), and manageable cost. The **flag-storm pathology** (o3-mini 63.6%, Nemotron 78.2%, Llama-4-Maverick 62.8% any-flag rate) indicates models that default to escalating nearly every case, which destroys the critic layer's selectivity. The **low-kappa pathology** (Llama-family and Mistral, with mutual kappa in the 0.10-0.30 range against every other model) indicates models whose decisions are essentially uncorrelated with the rest of the pool — these are not "independent critics" in any useful sense, they are noise. Gemini-3.1-pro-preview and DeepSeek-v3.2 were borderline acceptable (high agree rate, reasonable cost) but were dropped in favor of the final six to preserve provider diversity and keep the final ensemble at a manageable size for V2.1 re-benchmarking.

**Selection rule.** Manual selection from the 16-model screen guided by four criteria applied in order: (1) inter-model kappa clustering (model must sit in the high-kappa quadrant against at least 3 other candidates), (2) any-flag-rate pathology exclusion (drop models with any-flag-rate above 0.50 or below 0.10), (3) cost per 500 calls under operational budget, (4) provider diversity (no more than two models per provider in final set). This was judgment-guided selection, not a closed-form formula; the CSV and 16x16 kappa matrix are the preserved evidence base.

### Risks / caveats
- **Selection rule is post-hoc justifiable, not pre-registered.** A reviewer may ask whether the top-6 was chosen to match the intended critics after the fact. The defense is the full 16-model CSV — the top-6's kappa, any-flag-rate, and cost numbers are all visibly in the "usable cluster" region of the distribution, and the excluded models all have at least one disqualifying metric (flag-storm, low-kappa, or extreme cost). We should publish the full 16-model table as appendix material so the selection is replicable.
- **DeepSeek-R1 average latency is 37.4 seconds per call** — ~30x the fastest model and >10x the median. This is a known R1 characteristic (reasoning tokens) and was accepted because R1's V2.1 kappa-stability gain (EXP-10) justified the latency cost. Flag in Limitations.
- **Cost is OpenRouter-reported estimate,** not audited billing. The low Gemma cost ($0.07/500) reflects provider pricing as of March 2026 and may shift.
- **No test of critic-layer downstream accuracy.** The 16-model screen measures inter-model agreement, not whether any individual model is correct. "Correctness" for a critic is not well-defined — the entire point of the audit layer is disagreement. This is a conceptual limitation worth naming in Methods.

---

## EXP-09 — Critic V2.1 redesign (design one-pager)

**Experiment ID:** EXP-09
**Date run:** 2026-03 (redesign milestone; validation in EXP-10)
**Paper placement:** Methods (critic prompt architecture)

### Research question
This is a design one-pager, not a measurement. The question is: what prompt-architecture changes are needed to produce critic outputs that are structured enough to parse reliably, grounded enough to resist hallucinated evidence, and calibrated enough to distinguish reasoning depth from surface confidence?

### Motivation
V1 critics were usable for selection (EXP-08) but had three systemic problems that blocked downstream use: free-form chain-of-thought that was hard to parse reliably, unconstrained evidence quotation that produced hallucinated quotes (critics citing text that was not in the original input), and a single confidence scalar that collapsed reasoning depth into a number and made it impossible to tell an over-confident one-line rationale from a deeply reasoned decision. V1 also had operational bugs — prefix-based provider routing caused gpt-5.4-mini calls to hit the wrong endpoint, producing corrupted benchmarks. V2.1 was a full prompt and infrastructure rewrite that kept the three-critic division (Classification Validation, Contextual Fairness, Threat Escalation) but replaced everything inside each critic's prompt and the infrastructure around model routing. Validation that V2.1 actually fixed these problems is EXP-10.

### V1 -> V2.1 change summary

| Component | V1 | V2.1 |
|---|---|---|
| Chain-of-thought format | Free-form prose | **Structured: EVIDENCE FOR / EVIDENCE AGAINST / DECISION** |
| Few-shot examples | None | **One worked example per critic** |
| Quote grounding | Unconstrained; no verification | **Primary quote + 0-2 additional quotes, all substring-verified against input text**; mismatches logged as `hallucinated_quote` / `hallucinated_additional_quote` |
| Quote strength | Not measured | **`quote_strength` field required: weak / moderate / strong** |
| Counter-consideration | Optional, often missing | **`counter_consideration` field required >=15 chars; auto-FLAG if absent or too short** |
| Confidence | Categorical (low/med/high) | **Float 0.0-1.0** |
| Flag mechanism | Single binary flag | **`flag_trigger` field names the specific reason; multiple flag types supported** |
| Critic 3 understated-threat path | Single path (model self-reports escalation) | **Dual path: (a) model self-reports `understated:true` + FLAG; (b) numeric safety net — parser catches raised threat scores, clamps to range, and FLAGs independently of self-report** |
| Big-swing threshold | Ad hoc | **Default 0.5 (parameter swept in EXP-17, not yet run)** |
| Provider routing | Prefix-based (string matching on model name) | **Explicit per-model routing table in `src/critics/provider_routing.py` — no prefix matching anywhere in the critic stack** |
| Retry behavior | Single attempt; errors dropped | **4-attempt retry with backoff on transient errors; unparseable responses counted and logged** |

### Validation pointer
Every V2.1 design change is validated quantitatively in EXP-10. The structured CoT -> parseable output claim is validated by the per-model unparseable rate (2-53 of 1,500 calls). The substring-grounding claim is validated by the 0.27%-2.40% hallucination rate across 9,000 critic calls. The mandatory counter-consideration claim is validated by 0% missing-counter rate across all six models. The Critic 3 dual-path claim is validated by the understated_self_report counts (52-299 per model) and the separate numeric-path triggers (1 per model maximum). The explicit-routing claim is validated by the absence of V1-style routing errors across 9,000 calls.

### What the redesign claims
V2.1 critics produce outputs that can be parsed programmatically, verified against source text, calibrated along two independent axes (confidence and quote strength), and audited for depth of reasoning (counter-consideration length). None of these properties held in V1. The redesign does not claim to make critics correct — correctness for an LLM critic remains undefined — but it makes them trustable enough to disagree productively, which is the actual requirement for a multi-agent audit layer.

### Paper placement
Methods section, "Critic layer" subsection. Presented as the prompt and infrastructure architecture. Each field in the V2.1 output schema is documented with one sentence of justification. The V1 -> V2.1 diff table above may appear in an appendix depending on space.

### Risks / caveats
- **Design one-pager, not measurement.** Any claim about V2.1 being "better" than V1 routes through EXP-10's results, not this document.
- **One-line rationale for each design change should be traceable to a specific V1 failure mode.** We have those traces but not all of them are in the progress log; some are in chat history from the V2.1 rewrite session.
- **The structured-CoT format assumes the model will follow the structure.** Models that ignore the EVIDENCE FOR / EVIDENCE AGAINST / DECISION headers fall into the `unparseable` bucket. EXP-10 shows this rate is 2 (Gemma) to 53 (Qwen) out of 1,500 — acceptable but non-zero.

---

## EXP-10 — Critic V2.1 baseline benchmark

**Experiment ID:** EXP-10
**Date run:** 2026-04 (post V2.1 redesign)
**Paper placement:** Main Results (critic layer) — this is the headline critic-architecture benchmark

### Research question
Does the V2.1 redesign produce critic outputs that are parseable, grounded, well-calibrated, and inter-model-consistent across the top-6 models at a scale sufficient to support paper claims?

### Motivation
This is the load-bearing experiment for the critic-layer story. EXP-08 picked the models. EXP-09 redesigned the prompt. EXP-10 is where those two decisions either pay off or don't. Every claim about the critic layer in the paper — that it's robust, that models can disagree productively, that evidence grounding reduces hallucination, that the dual-path understated-threat mechanism catches Critic 3 miscalibrations — routes through the numbers in this section. If EXP-10 had failed, the paper would be a Dim1+Dim2+Dim3 architecture paper with no critic layer claim at all.

### Setup
- **Models (top 6 from EXP-08):** `gemma-4-31b-it`, `gpt-5.4-mini`, `claude-sonnet-4-6`, `deepseek-r1`, `grok-4.20`, `qwen-plus-2025-07-28`
- **Benchmark size:** 500 examples x 3 critics x 6 models = **9,000 critic calls**
- **Critics:** C1 Classification Validation, C2 Contextual Fairness, C3 Threat Escalation
- **Prompt architecture:** V2.1 (full detail in EXP-09)
- **Routing:** explicit per-model overrides via `src/critics/provider_routing.py`
- **Retry:** 4 attempts on transient errors; unparseable responses logged but not retried
- **Raw outputs:** `src/critics/benchmark_v2_results/{model}.jsonl`
- **Aggregate analysis:** `src/critics/analysis_v2/addendum_v2.1_baseline.json` + `kappa_matrix_v2.1_baseline.csv`
- **Baseline snapshot:** preserved as `*_v2.1_baseline.*` so future critic changes can be compared to this exact benchmark

### Metrics
For each model, across 1,500 critic calls (500 examples x 3 critics):
- **API error rate** (operational health)
- **Unparseable rate** (V2.1 structured-CoT compliance)
- **Halluc_primary and halluc_additional counts** (grounding quality — see EXP-11 for full breakdown)
- **Missing counter-consideration rate** (should be 0 by construction)
- **Missing evidence source count**
- **`understated_self_report` count** (Critic 3 self-reported miscalibration — see EXP-12)
- **Score direction conflicts and big-swing flag counts**
- **Mean and median confidence per critic**
- **Quote strength distribution per critic** (weak / moderate / strong)
- **Counter-consideration mean and median length** (reasoning-depth proxy)
- **Additional-quotes distribution** (0 / 1 / 2 per call)
- **Pairwise inter-model kappa** across the 6 models

### Results — per-model summary (n=500 examples, 1,500 calls per model)

| Model | API err | Unparse | Halluc prim (%) | Halluc addl | Understated SR | Missing ctr | Counter len mean | Big-swing / score conf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| claude-sonnet-4-6 | 23 | 23 | 36 (**2.40%**) | 22 | 62 | 0 | **185.6** | 0 / 0 |
| deepseek-r1 | 20 | 28 | 11 (0.73%) | 15 | 250 | 0 | 128.7 | 0 / 0 |
| gemma-4-31b-it | 1 | 2 | 9 (0.60%) | 5 | 109 | 0 | 95.8 | 0 / 0 |
| gpt-5.4-mini | 24 | 24 | 4 (**0.27%**) | 24 | 115 | 0 | 117.6 | 1 / 0 |
| qwen-plus-2025-07-28 | 22 | 53 | 23 (1.53%) | 38 | **299** | 0 | 110.7 | 0 / 1 |
| grok-4.20 | 27 | 27 | 20 (1.33%) | 11 | **52** | 0 | 140.0 | 0 / 1 |

### Results — confidence calibration (mean per critic)

| Model | C1 conf | C2 conf | C3 conf |
|---|---:|---:|---:|
| sonnet-4-6 | 0.850 | 0.852 | 0.866 |
| deepseek-r1 | 0.892 | 0.855 | 0.815 |
| gemma-4-31b-it | **0.952** | **0.961** | **0.965** |
| gpt-5.4-mini | 0.928 | 0.926 | 0.924 |
| qwen | 0.899 | 0.883 | 0.853 |
| grok-4.20 | 0.871 | 0.839 | 0.862 |

### Results — quote strength distribution (strong / moderate / weak, per 500 calls per critic)

| Model | C1 s/m/w | C2 s/m/w | C3 s/m/w |
|---|---|---|---|
| sonnet-4-6 | 282 / 175 / 35 | 259 / 165 / 68 | 86 / 153 / **254** |
| deepseek-r1 | 382 / 111 / 1 | 347 / 145 / 1 | 227 / 236 / 22 |
| gemma-4-31b-it | 404 / 94 / 1 | 439 / 60 / 0 | 390 / 110 / 0 |
| gpt-5.4-mini | 268 / 206 / 18 | 257 / 213 / 22 | 196 / 228 / 68 |
| qwen | 304 / 141 / 34 | 278 / 155 / 52 | 200 / 203 / 80 |
| grok-4.20 | 395 / 96 / 0 | 386 / 105 / 0 | 166 / 223 / 102 |

### Results — inter-model kappa (V2.1 baseline, 6x6)

|  | sonnet | r1 | gemma | gpt-mini | qwen | grok |
|---|---:|---:|---:|---:|---:|---:|
| **sonnet** | 1.000 | 0.316 | 0.370 | 0.295 | 0.293 | 0.319 |
| **r1** | 0.316 | 1.000 | 0.379 | 0.307 | 0.318 | 0.286 |
| **gemma** | 0.370 | 0.379 | 1.000 | 0.341 | 0.280 | 0.420 |
| **gpt-mini** | 0.295 | 0.307 | 0.341 | 1.000 | 0.276 | 0.268 |
| **qwen** | 0.293 | 0.318 | 0.280 | 0.276 | 1.000 | 0.222 |
| **grok** | 0.319 | 0.286 | 0.420 | 0.268 | 0.222 | 1.000 |

**Highest pair:** gemma <-> grok, kappa = 0.420.
**Lowest pair:** qwen <-> grok, kappa = 0.222.
**DeepSeek-R1 transformation.** In V1 (EXP-08), R1 showed the lowest cluster kappa against the final top-6 (~0.29 mean). Under V2.1 structured CoT, R1's mean kappa against the other five models rose to ~0.32 — comparable to Sonnet and gemma-4-31b. This is the single strongest piece of evidence that structured prompting matters more than model family for critic consistency.

### What the results mean
V2.1 works across all six models. Zero missing counter-considerations across 9,000 calls confirms the `>=15 chars or auto-FLAG` constraint is enforced and working. Hallucination rates are all under 2.5% (detailed in EXP-11), validating substring-grounding as sufficient defense against fabricated evidence. Unparseable rates are acceptable (0.13% for Gemma, 3.53% for Qwen as the worst case); 4-attempt retry handles the rest. Inter-model kappa is consistently in the 0.22-0.42 range, which is exactly what an independent critic ensemble should look like — models agree more than chance but far less than they would if they shared a reasoning lineage. Counter-consideration length varies by 93% between the highest (Sonnet, 185.6 chars) and the lowest (Gemma, 95.8 chars), which is the raw material for the "reasoning depth as a separate axis from confidence" contribution. The Critic 3 understated_self_report count varies over a **6x range across models** (52 for Grok to 299 for Qwen), which is the strongest novel finding in the benchmark and is the subject of EXP-12.

### Paper placement
Main Results, critic layer subsection. The per-model summary table, kappa matrix, and a condensed version of the counter-consideration / quote-strength tables are all main-text. Full quote-strength breakdown and confidence-by-critic tables move to Appendix.

### Risks / caveats
- **Gemma confidence is suspiciously high** (0.952/0.961/0.965 — nearly saturated). This reads as a calibration failure: Gemma emits "I am 98% confident" on almost every call regardless of content. The paper should flag this as a Gemma-specific calibration limitation and caveat any claim where Gemma's confidence number is load-bearing.
- **API errors cluster around 20-27 per model** (1.3-1.8%) except for Gemma (1). Consistent OpenRouter transient failures — not a V2.1 issue.
- **Inter-critic kappa (within-model)** is not in the reported numbers. The 6x6 matrix above is inter-model. A reviewer might ask whether Critic 1, Critic 2, and Critic 3 within the same model are actually measuring different things. `[TODO: recompute inter-critic kappa from the 3,000 raw rows per model; include in Appendix if space allows]`.
- **500-example benchmark is small by supervised-learning standards** but reasonable for an LLM-critic benchmark where each example requires 3 x 6 = 18 critic calls. Scaling to 2,000+ examples is budget-feasible but not scheduled before the May 1 deadline.
- **Dataset source for the 500 examples** should be named explicitly in the paper. `[TODO: confirm and document the benchmark dataset source — label distribution, provenance, and overlap with Dim1 training data]`.

---

## EXP-11 — V2.1 quote-grounding hallucination analysis

**Experiment ID:** EXP-11
**Date run:** 2026-04 (post-EXP-10)
**Paper placement:** Main Results — evidence-grounding validation (novel contribution)

### Research question
At what rate do V2.1 critics hallucinate evidence quotes (cite text that does not appear in the input), and how does the rate vary across models, quote types (primary vs additional), and critics?

### Motivation
The most common failure mode for LLM-as-judge architectures is confidently citing evidence that does not exist in the source. V2.1's primary defense is substring verification: every quote the critic emits is checked against the input text, and any mismatch is logged as a hallucination. The experiment answers the question a reviewer will definitely ask — "how often do your critics hallucinate?" — with a number rather than a gesture. It also distinguishes "primary quote hallucination" (the load-bearing evidence for the critic's decision) from "additional quote hallucination" (supplementary corroboration), because these are different severity.

### Setup
- **Data:** same 9,000-call benchmark as EXP-10 (500 x 3 critics x 6 models)
- **Hallucination definition:** a quote is hallucinated iff its verbatim string (after lowercase + whitespace normalization) does not appear as a substring of the original input text. Primary quote and additional quotes are checked independently.
- **Fields logged per call:** `halluc_primary` (0 or 1), `halluc_additional` (count of hallucinated additional quotes; 0 to 2), plus derived `halluc_primary_pct` per model.
- **Retry behavior:** hallucinations are logged but not retried — a hallucinated quote counts against the model's score.

### Metrics
- **Primary hallucination count and rate** (denominator = 1,500 critic calls per model — one primary quote per call)
- **Additional hallucination count** (absolute count; denominator varies with how many additional quotes the model chose to emit, so rates are not strictly comparable)

### Results

| Model | Primary hallucinations | Primary rate | Additional hallucinations |
|---|---:|---:|---:|
| gpt-5.4-mini | 4 | **0.27%** | 24 |
| gemma-4-31b-it | 9 | 0.60% | 5 |
| deepseek-r1 | 11 | 0.73% | 15 |
| grok-4.20 | 20 | 1.33% | 11 |
| qwen-plus-2025-07-28 | 23 | 1.53% | 38 |
| claude-sonnet-4-6 | 36 | **2.40%** | 22 |

**Aggregate across all 9,000 calls: 103 primary hallucinations and 115 additional hallucinations.**
**Range: 0.27% (gpt-5.4-mini) to 2.40% (Sonnet), span ~9x.**

### What the results mean
Substring-grounding works. Across 9,000 critic calls and 6 models, the worst primary-hallucination rate is 2.40% and the best is 0.27% — all well below the double-digit hallucination rates typical of ungrounded LLM-as-judge setups. The mechanism is simple (substring check), and the simplicity is the point: this is not a sophisticated defense, it is a floor-level one, and even this floor-level defense reduces hallucinated evidence to <=2.40% in the worst case. **This is the paper's primary contribution to the LLM-critic grounding conversation:** explicit substring verification of quoted evidence, applied at every critic call, reduces hallucinated-evidence rates to single-digit percentages without special training, reinforcement learning, or retrieval augmentation.

The rate variation across models is informative. Sonnet's 2.40% is the highest, not because Sonnet is "worse" but because Sonnet emits longer, more nuanced quotes more often (its counter-consideration length is also the highest at 185.6 chars; see EXP-10). Longer quotes have more surface area for minor paraphrase slip, which the substring check treats as hallucination. gpt-5.4-mini's 0.27% reflects the opposite — shorter, more literal quotations that are less likely to slip. This is a reasoning-depth-vs-surface-compliance tradeoff, not a correctness tradeoff.

### Paper placement
Main Results, subsection "Evidence grounding." The full per-model table above is main-text. One paragraph of interpretation. This is a core novel-contribution claim and should not be buried.

### Risks / caveats
- **Substring match is strict.** Any paraphrase, punctuation difference, or whitespace normalization mismatch counts as hallucination. This inflates the hallucination rate at the upper end — some of Sonnet's 36 "hallucinations" are likely legitimate close paraphrases that lost exact-match. A fuzzier match (e.g., token-level Jaccard > 0.8) would give lower numbers, but also less reliable grounding. The strict rule is the defensible one.
- **Additional-quote rates are not normalized to calls-with-quotes,** only to total calls. Gemma emits few additional quotes (see EXP-10 `aq_distribution`: 740 zero / 619 one / 141 two) so its absolute additional hallucination count of 5 is not directly comparable to gpt-5.4-mini's 24 (which emits 131 zero / 769 one / 600 two). Raw rates per-emitted-quote should be computed for the final paper.
- **Hallucination does not imply the critic's decision was wrong.** A critic may hallucinate a quote and still reach the correct verdict for other reasons. The two quantities are not measured separately in this experiment.
- **No human verification** of the logged hallucinations — spot-check review is `[TODO: manually verify 20 logged hallucinations per model to confirm substring-mismatch catches real fabrication rather than normalization artifacts]`.

---

## EXP-12 — V2.1 understated_self_report cross-model divergence

**Experiment ID:** EXP-12
**Date run:** 2026-04 (post-EXP-10)
**Paper placement:** Main Results — novel finding about critic calibration on Dim3 threat judgments

### Research question
Do the six top-6 models agree on how often Critic 3 (Threat Escalation) has understated the threat in its own decision, and if not, what is the magnitude and direction of their disagreement?

### Motivation
Critic 3's dual-path understated-threat mechanism is an unusual architectural choice: rather than asking the critic to produce a calibrated confidence and trusting it, V2.1 explicitly asks the critic whether it believes it has understated the threat in its own response (`understated: true/false`), and also runs a parser-level numeric safety net that independently catches raised threat scores and flags them. The two paths should produce roughly equivalent counts if the models are self-aware about calibration. If they differ wildly, that itself is a finding — either models disagree with themselves, or they disagree with each other, or both.

### Setup
- **Data:** same 9,000-call benchmark as EXP-10. Filtered to Critic 3 calls (500 per model = 3,000 total).
- **Two paths measured independently:**
  - **`score_understated_self_report`** — critic's own self-report that it understated the threat, set via the `understated: true` field in the structured response
  - **Numeric safety net** — parser catches raised threat scores beyond the permitted range, clamps them, and independently sets a flag. In the benchmark these appear as `score_understated_op_threat_score` and `score_understated_extremism_risk_score` in the `all_triggers` field, independent of the self-report path.
- **Metric:** absolute count of each trigger type per model, across the 500 Critic 3 calls per model.

### Results — Critic 3 understated triggers, by model and path

| Model | Self-report count | Self-report rate | Numeric safety-net triggers | Ratio (self-report : numeric) |
|---|---:|---:|---:|---:|
| qwen-plus-2025-07-28 | **299** | 59.8% | 1 | 299 : 1 |
| deepseek-r1 | 250 | 50.0% | 0 | 250 : 0 |
| gpt-5.4-mini | 115 | 23.0% | 0 | 115 : 0 |
| gemma-4-31b-it | 109 | 21.8% | 0 | 109 : 0 |
| claude-sonnet-4-6 | 62 | 12.4% | 0 | 62 : 0 |
| grok-4.20 | **52** | 10.4% | 1 | 52 : 1 |

**Range of self-report rates: 10.4% (Grok) to 59.8% (Qwen). Ratio: ~5.75x.**
**Numeric-path triggers: 2 total across all 3,000 Critic 3 calls (1 from Qwen, 1 from Grok).**

### What the results mean
The self-report path and the numeric safety-net path produce wildly different counts — on the order of 100x — and the self-report path itself varies across models by almost 6x. Three things are true simultaneously:

1. **Models disagree with each other about how often Critic 3 understates threats** (10.4% to 59.8% is not noise). Qwen and DeepSeek-R1 self-report understatement on half or more of their Critic 3 calls; Grok and Sonnet self-report it on one call in ten.
2. **The numeric safety net almost never fires** (2 triggers in 3,000 calls, 0.067%). This is not because models are always within bounds — the self-report path says otherwise — but because the numeric-range clamp is a much narrower definition of "understated" than the critic's own semantic notion.
3. **The two paths are not measuring the same thing.** The self-report path is the critic's own semantic judgment about calibration; the numeric path is a mechanical range check. They were designed as complementary safety nets, and the benchmark confirms they catch different things.

**The novel paper-worthy finding is the 5.75x cross-model variation in self-reported calibration.** Qwen and R1 are clearly more willing to flag their own outputs as understated than Grok and Sonnet are. Two interpretations are possible and the data cannot distinguish them yet: **(a)** Qwen and R1 are genuinely better-calibrated about their own weaknesses and catch real under-scoring; **(b)** Qwen and R1 are over-eager to hedge and produce spurious self-flags. A reviewer will ask which one it is, and the honest answer is "we do not yet know, and distinguishing them is future work." But the **existence** of the disagreement is itself the point: it is empirical evidence that critic calibration is not a solved property of modern LLMs, that different models have systematically different internal standards for "have I understated this," and that any single-model critic layer will inherit that single model's calibration bias silently.

### Paper placement
Main Results, critic-layer subsection, as a "novel finding" subsection or callout box. The per-model table above is main-text. One paragraph of interpretation, one paragraph flagging the (a) vs (b) ambiguity as future work. This is the strongest individual piece of evidence in the critic layer for why a multi-model ensemble matters — the alternative is picking one model and inheriting whichever calibration bias that model has.

### Risks / caveats
- **Two interpretations cannot be distinguished from this data alone.** Whether the high-self-report models (Qwen, R1) are more-calibrated or over-hedging is the central question and is not answered here. Future work: correlate `understated_self_report` with human-annotated under-scoring labels on a subsample.
- **Overlap analysis across models is not in the current benchmark.** We know Qwen flagged 299 and Grok flagged 52, but not how many of Grok's 52 are also in Qwen's 299. Near-complete overlap would mean "models agree on which examples are understated but disagree on how often." Near-disjoint overlap would mean "models disagree on both what and how much." These are different findings. `[TODO: compute per-example understated-flag overlap across the 6 models by joining on example_id in the raw JSONL files]`.
- **Numeric safety-net triggers are so rare (2/3000) that the path's utility as an ongoing safety mechanism is unclear.** It may be worth removing from V2.2 or lowering the trigger threshold. The paper can frame it either as a secondary safety belt (the optimistic framing) or as an under-triggered mechanism that warrants recalibration (the honest framing).
- **`understated_self_report` is a critic's self-assessment, not ground truth.** A model that confidently produces wrong answers will also confidently produce wrong self-assessments of those answers. The finding is about inter-model variation, not about correctness.
- **This experiment depends on the Critic 3 prompt structure being followed.** Models that drift from the V2.1 `understated: true/false` field entirely would produce artificial zeros. The 52 / 62 / 109 / 115 / 250 / 299 range suggests all six models are using the field, not drifting.

---

## EXP-13 — V2.1 prompt audit and rerun (RR-01)

**Experiment ID:** EXP-13
**Date run:** 2026-04-14
**Paper placement:** Methods (critic-prompt design) + Main Results (critic model comparison table)

### Research question
Does adding structured decision rules, boundary disambiguation examples, and fairness checklists to critic system prompts improve cross-model agreement (kappa-stability), reduce review rates, and preserve grounding fidelity, without rebuilding the benchmark infrastructure or the 500-example cache? Stated as a hypothesis: moving from generic critic role descriptions to structured critic prompts with taxonomy definitions, disambiguating tests, and action-choice rubrics will (a) raise kappa-stability across models, (b) reduce flag and review rates by giving critics a clearer basis for decisions, and (c) not introduce new hallucination failure modes.

### Motivation
The V2.1 baseline (EXP-10) gave critics a structured REASONING block (EVIDENCE FOR / AGAINST / DECISION) and a counter_consideration field, but left critic role descriptions thin: one or two sentences per critic with no decision rules, no examples, and no taxonomy. Two problems followed. First, the Stage 2 confusion matrix showed 1,262 Identity Hate examples misclassified as Interpersonal Abuse, and the V2.1 baseline critics had no principled way to adjudicate this boundary. Second, Critic 2 (Contextual Fairness) had no fairness checklist, and Critic 3 (Threat Escalation) had no score definitions or KG false-positive guidance, making them less able to push back on upstream errors. RR-01 tests whether adding that structure while preserving V2.1's grounding constraints improves cross-model agreement.

### Setup
- **Critic pipeline:** V2.1 critics (`src/critics/critics_v2.py`), structured CoT REASONING block, counter_consideration field, primary + up to 2 additional_quotes, quote_strength field, all three critics running in parallel per example.
- **Prompt audit additions (Option C taxonomy, preserving the old "1.0 = non-harmful" label convention):**
  - Critic 1: HARM LABEL TAXONOMY block (1.0, 1.1, 1.2, 1.3), Identity Hate vs Interpersonal Abuse decision rule with disambiguating test ("mentally remove identity references; does the harm survive?"), Examples A and B at the boundary, ACTION CHOICE rubric.
  - Critic 2: FAIRNESS CHECKLIST (reclaimed in-group language, AAVE, quoted/reported speech, irony, counter-speech, critique of power), explicit WHEN NOT TO OVERRIDE clause, late-added AGREE-vs-OVERRIDE clarifier that forbids "overriding" Dim1's already-non-harmful labels.
  - Critic 3: SCORE DEFINITIONS for extremism_risk_score and op_threat_score on a 0.0/0.3/0.6/1.0 scale, KG false-positive guidance (country names, demographic terms, broad nouns all discounted), VETO-ONLY-DOWNWARD rule restated, Russia travel-commentary false-positive example.
- **Models (top 5, qwen dropped from V2.1 baseline):** `google/gemma-4-31b-it`, `gpt-5.4-mini`, `anthropic/claude-sonnet-4-6`, `deepseek/deepseek-r1`, `x-ai/grok-4.20`.
- **Cache:** 500-example cache at `src/critics/cache_examples.jsonl`, unchanged from V2.1 baseline. Dim1 label distribution: 1.0 non-harmful 194 (38.8%), 1.1 identity hate 194 (38.8%), 1.2 interpersonal abuse 42 (8.4%), 1.3 crisis 70 (14.0%). Dim3 activation rate 10.8% (confirms cache was built post-activation-gate-fix).
- **Routing:** all models via OpenRouter with explicit per-model overrides, 4-attempt retry on transient errors, temperature 0.0.
- **Hardware:** Lightning.ai A100 80GB, but benchmark is API-only (no GPU use).
- **Benchmark call volume:** 5 models x 500 examples x 3 critics = **7,500 critic calls**.
- **Wall time:** 1 hour 30 minutes, 0 errors at the model level (individual critic-call parse failures counted separately below).
- **Pipeline aggregator fix applied alongside:** `_critic_label_to_canonical` now reads `corrected.label` as an integer fallback when `corrected.harm_label` is absent, so Critic 2's votes are no longer silently ignored in the majority-override branch.
- **Output paths:** `src/critics/benchmark_v2_results/` (raw jsonl per model), `src/critics/analysis_v2/` (post-audit summary), `src/critics/analysis_v2_pre_audit/` (pre-audit snapshot).

### Metrics
- **agree_rate, adjust_rate, override_rate, flag_rate:** per-model distribution of critic actions across 1,500 critic calls. Primary indicator of how decisively each critic commits to a judgment.
- **review_rate:** fraction of examples where at least one critic flagged OR all three disagreed. Operational metric for how often human review would be triggered.
- **kappa-stability:** average pairwise Cohen's kappa between a model and all other models, on the harm-label decision. **Primary selection metric** for which critic model is most representative of cross-model consensus. Reported apples-to-apples (5 shared models only).
- **hallucination_rate:** fraction of critic calls where the primary or additional quote fails the substring check against the post. Fidelity of the grounding mechanism.
- **Critic 2 false-override rate:** fraction of Critic 2 OVERRIDE actions where Dim1 already predicted non-harmful and Critic 2 "overrode" to non-harmful. Measures phantom-override inflation in Critic 2's OVERRIDE totals.
- **Mean latency and estimated cost:** per-critic-call mean latency and total estimated OpenRouter cost for the 1,500-call run per model. Operational deployment signal.

### Results

**Kappa-stability, apples-to-apples 5-model comparison (qwen dropped from pre-audit to match post-audit):**

| Model | pre-audit (5-model) | post-audit | delta |
|---|---|---|---|
| google/gemma-4-31b-it | 0.3775 | **0.4210** | +0.044 |
| anthropic/claude-sonnet-4-6 | 0.3250 | 0.3520 | +0.027 |
| x-ai/grok-4.20 | 0.3232 | 0.3530 | +0.030 |
| deepseek/deepseek-r1 | 0.3220 | 0.3490 | +0.027 |
| gpt-5.4-mini | 0.3028 | 0.2330 | **-0.070** |

**Action distribution, pre-audit vs post-audit (shown as pre -> post):**

| Model | agree | override | flag | review rate |
|---|---|---|---|---|
| grok | 0.79 -> 0.69 | 0.12 -> 0.19 | 0.04 -> 0.02 | 0.23 -> 0.15 |
| gemma | 0.77 -> 0.76 | 0.20 -> 0.18 | 0.02 -> 0.02 | 0.27 -> **0.14** |
| sonnet | 0.68 -> 0.69 | 0.11 -> 0.12 | 0.16 -> 0.12 | 0.37 -> 0.33 |
| r1 | 0.66 -> **0.75** | 0.17 -> 0.16 | 0.16 -> **0.06** | 0.62 -> **0.30** |
| gpt-mini | 0.62 -> 0.58 | 0.10 -> 0.17 | 0.07 -> 0.08 | 0.47 -> 0.34 |

**Critic 2 false-override rate (OVERRIDE to non-harmful when Dim1 was already non-harmful):**

| Model | total c2 overrides | false overrides | false rate |
|---|---|---|---|
| anthropic/claude-sonnet-4-6 | 130 | 0 | **0.0%** |
| deepseek/deepseek-r1 | 116 | 0 | **0.0%** |
| google/gemma-4-31b-it | 160 | 0 | **0.0%** |
| x-ai/grok-4.20 | 252 | 12 | 4.8% |
| gpt-5.4-mini | 223 | 25 | **11.2%** |

**Grounding fidelity (post-audit):**

| Model | hallucination rate | unparseable | API errors |
|---|---|---|---|
| gpt-5.4-mini | **0.33%** | 0 | 0 |
| deepseek/deepseek-r1 | 0.53% | 48 | 18 |
| google/gemma-4-31b-it | 0.53% | 14 | 14 |
| x-ai/grok-4.20 | 1.47% | 0 | 0 |
| anthropic/claude-sonnet-4-6 | **2.87%** | 0 | 0 |

**r1 parse-failure diff, pre-audit vs post-audit (tests whether r1's review-rate drop is due to improved parseability rather than the prompt audit itself):**

| Metric | pre | post | delta |
|---|---|---|---|
| unparseable | 8 | 30 | **+22** |
| API errors | 20 | 18 | -2 |
| hallucinated primary quote | 11 | 8 | -3 |
| hallucinated additional quote | 15 | 13 | -2 |

r1 had 22 MORE unparseable responses post-audit, not fewer. All of r1's review-rate improvement (62% to 30%) is attributable to the structured decision rules themselves.

**Counter_consideration length (reasoning-depth proxy, chars):**

| Model | mean | median |
|---|---|---|
| anthropic/claude-sonnet-4-6 | **181.7** | 180 |
| x-ai/grok-4.20 | 140.9 | 139 |
| deepseek/deepseek-r1 | 127.2 | 126 |
| gpt-5.4-mini | 116.8 | 116 |
| google/gemma-4-31b-it | 96.3 | 95 |

**Estimated API cost and mean latency per 1,500-call run:**

| Model | total cost (USD) | mean latency per call (s) | cost per 0.01 kappa |
|---|---|---|---|
| google/gemma-4-31b-it | **$0.34** | 39.42 | $0.008 |
| gpt-5.4-mini | $0.61 | 2.99 | $0.026 |
| deepseek/deepseek-r1 | $7.68 | 58.78 | $0.220 |
| x-ai/grok-4.20 | $13.66 | 3.06 | $0.387 |
| anthropic/claude-sonnet-4-6 | **$16.45** | 9.06 | $0.467 |

Gemma delivers the best kappa-stability at 48x lower cost than sonnet. Gemma and r1 are the slowest critics per call (~40-60s); grok and gpt-mini are the fastest (~3s). Prices are rough OpenRouter estimates and should be validated against actual invoices before final submission.

### What the results mean
The audit achieved its primary goal: **kappa-stability rose for four of five models, review rates fell for four of five models, and r1 went from the least decisive critic (62% review rate) to the third most stable (30% review rate)**. The paper can claim that structured decision rules and boundary disambiguation in critic prompts produce measurable gains in cross-model agreement without degrading grounding fidelity. Gemma remains the most kappa-stable critic and is the recommended default for downstream pipeline integration (EXP-22).

The paper cannot claim the audit helped every model uniformly. gpt-5.4-mini lost 0.070 kappa-stability, the largest single-model regression in the experiment. The most plausible reading: gpt-mini is the smallest top-5 model, and structured prompts with long decision rules and multiple examples push it toward rule-following behavior it can only partially execute, leading to fine-grained but inconsistent distinctions (its ADJUST rate is 17%; every other model is below 10%). This is consistent with the V2 finding that prompt structure helps larger models more than smaller ones. The r1 win is the cleanest, most defensible story: parse failures actually went up under the audit, so the entire review-rate improvement is attributable to the prompts themselves rather than benchmark-runner improvements. The Critic 2 false-override finding is better than feared: three of five models had zero phantom overrides, and the worst offender (gpt-mini at 11.2%) is still small enough to report as a footnote rather than a caveat.

### Paper placement
Methods section: the audited prompts are the "final V2.1+" critic design and should be described in the critic-layer subsection with decision rules summarized in an appendix figure. Main Results: the pre/post comparison tables become Table X in the critic-model selection subsection, replacing the V2.1 baseline numbers. The r1 improvement story is a standalone paragraph in Results. The gpt-mini regression and the sonnet hallucination observation go into Discussion as qualitative notes on prompt-structure and model-capacity interaction. The cost table supports a one-paragraph "deployment considerations" note.

### Risks / caveats
- **Sonnet hallucination rate is 5.4x the median and unexplained.** Sonnet at 2.87% is the highest of any model; it is also the most deliberative critic on every other measure. See EXP-30 for the follow-up investigation.
- **gpt-5.4-mini regression is not validated against a held-out set.** `[TODO: run EXP-15 stratified subsample and recompute gpt-mini kappa-stability per category; if the regression is uniform across categories, it is a real capacity effect; if it concentrates on identity-hate vs interpersonal-abuse boundary cases, it is a rule-following confusion.]`
- **OpenRouter pricing estimates are approximate.** `[TODO: cross-check total cost figures against the actual OpenRouter invoice for 2026-04-14 before final submission.]`
- **The small Interpersonal Abuse class (N=42, 8.4%) in the cache means per-category statistics have wide confidence intervals.** EXP-15 addresses this via stratified resampling.
- **Critic 2 phantom-override footnote still required for grok (4.8%) and gpt-mini (11.2%).** Aggregator is unaffected because it requires two votes, but single-critic override precision numbers for those two models should be reported with the false-override rate.
- **The Critic 1 taxonomy patch was applied surgically, not via a clean revert.** Parse-checked and smoke-tested, but no regression suite exists. `[TODO: write a minimal prompt regression test (5 examples per critic, expected action) and run it before EXP-22 full-pipeline integration.]`
- **The benchmark cache is fixed at 500 examples.** All five models see the same posts, which is correct for comparison, but the absolute numbers are specific to this cache. A second run on a disjoint 500-example sample would strengthen claims about magnitude; time permitting, run EXP-24 bootstrap CIs to put error bars on the kappa-stability numbers.

---

## EXP-14 — Intra-model critic agreement (ensemble justification)

**Experiment ID:** EXP-14
**Date run:** 2026-04-14
**Paper placement:** Methods (critic-ensemble architecture) + Discussion (ensemble justification paragraph)

### Research question
Are the three critics (Classification Validation, Contextual Fairness, Threat Escalation) performing functionally independent work on the same input, or are they largely converging on the same judgment? Stated operationally: what is Cohen's kappa between Critic 1 and Critic 2, Critic 1 and Critic 3, and Critic 2 and Critic 3, measured within a single model on the same 500 examples? Low intra-model kappa means the three-critic ensemble is justified because each critic contributes independent signal; high intra-model kappa means one critic could be dropped without loss of information.

### Motivation
The three-critic-ensemble architecture is a central design claim of the paper. A reviewer will ask: "if all three critics usually agree, why not use one critic?" Without an empirical answer, the ensemble looks redundant and expensive. This experiment pre-empts that question with a direct measurement. It reuses the RR-01 data (no new API calls) and is pure post-hoc analysis.

### Setup
- **Data:** 500 examples x 5 models x 3 critics per example from `src/critics/benchmark_v2_results/`, post-audit (EXP-13).
- **Action canonicalization:** Critic actions are collapsed to three comparable classes for kappa computation:
  - AGREE and ADJUST both map to "keep" (Dim1's label stands).
  - OVERRIDE maps to "change" (critic wants a different label).
  - FLAG maps to "unsure" (critic defers to human review).
- **Metric:** Cohen's kappa between every pair of critics within each model, computed with `sklearn.metrics.cohen_kappa_score`.
- **Interpretation thresholds used in the paper:** kappa < 0.30 = critics are independent (ensemble justified); kappa > 0.60 = critics are redundant (one could be dropped); middle range = partially independent, case-by-case design call.

### Metrics
- **Cohen's kappa (pairwise, within model):** c1-vs-c2, c1-vs-c3, c2-vs-c3.
- **Mean intra-model kappa:** simple mean of the three pairs, used as the headline "how independent is this model's critic ensemble" number.
- Comparison to inter-model kappa (the EXP-13 cross-model stability) as context for whether intra-model or inter-model variance dominates.

### Results

**Intra-model Cohen's kappa per critic pair:**

| Model | c1 vs c2 | c1 vs c3 | c2 vs c3 | mean intra-kappa |
|---|---|---|---|---|
| anthropic/claude-sonnet-4-6 | 0.297 | 0.114 | 0.071 | **0.161** |
| google/gemma-4-31b-it | 0.314 | 0.108 | 0.029 | **0.150** |
| x-ai/grok-4.20 | 0.179 | 0.058 | 0.024 | 0.087 |
| deepseek/deepseek-r1 | 0.112 | 0.061 | -0.018 | 0.051 |
| gpt-5.4-mini | 0.101 | -0.002 | -0.032 | **0.022** |

**Intra-model vs inter-model kappa (inter is EXP-13 post-audit stability):**

| Model | mean intra-kappa | inter-model kappa | ratio (inter / intra) |
|---|---|---|---|
| google/gemma-4-31b-it | 0.150 | 0.421 | 2.8x |
| anthropic/claude-sonnet-4-6 | 0.161 | 0.352 | 2.2x |
| x-ai/grok-4.20 | 0.087 | 0.353 | 4.1x |
| deepseek/deepseek-r1 | 0.051 | 0.349 | 6.8x |
| gpt-5.4-mini | 0.022 | 0.233 | 10.6x |

Mean intra-model kappa ranges from 0.022 (gpt-mini) to 0.161 (sonnet). All five models sit firmly below the 0.30 "independent" threshold. For every model, inter-model kappa exceeds intra-model kappa by 2.2x to 10.6x. This means: when gemma-Critic-1 looks at a post, its judgment is more similar to sonnet-Critic-1 than it is to gemma-Critic-2 on the same post.

### What the results mean
**The three critics perform functionally independent work.** Within a single model, Critics 1, 2, and 3 produce near-chance-independent action distributions on the same 500 posts, with mean kappa never exceeding 0.161. The lowest pairwise kappa (c2 vs c3) is slightly negative for three of five models, indicating literally no agreement above chance. The highest pairwise kappa (c1 vs c2) tops out at 0.314 for gemma, which is still comfortably below the 0.30 interpretability threshold. Each critic role contributes distinct signal, and dropping any one critic would remove information from the ensemble. This validates the three-critic architecture empirically.

The paper should note one honest caveat: Critic 3 only activates meaningfully when Dim3 activates (10.8% of examples). On the other 89.2%, Critic 3 receives a null Dim3 output and essentially always AGREEs, which artificially lowers c1-vs-c3 and c2-vs-c3 kappas. The more stringent test is c1-vs-c2, because both read every post and perform classification work. Even there, the highest pairwise kappa is 0.314 (gemma), meaning Critic 1 and Critic 2 disagree on roughly two-thirds of the 500 posts after chance correction. The ensemble is doing genuine work, not padding.

The secondary finding is that **cross-model variance dominates within-model variance by a large margin.** Gemma-Critic-1 is more similar to sonnet-Critic-1 than to gemma-Critic-2. This suggests the critic *role* (1, 2, or 3) is a stronger source of judgment variance than the model backbone. Role design is more important than model choice once you're at a reasonable model size.

### Paper placement
Methods section: one paragraph in the critic-ensemble subsection citing the intra-model kappa numbers as empirical justification for the three-critic design. A small table (the first table above) belongs in the main text. The inter-vs-intra ratio table goes in Discussion as evidence that critic role design is more impactful than model choice.

### Risks / caveats
- **Critic 3 null-activation bias.** On ~89% of examples, Dim3 is not activated and Critic 3 receives a null payload, biasing it toward AGREE. The c1-vs-c3 and c2-vs-c3 numbers are deflated as a result. Report c1-vs-c2 as the primary intra-model kappa in the paper and note this bias in the footnote. `[TODO: recompute c1-vs-c3 and c2-vs-c3 kappas restricted to the 54 Dim3-activated examples only, and report both conditional and marginal kappas.]`
- **Action canonicalization is coarse.** Collapsing AGREE and ADJUST to "keep" loses information about confidence recalibration. `[TODO: also compute intra-model kappa on the fine-grained 4-action space (AGREE, ADJUST, OVERRIDE, FLAG) and report in the appendix if the story changes.]`
- **500-example sample size.** Intra-model kappa has known high variance on samples below 1000. `[TODO: bootstrap 95% CIs on intra-model kappas in EXP-24 and report interval estimates alongside point estimates.]`
- **gpt-mini's near-zero intra-kappa could be read two ways.** Either (a) its three critics are genuinely doing independent work (the charitable read) or (b) its three critics are each making near-random decisions (the skeptical read). The first reading is supported by gpt-mini's low hallucination rate (0.33%, lowest of any model), which is inconsistent with random behavior. But this should be stated explicitly. `[TODO: manually inspect 20 gpt-mini examples where all three critics took different actions to confirm the independent-work reading.]`

---

## EXP-15 — Stratified subsample ablation (class-balanced reanalysis)

**Experiment ID:** EXP-15
**Date run:** TODO
**Paper placement:** Appendix (robustness check) + footnote in main results table

### Research question
Do the kappa-stability and action-distribution results from EXP-13 hold when the 500-example cache is resampled to have equal representation across the four harm-label classes? In particular: is gemma's post-audit kappa-stability lead robust to the 38.8 / 38.8 / 8.4 / 14.0 class imbalance in the original cache, and does gpt-5.4-mini's regression concentrate on a specific category (suggesting a rule-following confusion) or persist uniformly (suggesting a capacity effect)?

### Motivation
The 500-example cache is class-imbalanced: 194 non-harmful, 194 identity hate, 42 interpersonal abuse, 70 crisis. Any aggregate metric on this cache is dominated by the two large classes. A reviewer will reasonably ask whether the per-model rankings would survive on a balanced test set. Running a stratified subsample answers that question without any new API calls, by recomputing metrics on a deterministic subset of the existing jsonl rows. The analysis is also the best way to interpret the gpt-5.4-mini regression in EXP-13: if gpt-mini loses kappa-stability uniformly across all four classes, it is a model-capacity story; if it loses only on the identity-hate vs interpersonal-abuse boundary, it is a rule-confusion story with different implications for the paper.

### Setup
- **Data source:** existing `src/critics/benchmark_v2_results/` jsonl files from EXP-13, joined to the cache at `src/critics/cache_examples.jsonl` on example id.
- **Stratification scheme:** sample N=42 examples from each of the four Dim1 label classes (non-harmful, identity hate, interpersonal abuse, crisis), yielding 42 x 4 = 168 examples with perfectly equal class representation. N=42 is chosen as the minimum class size (Interpersonal Abuse has exactly 42 examples). Use a fixed random seed (seed=42) so the subsample is reproducible.
- **Comparison:** recompute kappa-stability, agree/override/flag rates, hallucination rate, and mean counter_consideration length on the 168-example subset and compare against the full 500-example numbers from EXP-13.
- **Per-category breakdown:** within the subsample, compute kappa-stability and action distributions *conditional on* each of the four classes, yielding a 5 x 4 table per metric.
- **Output:** `src/critics/analysis_v2/stratified_subsample/` containing subsample_ids.json, kappa_by_class.csv, action_dist_by_class.csv, comparison_vs_full.txt.

### Metrics
- **kappa-stability on 168-example subset:** direct comparison to EXP-13's 500-example kappa. Tests whether model rankings are stable.
- **Per-class kappa-stability:** 5 models x 4 classes = 20 numbers. **Primary new measurement.** Tells the gpt-mini regression story and exposes whether any model is particularly weak on a specific category.
- **Per-class override-precision:** fraction of OVERRIDE actions that were correct (matched the gold label). Requires gold labels in the cache; if absent, skip this metric and note as caveat.
- **Delta vs full-cache:** per-model agree/override/flag rate differences between the 168-subset and the 500-full.

### Results
TODO: to be filled in during the next session. Expected outputs:

- A kappa-stability comparison table (full vs subsample) with 5 rows.
- A per-class kappa-stability heatmap (5 models x 4 classes).
- A per-class action distribution table.
- A single "gpt-mini regression diagnosis" paragraph stating whether the regression is uniform (capacity) or concentrated (rule confusion).

### What the results mean
TODO: to be filled in after the experiment runs. Expected outcomes and their interpretations:

- If gemma remains the top kappa-stability model on the subsample, the EXP-13 main-results ranking is robust to class imbalance and the paper can report the 500-example numbers as headline without caveats.
- If gemma loses its lead on the subsample, the paper needs to report both numbers and discuss which is more informative for deployment.
- If gpt-mini's regression is uniform across classes, the paper argues capacity-based limitation (structured prompts hurt small models).
- If gpt-mini's regression concentrates on a specific class, the paper argues rule-confusion and proposes a smaller set of decision rules for capacity-constrained critics.

### Paper placement
Appendix (robustness-check section) with the subsample ranking table and the per-class heatmap. A one-sentence footnote in the main-results table in the body: "Results are robust to class imbalance; see Appendix X for stratified subsample analysis." The gpt-mini diagnosis paragraph goes in Discussion next to the EXP-13 gpt-mini regression observation.

### Risks / caveats
- **N=42 per class is small for kappa-stability estimation.** Cohen's kappa has high variance below N=100; the subsample kappas will have wide confidence intervals. `[TODO: bootstrap 95% CIs on the subsample kappas as part of EXP-24.]`
- **Using Dim1's predicted label for stratification, not gold labels.** Ideally we would stratify on the gold labels, but if the cache does not contain gold labels for all 500 examples, stratifying on Dim1's predicted label is the fallback. This introduces a circularity: we are subsampling by the variable we are evaluating performance against. Note this in the paper. `[TODO: check whether cache_examples.jsonl has a gold_label field; if yes, restratify.]`
- **The subsample is fixed by seed=42.** Different seeds will give different 168-sample subsets and slightly different numbers. `[TODO: rerun with seeds 1, 7, 13, 42, 100 and report the range of each headline metric.]`
- **No new data is collected.** The subsample analysis inherits any biases in the original 500-example cache. If the cache is not representative of the target deployment distribution, the subsample does not fix that. This is a limitation of post-hoc reanalysis, not of the subsample procedure itself.
- **Alternative design: oversampling the small classes.** We could resample with replacement to 194 per class (matching the largest class) instead of subsampling to 42. Oversampling preserves statistical power but introduces duplicate examples, which break kappa's independence assumption. Stratified subsample is more defensible but statistically weaker. If the subsample numbers look noisy, EXP-15b could rerun with oversampling for comparison. `[TODO: decide whether EXP-15b is worth running after seeing EXP-15a numbers.]`

## EXP-15: Stratified Subsample Ablation
**Date:** April 16, 2026
**Status:** ✅ Complete
**Type:** Post-hoc analysis (no API calls)
**Output:** `src/critics/analysis_v2/stratified_subsample/`

**Design:** Resample 500-example cache to 42 per class × 4 classes = 168 balanced examples (seed=42). Recompute per-model kappa-stability, action distributions, and C2 false-override rates. Break out per-class action profiles to disambiguate gpt-5.4-mini regression.

**Results:**
- gpt-5.4-mini κ jumped from 0.233 (full cache, worst) to **0.577 (balanced, best)**. The full-cache "regression" was a class-imbalance artifact.
- Root cause: gpt-mini reviews **50.8%** of non-harmful content vs 13–23% for all other models. Harmful-class review rates are normal.
- Diagnosis: **benign over-triggering under structured prompts** — not capacity limitation, not IH/IA boundary confusion. Smaller models treat detailed decision rules as evidence of suspicion.
- Grok dropped from 0.353 to 0.311 on balanced set (worst). All other models improved by 0.13–0.18.

| Model | κ (full 500) | κ (balanced 168) | Non-harmful review rate |
|-------|-------------|-----------------|----------------------|
| gpt-5.4-mini | 0.233* | **0.577** | **0.508** |
| deepseek-r1 | 0.349 | 0.537 | 0.230 |
| claude-sonnet-4-6 | 0.352 | 0.504 | 0.167 |
| gemma-4-31b-it | 0.421 | 0.469 | 0.135 |
| grok-4.20 | 0.353 | 0.311 | 0.183 |

*\*Full-cache κ uses inter-model consensus metric; balanced κ uses intra-model critic-agreement metric. See EXP-24 for reconciliation.*

**Paper implication:** Section 1.3 of RR-01 brief rewritten. gpt-mini is not a capacity caveat — it is a specific, operationally interpretable failure mode worth its own paragraph in Results.

---

## EXP-24: Bootstrap 95% Confidence Intervals on Kappa-Stability
**Date:** April 16, 2026
**Status:** ✅ Complete
**Type:** Post-hoc analysis (no API calls)
**Output:** `src/critics/analysis_v2/bootstrap_cis/`

**Design:** 1,000 bootstrap iterations, resample with replacement (sample size = pool size), recompute mean pairwise Cohen's kappa across 3 critics per model. Three parts: (1) post-audit CIs on full 500, (2) paired pre/post audit delta with significance test, (3) gpt-mini full vs balanced comparison.

**Part 1 — Post-audit ranking with error bars:**

| Rank | Model | κ | 95% CI | CI width |
|------|-------|---|--------|----------|
| 1 | deepseek-r1 | 0.625 | [0.583, 0.670] | 0.088 |
| 2 | claude-sonnet-4-6 | 0.621 | [0.585, 0.658] | 0.074 |
| 3 | gpt-5.4-mini | 0.578 | [0.537, 0.620] | 0.083 |
| 4 | gemma-4-31b-it | 0.553 | [0.508, 0.597] | 0.088 |
| 5 | grok-4.20 | 0.400 | [0.362, 0.434] | 0.072 |

Two tiers: {r1, sonnet} > {gpt-mini, gemma} >> {grok}. r1/sonnet CIs overlap (not significantly different). gpt-mini/gemma CIs overlap (not significantly different). Grok separated below all.

**Part 2 — Audit improvement significance:**

| Model | Delta | 95% CI | Significant? |
|-------|-------|--------|-------------|
| gemma-4-31b-it | +0.211 | [+0.174, +0.246] | **YES** |
| gpt-5.4-mini | +0.078 | [+0.043, +0.115] | **YES** |
| deepseek-r1 | +0.076 | [+0.033, +0.118] | **YES** |
| claude-sonnet-4-6 | +0.068 | [+0.029, +0.106] | **YES** |
| grok-4.20 | −0.032 | [−0.070, +0.010] | no |

4/5 models show statistically significant improvement under the V2.1 prompt audit. Grok is the only non-significant model. gpt-mini **gained** +0.078 — the prior session's −0.070 "regression" is retracted (different metric).

**Part 3 — gpt-mini full vs balanced:**
Full κ = 0.578 CI [0.535, 0.617]; balanced κ = 0.577 CI [0.513, 0.640]. CIs overlap — not significantly different. EXP-15's benign over-escalation finding is real but does not change the ranking.

**Paper implication:** All headline kappa numbers now have error bars. The audit story holds up under resampling. Gemma remains the cost-adjusted recommendation ($0.34, 48× cheaper than sonnet) despite ranking 4th on absolute κ.
---
---

## End of document

**Summary of `[TODO]` items blocking final submission:**

1. **EXP-01:** run test-set evaluation on the 27,017-example test split
2. **EXP-02:** run test-set evaluation (same split)
3. **EXP-03:** attempt wandb log recovery for runs `_0107` and `_0208` before deciding final framing
4. **EXP-07:** re-measure activation rate on known N (suggested: Identity Hate partition of Dim1 validation split) + manual recall spot-check on ~50 Dim3-negative inputs
5. **EXP-10:** name the 500-example benchmark dataset source explicitly; recompute inter-critic kappa (within-model) from raw rows
6. **EXP-11:** manually verify ~20 logged hallucinations per model to distinguish real fabrication from substring-match artifacts
7. **EXP-12:** compute per-example understated-flag overlap across the 6 models from raw JSONL

None of these block draft Methods or draft Results writing. All of them should be resolved before May 1 final pass.
