# Algorithm notes

[English](ALGORITHM_en.md) · [中文](ALGORITHM_cn.md)

The repository deliberately uses two modelling routes instead of forcing one:

1. **composition → band gap**: arbitrary formulas, KG-subgraph GNN plus a
   graph-augmented Transformer;
2. **TiO2 process conditions → HER**: a small process table, RBF kernel ridge.

A third **synthetic scenario** exercises the KG, ranking and active-learning
code. It does not contribute to the real-data conclusions.

---

## 1. Composition model: GNN + graph-augmented Transformer

### 1.1 Formula parsing and element roles

The parser (`photocat/data/formula.py`) handles nested parentheses and
fractional stoichiometry, for example `Ag(AuS)2`, `Ba2Sb7HO14` and
`Ca0.5Sr0.5TiO3`. It assigns cation/anion roles: O, N, S, Se, Te, F, Cl, Br and
I are anions; P is an anion only when no oxygen is present; the rest are
cations. This is a composition heuristic, not an oxidation-state balance.

The element table contains 103 elements. The original 37 hand-curated entries
keep literature ionic radii; 63 additional entries are generated from
pymatgen 2026.10.24. Missing d-electron counts and abundances use conservative
placeholders. The model can therefore read any Matbench formula, but the
descriptor accuracy is lower than a dedicated structure database.

### 1.2 KG subgraph

Each material is a small graph:

- one material node and up to six element nodes;
- `material—hasElement—element` edges weighted by stoichiometric fraction,
  element co-occurrence edges weighted by \(2f_i f_j\), and self-loops;
- row-normalised adjacency \(\hat A = D^{-1}(A+I)\).

Node features are 9 element properties plus an `is_material` flag and a
role one-hot (14 dimensions) for element nodes; the material node uses 9
composition descriptors. Message passing is

\[
H^{(\ell+1)} = \mathrm{ReLU}\!\left(\hat A H^{(\ell)} W^{(\ell)} + b^{(\ell)}\right).
\]

The material vector is the material-node row; the graph vector is the mean over
nodes.

### 1.3 Token sequence

```text
[CLS]  element×≤6  family  crystal  cocatalyst  method  dopant
       pH-bucket  heterojunction  sacrificial-agent
```

Element tokens combine an element embedding, role embedding, stoichiometry
projection, projected GNN node embedding and position embedding. `[CLS]` also
receives the material graph vector and a 10-dimensional numeric context (pH,
surface area, strain, defect, heterojunction, dopant fraction, cocatalyst
loading, light intensity, catalyst loading, sacrificial concentration).
Missing context fields map to an unknown slot and do not crash the model.

### 1.4 Graph-biased attention

\[
s_{ij} = \frac{q_i^\top k_j}{\sqrt{d_h}} + \beta_h \tilde A_{ij},
\qquad
\alpha_{ij} = \mathrm{softmax}_{j\in\text{valid}}(s_{ij}),
\]

where \(\tilde A_{ij}\) is the normalised adjacency between the tokens' nodes
and \(\beta_h\) is a learned scalar per head. Padding keys are masked. After two
pre-LN residual blocks, the CLS vector and mean-pooled vector are concatenated
and passed through a shared MLP to three outputs: band gap,
\(\log(1+HER)\), and a stability logit.

### 1.5 Masked multi-task loss

Real datasets usually contain only one target. Let \(m_{i,t}\in\{0,1\}\)
indicate whether target \(t\) exists for item \(i\):

\[
\mathcal L = \sum_t \lambda_t
\frac{\sum_i m_{i,t} w_i (\hat y_{i,t}-y_{i,t})^2}
     {\sum_i m_{i,t} w_i},
\]

with a weighted BCE for stability. A task with no labels in the current batch
is skipped. Standardisation uses `nanmean`/`nanstd`; a target column with no
observations is returned as NaN at inference. This is what allows the same
trainer to use Matbench (band gap only), EES (band gap + HER) and the synthetic
scenario (all three) without inventing labels.

### 1.6 Training

- AdamW, cosine decay to 10% of the initial learning rate, gradient clipping 5;
- early stopping on validation loss;
- optional family re-weighting or oversampling in the synthetic scenario;
- ensemble across seeds, with mean prediction and standard deviation.

For the Matbench benchmark, four models are averaged: two plain models and two
with dropout 0.1. This mixed ensemble was chosen by validation error and
improves both RMSE and MAE over the two-plain-model ensemble.

Matbench result: 4,603 records, element-system holdout, RMSE 0.799 eV,
MAE 0.437 eV. Ridge and kNN baselines on the same split are 1.218 and
1.138 eV. See `docs/RESULTS_en.md`.

---

## 2. Small-table model: RBF kernel ridge

The EES table has only 552 complete records from 34 papers. The composition
Transformer overfits the process variables at this size, so the HER benchmark
uses kernel ridge regression:

\[
f(x) = \sum_{i=1}^{n} \alpha_i
\exp\!\left(-\gamma \|x-x_i\|^2\right) + \bar y,
\qquad
(K+\lambda I)\alpha = y-\bar y.
\]

Features are cocatalyst one-hot, cocatalyst loading, surface area, band gap,
sacrificial-agent one-hot and concentration, light intensity, and synthesis
route one-hot. The target is \(\log_{10}(1+\text{Rate})\). Features are
standardised. \(\gamma=0.03\) and \(\lambda=0.1\) are fixed and not tuned on the
test folds.

Evaluation uses leave-papers-out 5-fold cross-validation: all records from one
paper fall in a single fold. Results: R² 0.568, MAE 0.289, Spearman 0.698;
linear ridge gives 0.470, 0.324 and 0.586.

This is a model-selection result: with 552 rows and 34 groups, a model with a
stronger prior and fewer parameters is more appropriate. It is not a general
claim that kernels are better than Transformers.

---

## 3. Knowledge graph (synthetic scenario)

`photocat/kg/` provides a SQLite triple store, ontology, rule extraction, rule
reasoning and DistMult link prediction. Entities come from the synthetic
scenario or user-provided data; the full Matbench set is not currently imported
as a real knowledge base.

Rules include:

- `inferred:Candidate`: \(1.23\le E_g\le 3.0\), \(HER\ge 6\), stability \(\ge0.55\);
- `inferred:EarthAbundant`: all constituent elements at least 10 ppm;
- `inferred:Novel`: maximum Tanimoto similarity to the training library \(<0.8\);
- `inferred:HeterojunctionWith`: shared element and staggered band alignment
  \((E_{CB}^a<E_{CB}^b<E_{VB}^a<E_{VB}^b)\) or the symmetric case.

Inferred edges carry a `source` field and can be traced to supporting triples.
The extractor is tested only on the synthetic corpus and makes no claim to
match SciBERT on real literature.

---

## 4. Candidate ranking (synthetic scenario)

Hard constraints are the band-gap window, HER, stability and Tanimoto novelty.
The score is

\[
0.42\,\tilde z(\log(1+HER))
+0.18\,e^{-\frac12\left(\frac{E_g-2.0}{0.55}\right)^2}
+0.14\,P_{stable}
+0.14\,\text{novelty}
+0.07\,\text{certainty}
+0.05\,\text{abundance}.
\]

The shortlist uses score ordering, a Tanimoto threshold of 0.92, duplicate-label
removal and a Pareto front. These weights are scenario settings, not values
optimised on experiments.

---

## 5. Uncertainty and active learning

The deep ensemble gives \(\mu(x)\) and \(\sigma(x)\). Split-conformal intervals
are calibrated on the validation set:

\[
q = \mathrm{Quantile}_{\lceil(n+1)(1-\alpha)\rceil/n}(|y_i-\hat y_i|).
\]

For Matbench, the 90% interval half-width is 2.182 eV with coverage 0.894. The
wide interval is driven by high-gap outliers and is a warning against
over-interpreting single predictions.

Active learning is simulated only in the synthetic scenario: a ridge surrogate,
bootstrap standard deviation, and an acquisition function with a band-gap
window penalty are compared with random sampling at the same label budget. It
tests that the loop closes; it is not an experimental-budget claim.

---

## 6. Evaluation and reproducibility

| Benchmark | Split | Metrics |
| --- | --- | --- |
| Matbench experimental gap | element-system holdout | RMSE / MAE / R² / conformal coverage |
| EES HER | 5-fold by paper | log R² / MAE / Spearman |
| Synthetic scenario | composition group, family holdout | workflow completeness only |

Data hashes, DOIs and licences are in `docs/DATA_en.md`; reference checks are in
`docs/REFERENCES_en.md`. `python scripts/verify_provenance.py` re-queries
Crossref/arXiv and verifies file hashes.

## 7. Complexity and limitations

- Matbench: 4,603 records, four models × 24 epochs, about 2–3 minutes on an
  Apple Silicon CPU.
- Synthetic scenario: 12,000 records, three models × 30 epochs, about 5 minutes.
- The composition model has no crystal structure, temperature, morphology or
  defect information; experimental gaps themselves carry measurement variance.
- The EES table is overwhelmingly TiO2-based and its rate units are not
  harmonised; it cannot be extrapolated to other material families.
- The repository does not use raw Materials Project, Open Catalyst or SciBERT
  data. Those sources appear only in the proposal discussion and are not
  validation data for this package.
