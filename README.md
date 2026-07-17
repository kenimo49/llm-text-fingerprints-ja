# Lexical and Rhythmic Fingerprints of Japanese LLM-Generated Text

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21413035.svg)](https://doi.org/10.5281/zenodo.21413035)

A cross-model, two-layer analysis of what makes Japanese AI-generated text machine-like: word choice (lexical fingerprints) vs. rhythm (sentence-length variation, burstiness, paragraph structure). 7 LLMs × 350 generated documents vs. 700 human technical-blog articles.

Paper: [`paper/main.tex`](paper/main.tex) (English, compiled PDF included) — published on Zenodo: [10.5281/zenodo.21413035](https://doi.org/10.5281/zenodo.21413035).

## Research questions

- **H1 (discriminative power)**: For distinguishing human from AI Japanese text, which feature family is stronger — vocabulary or rhythm? The practitioner claim ([coji/natural-japanese](https://github.com/coji/natural-japanese)) predicts rhythm > vocabulary.
- **H2 (fingerprint layering)**: Are rhythm fingerprints homogeneous across models while lexical fingerprints differentiate them (a two-layer structure)?

Along the way, the study rebuilds the excess-vocabulary statistics of our prior work on a **document-occurrence basis** (following Kobak et al., Science Advances 2025) and corrects our own earlier token-based results.

## Key results

### 1. Correction of our prior 651-word list (document-occurrence re-analysis)

Token-level χ² tests in the prior study treated within-document repetition as independent evidence (pseudoreplication). Re-analysis on a document basis (Laplace smoothing, r/δ indices, BH-FDR, bootstrap CIs):

| Criterion | Survivors / 651 | AI-excess side (/458) |
|---|---|---|
| Document χ² + BH-FDR (primary) | **424 (65.1%)** | 237 (51.7%) |
| Document χ² + Bonferroni | 259 (39.8%) | 96 (21.0%) |

The prior #1 word "Hash" (excess score +189.1, 402 token occurrences) appeared in only 2 of 350 AI documents — indistinguishable from chance at the document level (r=2.00, q=0.955). The full re-analysis finds 797 significant words among 1,684 tested.

### 2. Rhythm: AI text is monotone, in every model

| Metric | Cohen's d (AI − human) [95% CI] |
|---|---|
| Burstiness (characters) | **−0.96** [−1.12, −0.81] |
| Burstiness (moras) | **−0.80** [−1.00, −0.62] |
| Paragraph sentence-count CV | **−0.67** [−0.83, −0.50] |
| Sentence-length CV (characters) | −0.56 [−0.78, −0.36] |

Direction is shared by **all 7 models** (per-model burstiness d = −0.59 to −1.72). GPT-3.5/GPT-4o are the most monotone; Claude Sonnet 4 / Opus 4 / GPT-OSS 20B are closest to human variation.

### 3. H1: lexical features dominate discrimination (opposite of the practitioner claim)

| Condition | Lexical AUC | Rhythm AUC | Combined |
|---|---|---|---|
| main (n=1,046) | 0.998 | 0.897 | 0.998 |
| length-residualized | 0.974 | 0.810 | 0.980 |
| degenerate docs removed | 0.998 | 0.913 | 0.998 |
| Llama excluded | 0.999 | 0.921 | 0.999 |

Paired difference (rhythm − lexical): **−0.101 [−0.105, −0.097]**, lexical wins in 100/100 folds; robust to length control and outlier exclusion. Rhythm is still non-redundant: combined − lexical = +0.006 [+0.005, +0.007] after length adjustment.

### 4. H2: two-layer fingerprint

| Feature set | 7-way model identification (chance 14.3%) |
|---|---|
| Lexical profile | **92.1% ± 3.8%** |
| Rhythm (13 metrics) | 50.9% ± 6.5% |

Rhythm shifts in a **shared direction** with model-dependent magnitude (layer 1); the **lexical profile identifies the model** (layer 2). Sentence-ending rhetoric (noun-ending rate, ending-bigram MATTR) disperses across models more than any core rhythm metric (descriptive).

> What is written tells you which model; how it is written tells you a machine wrote it.

## Repository structure

```
paper/                     main.tex + references.bib + compiled main.pdf
scripts/
  lexical/                 document-occurrence excess vocabulary (build_doc_matrix.py, excess_stats.py)
  rhythm/                  13 rhythm metrics (rhythm_metrics.py, analyze_groups.py)
  discrimination/          H1/H2 (compute_mattr.py, prepare_dataset.py, run_h1.py, run_h2.py, run_h2_model_id.py)
results/
  lexical/                 word_stats.csv (1,684 words), summary.md, run metadata
  rhythm/                  rhythm_metrics.csv (per document), group/model comparisons, summary.md
  discrimination/          H1 AUC tables, H2 dispersion/model-ID tables, dataset/, summary.md
data/                      pointer to the source corpora (see data/README.md)
```

The `summary.md` in each results directory is the analysis report for that stage (in Japanese; the paper is the English write-up).

## Reproduction

The corpora are reused read-only from [excess-vocabulary-ja](https://github.com/kenimo49/excess-vocabulary-ja) (`data/ai_samples`, `data/human_corpus`), expected at `~/repos/excess-vocabulary-ja`. All randomized steps are seeded (seed=20260717); dependency versions are recorded in `results/*/run_metadata.json` / `environment.json`.

```bash
pip install mecab-python3 unidic-lite numpy scipy pandas scikit-learn

# 1. Lexical (document-occurrence excess vocabulary)
python3 scripts/lexical/build_doc_matrix.py
python3 scripts/lexical/excess_stats.py

# 2. Rhythm (13 metrics)
python3 scripts/rhythm/rhythm_metrics.py
python3 scripts/rhythm/analyze_groups.py

# 3. Discrimination (H1/H2)
python3 scripts/discrimination/compute_mattr.py
python3 scripts/discrimination/prepare_dataset.py
python3 scripts/discrimination/run_h1.py          # ~4 conditions × 100 folds
python3 scripts/discrimination/run_h2.py
python3 scripts/discrimination/run_h2_model_id.py
```

To build the paper (requires a LaTeX installation with `CJKutf8`):

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Prior work in this series

1. **AI Text Slop: Quantitative Analysis of 16 Structural Patterns in LLM-Generated Japanese Text** — DOI: [10.5281/zenodo.19173035](https://doi.org/10.5281/zenodo.19173035) ([repo](https://github.com/kenimo49/ai-text-slop))
2. **Excess Vocabulary in Japanese AI-Generated Text: A Cross-Model Quantitative Analysis** — DOI: [10.5281/zenodo.19233934](https://doi.org/10.5281/zenodo.19233934) ([repo](https://github.com/kenimo49/excess-vocabulary-ja)) — the corpus source; its token-based statistics are corrected by this study

## Title

Working title: *Lexical and Rhythmic Fingerprints of Japanese LLM-Generated Text: A Cross-Model, Two-Layer Analysis*.

Alternatives considered:

- *Vocabulary Tells the Model, Rhythm Tells the Machine: A Two-Layer Fingerprint of Japanese LLM Text*
- *Two Layers of Machine Style: Lexical vs. Rhythmic Fingerprints in Japanese AI-Generated Text*

## Citation

```bibtex
@misc{imoto2026fingerprints,
  title={Lexical and Rhythmic Fingerprints of {Japanese} {LLM}-Generated Text: A Cross-Model, Two-Layer Analysis},
  author={Imoto, Ken},
  year={2026},
  doi={10.5281/zenodo.XXXXXXX},  % TODO: Zenodo concept DOI (reserved at deposit time)
  note={Zenodo}
}
```

## License

MIT — see [LICENSE](LICENSE).
