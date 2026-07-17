# Data

This study reuses the corpora of the prior study **read-only** and adds no new raw data:

- Source: [excess-vocabulary-ja](https://github.com/kenimo49/excess-vocabulary-ja), commit `fdb24f4c970d0af29315fa6f31c3f67f000322bb`
  - `data/ai_samples/` — 350 AI documents (7 LLMs × 10 themes × 5 generations)
  - `data/human_corpus/` — 700 pre-LLM human articles (Qiita 567 + Zenn 133, 2020–2022)
- Expected local path for the scripts: `~/repos/excess-vocabulary-ja`

Derived document-level data produced by this study lives under `results/`:

- `results/lexical/doc_words_{ai,human}.jsonl` — per-document binary word sets (tokenizer-compatible with the prior study; token totals verified identical)
- `results/rhythm/rhythm_metrics.csv` — per-document rhythm metrics
- `results/discrimination/dataset/` — merged analysis dataset (docs.csv, lexical_binary.npz, vocab.txt)
