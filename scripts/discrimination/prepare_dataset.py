#!/usr/bin/env python3
"""exp028 Phase 3: H1/H2 用の解析データセット構築。

入力 (すべて既存 Phase の成果物、READ ONLY):
  results/lexical/doc_words_{ai,human}.jsonl  — 文書ごと binary 語集合
  results/rhythm/rhythm_metrics.csv          — 文書ごとリズム13指標
  results/discrimination/ending_mattr.csv    — 文末bigram MATTR (window=20)

出力 (results/dataset/):
  lexical_binary.npz  — scipy CSR (n_docs x vocab) binary 行列
  vocab.txt           — 列順の語リスト (全語彙、事前選択なし)
  docs.csv            — 行順の doc_id, y(1=AI), model, theme, リズム13指標,
                        長さ共変量, 縮退フラグ

設計判断:
  - 語彙ユニバースは doc_words jsonl の全語彙 (25,902語)。Phase 1.5 の
    1,684語ユニバースや FDR 有意 797語は「全データで選択された」集合なので
    H1 分類には持ち込まない (リーク防止)。語の選択は CV の学習 fold 内で行う。
  - リズム特徴は Phase 2 の 13指標のうち ending_bigram_ttr を
    ending_mattr20 に置き換えた 13 個。
  - 解析対象 = rhythm included==1 (文数>=5) の 1,046 文書。
  - 長さ共変量: log1p(文数), log1p(総文字数≈文長平均x文数)。
  - llama縮退フラグ: llama3.2-1b の文書で sent_len_char_mean /
    sent_len_char_sd / comma_per_sentence のいずれかが人間コーパスの
    99パーセンタイルを超えるもの。Phase 2 summary は基準を確定して
    いないため本 Phase で定義 (summary.md に明記、【要確認】)。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

REPO = Path(__file__).resolve().parent.parent.parent
P15 = REPO / "results" / "lexical"
P2 = REPO / "results" / "rhythm"
OUT = REPO / "results" / "discrimination" / "dataset"

RHYTHM_FEATURES = [
    "sent_len_char_mean", "sent_len_char_sd", "sent_len_char_cv",
    "sent_len_mora_mean", "sent_len_mora_sd", "sent_len_mora_cv",
    "burstiness_char", "burstiness_mora",
    "para_sent_cv",
    "taigendome_rate", "dewanaku_per_sentence", "comma_per_sentence",
    "ending_mattr20",
]


def jsonl_path_to_doc_id(path: str) -> str:
    # data/ai_samples/<model>/<stem>.md      -> ai/<model>/<stem>
    # data/human_corpus/<src>/<theme>/<stem>.md -> human/<src>/<theme>/<stem>
    p = Path(path)
    parts = p.with_suffix("").parts
    if parts[1] == "ai_samples":
        return "ai/" + "/".join(parts[2:])
    if parts[1] == "human_corpus":
        return "human/" + "/".join(parts[2:])
    raise ValueError(path)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    rhythm = pd.read_csv(P2 / "rhythm_metrics.csv")
    mattr = pd.read_csv(OUT.parent / "ending_mattr.csv")
    df = rhythm.merge(mattr[["doc_id", "n_endings", "window_used", "mattr20"]], on="doc_id", how="left")
    df = df[df.included == 1].copy()
    df = df.rename(columns={"mattr20": "ending_mattr20"})

    # 長さ共変量
    df["n_chars_total"] = df["sent_len_char_mean"] * df["n_sentences"]
    df["cov_log_sentences"] = np.log1p(df["n_sentences"])
    df["cov_log_chars"] = np.log1p(df["n_chars_total"])

    # llama 縮退フラグ (人間 p99 基準)
    hu = df[df.group == "human"]
    thr = {c: hu[c].quantile(0.99) for c in
           ["sent_len_char_mean", "sent_len_char_sd", "comma_per_sentence"]}
    degen = (df.model == "llama3.2-1b") & (
        (df.sent_len_char_mean > thr["sent_len_char_mean"])
        | (df.sent_len_char_sd > thr["sent_len_char_sd"])
        | (df.comma_per_sentence > thr["comma_per_sentence"])
    )
    df["llama_degenerate"] = degen.astype(int)
    print("degenerate llama docs:", int(degen.sum()), "thresholds:",
          {k: round(v, 1) for k, v in thr.items()})

    # 語彙行列
    docs_words: dict[str, list[str]] = {}
    for f in ["doc_words_ai.jsonl", "doc_words_human.jsonl"]:
        for line in (P15 / f).open():
            d = json.loads(line)
            docs_words[jsonl_path_to_doc_id(d["path"])] = d["words"]

    missing = [i for i in df.doc_id if i not in docs_words]
    assert not missing, f"doc_id mismatch: {missing[:5]}"

    vocab = sorted({w for words in docs_words.values() for w in words})
    vidx = {w: i for i, w in enumerate(vocab)}

    doc_ids = df.doc_id.tolist()
    rows, cols = [], []
    for r, doc_id in enumerate(doc_ids):
        for w in docs_words[doc_id]:
            rows.append(r)
            cols.append(vidx[w])
    X = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(len(doc_ids), len(vocab)),
    )

    df["y"] = (df.group == "ai").astype(int)
    keep = (["doc_id", "y", "group", "model", "source", "theme",
             "n_sentences", "n_sentences_prose", "n_endings", "window_used",
             "n_chars_total", "cov_log_sentences", "cov_log_chars",
             "llama_degenerate"] + RHYTHM_FEATURES)
    df[keep].to_csv(OUT / "docs.csv", index=False)
    sparse.save_npz(OUT / "lexical_binary.npz", X)
    (OUT / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")

    print(f"docs: {len(df)} (AI {df.y.sum()} / human {(1 - df.y).sum()})")
    print(f"vocab: {len(vocab)}, matrix nnz: {X.nnz}")
    nan_counts = df[RHYTHM_FEATURES].isna().sum()
    print("rhythm NaNs:\n", nan_counts[nan_counts > 0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
