#!/usr/bin/env python3
"""exp028 Phase 3 H2 補助検証: 多変量でのモデル識別力 (語彙 vs リズム)。

動機:
  H2 主検定 (run_h2.py) の between/within ratio は単一特徴の分散比。
  binary 語は within 分散が構造的に大きい (Bernoulli 分散 p(1-p)) ため、
  「語彙全体としてのモデル指紋」が単一語 ratio の低さに隠れる可能性がある。
  補助として、AI 350文書の 7モデル識別 (7クラス分類) を語彙のみ /
  リズムのみで行い、多変量としてどちらがモデルを特定できるかを見る。
  識別精度が高い = その特徴空間でモデル間が分化している。

設計:
  - 多項ロジスティック回帰 (L2, C=1.0, lbfgs)。
  - stratified 5-fold x 5 repeats (seed=20260717)。
  - 語彙: 学習 fold 内で頻度フィルタ (AI学習文書 >=5 に出現) のみ。
    モデルラベルを使った選択はしない (選択リーク防止)。
  - リズム: 13指標、fold 内 impute (中央値) + z-score。
  - 指標: accuracy と macro-F1 (チャンス 1/7 ≈ 0.143)。
  - 感度: llama3.2-1b 除外 (6クラス、チャンス 1/6 ≈ 0.167)。

出力: results/h2_model_id.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

SEED = 20260717
MIN_DOC_COUNT = 5

REPO = Path(__file__).resolve().parent.parent.parent
DATA = REPO / "results" / "discrimination" / "dataset"
OUT = REPO / "results" / "discrimination"

RHYTHM_FEATURES = [
    "sent_len_char_mean", "sent_len_char_sd", "sent_len_char_cv",
    "sent_len_mora_mean", "sent_len_mora_sd", "sent_len_mora_cv",
    "burstiness_char", "burstiness_mora",
    "para_sent_cv",
    "taigendome_rate", "dewanaku_per_sentence", "comma_per_sentence",
    "ending_mattr20",
]


def impute_standardize(train, test):
    med = np.nanmedian(train, axis=0)
    tr = np.where(np.isnan(train), med, train)
    te = np.where(np.isnan(test), med, test)
    mu, sd = tr.mean(axis=0), tr.std(axis=0)
    sd[sd == 0] = 1.0
    return (tr - mu) / sd, (te - mu) / sd


def run(tag: str, docs: pd.DataFrame, X: sparse.csr_matrix) -> list[dict]:
    y = docs["model"].astype("category").cat.codes.to_numpy()
    R = docs[RHYTHM_FEATURES].to_numpy(dtype=np.float64)
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    rows = []
    for fold_i, (tr, te) in enumerate(rkf.split(np.zeros(len(y)), y)):
        y_tr, y_te = y[tr], y[te]
        # 語彙 (頻度フィルタのみ、ラベル不使用)
        dc = np.asarray(X[tr].sum(axis=0)).ravel()
        cols = np.nonzero((dc >= MIN_DOC_COUNT) & (dc <= len(tr) - MIN_DOC_COUNT))[0]
        Xl_tr, Xl_te = X[tr][:, cols], X[te][:, cols]
        clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                 max_iter=5000)
        clf.fit(Xl_tr, y_tr)
        pred = clf.predict(Xl_te)
        rows.append(dict(dataset=tag, fold=fold_i, features="lexical",
                         accuracy=accuracy_score(y_te, pred),
                         macro_f1=f1_score(y_te, pred, average="macro"),
                         n_features=len(cols)))
        # リズム
        Xr_tr, Xr_te = impute_standardize(R[tr], R[te])
        clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                 max_iter=5000)
        clf.fit(Xr_tr, y_tr)
        pred = clf.predict(Xr_te)
        rows.append(dict(dataset=tag, fold=fold_i, features="rhythm",
                         accuracy=accuracy_score(y_te, pred),
                         macro_f1=f1_score(y_te, pred, average="macro"),
                         n_features=len(RHYTHM_FEATURES)))
    return rows


def main() -> int:
    docs = pd.read_csv(DATA / "docs.csv")
    X = sparse.load_npz(DATA / "lexical_binary.npz").tocsr()
    ai = (docs.group == "ai").to_numpy()
    docs_ai = docs[ai].reset_index(drop=True)
    X_ai = X[ai]

    rows = run("ai7", docs_ai, X_ai)
    m6 = (docs_ai.model != "llama3.2-1b").to_numpy()
    rows += run("ai6_nollama", docs_ai[m6].reset_index(drop=True), X_ai[m6])

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "h2_model_id_folds.csv", index=False)
    agg = (df.groupby(["dataset", "features"])
             .agg(acc_mean=("accuracy", "mean"), acc_sd=("accuracy", "std"),
                  f1_mean=("macro_f1", "mean"), f1_sd=("macro_f1", "std"))
             .reset_index())
    agg.to_csv(OUT / "h2_model_id_summary.csv", index=False)
    print(agg.to_string(index=False))
    (OUT / "h2_model_id_metadata.json").write_text(json.dumps(dict(
        seed=SEED, cv="5x5 RepeatedStratifiedKFold", min_doc_count=MIN_DOC_COUNT,
        chance_ai7=1 / 7, chance_ai6=1 / 6), indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
