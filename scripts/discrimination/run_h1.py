#!/usr/bin/env python3
"""exp028 Phase 3 H1: 語彙 vs リズムの人間/AI判別力比較 (AUC)。

仮説 H1: 日本語AI生成文の人間/AI判別力は語彙特徴とリズム特徴で異なる。

分類器: ロジスティック回帰 (L2, C=1.0, sklearn 1.7.2)。
特徴セット:
  lexical  — binary 語出現 (選択は fold 内、下記)
  rhythm   — リズム13指標 (Phase 2 の 12 + ending_mattr20)
  combined — 両方の連結

リーク防止 (設計の根幹):
  語彙特徴の選択 (最低文書数フィルタ + χ²[Yates] + BH-FDR q<0.05) は
  各 CV の学習 fold 内のデータだけで行う。Phase 1.5 の「FDR有意797語」は
  全データで選択された集合なので使わない (選択バイアスで AUC が過大になる)。
  リズム特徴の欠損補完 (中央値) と標準化も学習 fold の統計のみ使用。

CV: stratified 5-fold x 20 repeats (sklearn RepeatedStratifiedKFold,
    random_state=20260717)。全特徴セット・全条件で同一の分割を共有し、
    fold 単位の paired 差を取れるようにする。

AUC 集計: 100 fold の平均±SD。95% CI は fold AUC の bootstrap
    (B=10000, seed=20260717, percentile)。fold 間は学習データが重複する
    ため独立ではなく、CI は近似 (summary.md に明記)。repeat 単位
    (5-fold 平均 x 20) の集計も併記。

条件:
  main         — 全 1,046 文書
  lenadj       — 長さ共変量 (log1p文数, log1p総文字数) を学習 fold 内で
                 線形回帰し、特徴を残差化してから分類。
                 length-only (共変量2つだけの LR) をベースラインに併記。
                 リズム指標の優位が「文書長の代理」でないことの確認。
  sens_degen   — llama3.2-1b 縮退文書 (人間p99基準, prepare_dataset.py) を除外
  sens_nollama — llama3.2-1b 全50文書を除外

出典:
  BH-FDR: Benjamini & Hochberg (1995)。Yates 補正 χ²: Phase 1.5 と同一
  (scipy chi2_contingency デフォルト相当をベクトル化)。
  MATTR: Covington & McFall (2010) — compute_mattr.py 参照。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import chi2 as chi2_dist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

SEED = 20260717
N_SPLITS = 5
N_REPEATS = 20
MIN_DOC_COUNT = 5      # 学習 fold 内の最低文書数フィルタ
FDR_ALPHA = 0.05
BOOT_B = 10000

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
COVARIATES = ["cov_log_sentences", "cov_log_chars"]


# ---------------------------------------------------------------- selection

def yates_chi2_pvals(X: sparse.csr_matrix, y: np.ndarray) -> np.ndarray:
    """語ごとの 2x2 (文書出現 x クラス) Yates 補正 χ² p 値 (ベクトル化)。"""
    n = len(y)
    n1 = int(y.sum())
    n0 = n - n1
    a = np.asarray(X[y == 1].sum(axis=0)).ravel().astype(np.float64)  # AI で出現
    c = np.asarray(X[y == 0].sum(axis=0)).ravel().astype(np.float64)  # 人間で出現
    b = n1 - a
    d = n0 - c
    row1 = a + c          # 出現文書数
    row0 = b + d
    denom = row1 * row0 * n1 * n0
    diff = np.abs(a * d - b * c) - n / 2.0
    diff = np.maximum(diff, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = n * diff**2 / denom
    stat = np.where(denom > 0, stat, 0.0)
    return chi2_dist.sf(stat, 1)


def bh_fdr_mask(pvals: np.ndarray, alpha: float) -> np.ndarray:
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = alpha * np.arange(1, m + 1) / m
    below = ranked <= thresh
    mask = np.zeros(m, dtype=bool)
    if below.any():
        k = np.max(np.nonzero(below)[0])
        mask[order[: k + 1]] = True
    return mask


def select_words(X_tr: sparse.csr_matrix, y_tr: np.ndarray) -> np.ndarray:
    """学習 fold 内での語選択。列 index を返す。"""
    doc_counts = np.asarray(X_tr.sum(axis=0)).ravel()
    n = X_tr.shape[0]
    eligible = (doc_counts >= MIN_DOC_COUNT) & (doc_counts <= n - MIN_DOC_COUNT)
    idx = np.nonzero(eligible)[0]
    pv = yates_chi2_pvals(X_tr[:, idx], y_tr)
    mask = bh_fdr_mask(pv, FDR_ALPHA)
    if not mask.any():  # フォールバック (実際には起きない想定)
        mask = pv <= np.sort(pv)[min(100, len(pv)) - 1]
    return idx[mask]


# ---------------------------------------------------------------- helpers

def impute_standardize(train: np.ndarray, test: np.ndarray):
    """学習 fold の中央値で欠損補完し、学習 fold の mean/sd で z-score。"""
    med = np.nanmedian(train, axis=0)
    tr = np.where(np.isnan(train), med, train)
    te = np.where(np.isnan(test), med, test)
    mu = tr.mean(axis=0)
    sd = tr.std(axis=0)
    sd[sd == 0] = 1.0
    return (tr - mu) / sd, (te - mu) / sd


def residualize(train: np.ndarray, test: np.ndarray,
                cov_tr: np.ndarray, cov_te: np.ndarray):
    """学習 fold で fit した線形回帰 (切片あり) の残差に置き換える。"""
    A_tr = np.column_stack([np.ones(len(cov_tr)), cov_tr])
    A_te = np.column_stack([np.ones(len(cov_te)), cov_te])
    beta, *_ = np.linalg.lstsq(A_tr, train, rcond=None)
    return train - A_tr @ beta, test - A_te @ beta


def fit_auc(X_tr, y_tr, X_te, y_te, sparse_input=False) -> float:
    solver = "liblinear" if sparse_input else "lbfgs"
    clf = LogisticRegression(penalty="l2", C=1.0, solver=solver, max_iter=5000)
    clf.fit(X_tr, y_tr)
    score = clf.decision_function(X_te)
    return roc_auc_score(y_te, score)


# ---------------------------------------------------------------- main loop

def run_condition(name: str, docs: pd.DataFrame, X: sparse.csr_matrix,
                  adjust_length: bool) -> list[dict]:
    y = docs["y"].to_numpy()
    R = docs[RHYTHM_FEATURES].to_numpy(dtype=np.float64)
    C = docs[COVARIATES].to_numpy(dtype=np.float64)

    rkf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                  random_state=SEED)
    rows = []
    for fold_i, (tr, te) in enumerate(rkf.split(np.zeros(len(y)), y)):
        repeat, fold = divmod(fold_i, N_SPLITS)
        y_tr, y_te = y[tr], y[te]

        # --- 語彙選択 (fold 内) ---
        sel = select_words(X[tr], y_tr)
        Xl_tr = X[tr][:, sel].toarray().astype(np.float64)
        Xl_te = X[te][:, sel].toarray().astype(np.float64)

        # --- リズム (fold 内 impute + z-score) ---
        Xr_tr, Xr_te = impute_standardize(R[tr], R[te])

        cov_tr_z, cov_te_z = impute_standardize(C[tr], C[te])

        if adjust_length:
            Xl_tr, Xl_te = residualize(Xl_tr, Xl_te, C[tr], C[te])
            Xr_tr, Xr_te = residualize(Xr_tr, Xr_te, C[tr], C[te])
            rows.append(dict(condition=name, repeat=repeat, fold=fold,
                             features="length_only",
                             auc=fit_auc(cov_tr_z, y_tr, cov_te_z, y_te),
                             n_words=0))

        auc_lex = fit_auc(Xl_tr, y_tr, Xl_te, y_te)
        auc_rhy = fit_auc(Xr_tr, y_tr, Xr_te, y_te)
        auc_com = fit_auc(np.hstack([Xl_tr, Xr_tr]), y_tr,
                          np.hstack([Xl_te, Xr_te]), y_te)

        rows.append(dict(condition=name, repeat=repeat, fold=fold,
                         features="lexical", auc=auc_lex, n_words=len(sel)))
        rows.append(dict(condition=name, repeat=repeat, fold=fold,
                         features="rhythm", auc=auc_rhy, n_words=0))
        rows.append(dict(condition=name, repeat=repeat, fold=fold,
                         features="combined", auc=auc_com, n_words=len(sel)))

        if fold_i % 10 == 9:
            print(f"[{name}] fold {fold_i + 1}/{N_SPLITS * N_REPEATS} "
                  f"lex={auc_lex:.3f} rhy={auc_rhy:.3f} com={auc_com:.3f} "
                  f"words={len(sel)}", flush=True)
    return rows


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator,
                 B: int = BOOT_B) -> tuple[float, float]:
    n = len(values)
    means = values[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate(fold_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    summaries, diffs = [], []
    for cond, sub in fold_df.groupby("condition", sort=False):
        piv = sub.pivot_table(index=["repeat", "fold"], columns="features",
                              values="auc")
        for feat in piv.columns:
            v = piv[feat].to_numpy()
            lo, hi = bootstrap_ci(v, rng)
            rep_means = piv[feat].groupby(level="repeat").mean()
            summaries.append(dict(
                condition=cond, features=feat,
                auc_mean=v.mean(), auc_sd=v.std(ddof=1),
                ci_lo=lo, ci_hi=hi,
                repeat_mean=rep_means.mean(), repeat_sd=rep_means.std(ddof=1),
                n_folds=len(v),
            ))
        pairs = [("rhythm", "lexical"), ("combined", "lexical"),
                 ("combined", "rhythm")]
        if "length_only" in piv.columns:
            pairs += [("lexical", "length_only"), ("rhythm", "length_only")]
        for f1, f2 in pairs:
            if f1 not in piv.columns or f2 not in piv.columns:
                continue
            d = (piv[f1] - piv[f2]).to_numpy()
            lo, hi = bootstrap_ci(d, rng)
            diffs.append(dict(condition=cond, comparison=f"{f1}-{f2}",
                              diff_mean=d.mean(), diff_sd=d.std(ddof=1),
                              ci_lo=lo, ci_hi=hi,
                              frac_folds_positive=float((d > 0).mean())))
    return pd.DataFrame(summaries), pd.DataFrame(diffs)


def main() -> int:
    t0 = time.time()
    docs = pd.read_csv(DATA / "docs.csv")
    X = sparse.load_npz(DATA / "lexical_binary.npz").tocsr()
    assert X.shape[0] == len(docs)

    conditions = [
        ("main", np.ones(len(docs), dtype=bool), False),
        ("lenadj", np.ones(len(docs), dtype=bool), True),
        ("sens_degen", (docs.llama_degenerate == 0).to_numpy(), False),
        ("sens_nollama", (docs.model != "llama3.2-1b").to_numpy(), False),
    ]

    all_rows = []
    for name, mask, adjust in conditions:
        sub = docs[mask].reset_index(drop=True)
        print(f"=== condition {name}: {len(sub)} docs "
              f"(AI {int(sub.y.sum())}) adjust={adjust} ===", flush=True)
        all_rows += run_condition(name, sub, X[mask], adjust)

    fold_df = pd.DataFrame(all_rows)
    fold_df.to_csv(OUT / "h1_auc_folds.csv", index=False)
    summary, diffs = aggregate(fold_df)
    summary.to_csv(OUT / "h1_auc_summary.csv", index=False)
    diffs.to_csv(OUT / "h1_auc_diffs.csv", index=False)

    meta = dict(
        seed=SEED, n_splits=N_SPLITS, n_repeats=N_REPEATS,
        min_doc_count=MIN_DOC_COUNT, fdr_alpha=FDR_ALPHA,
        bootstrap_B=BOOT_B,
        python=sys.version.split()[0],
        numpy=np.__version__,
        sklearn=__import__("sklearn").__version__,
        scipy=__import__("scipy").__version__,
        pandas=pd.__version__,
        elapsed_sec=round(time.time() - t0, 1),
    )
    (OUT / "h1_run_metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(summary.to_string(index=False))
    print(diffs.to_string(index=False))
    print(f"elapsed: {meta['elapsed_sec']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
