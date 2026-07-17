#!/usr/bin/env python3
"""exp028 Phase 3 H2: 指紋のレイヤー構造 (モデル間分散比の語彙 vs リズム比較)。

仮説 H2: リズム指紋はモデル間で均質、語彙指紋はモデル間で分化する。

指標 (Phase 2 analyze_groups.py と同一定義):
  between = 7モデルの文書平均値間の SD (ddof=1)
  within  = 各モデル内の文書間 SD (ddof=1) の平均
  ratio   = between / within。大きいほどモデル間で分化。

特徴:
  リズム = Phase 2 の 12指標 + ending_mattr20 (計13)。AI 350文書。
  語彙   = Phase 1.5 の FDR 有意 797語 (word_stats.csv sig_fdr=True) の
           文書出現 binary。per-model 平均 = そのモデルの文書出現率。
  注: 797語は「人間 vs AI」の全データ検定で選択された集合。H2 は
      モデル間構造の記述であり held-out 予測ではないため選択リークの
      問題は H1 と異なるが、選択基準 (人間vsAI差) がモデル間分散と
      無相関である保証はない (summary.md の要確認に記載)。
      感度分析として AI 文書出現数 >=10 の語に絞った集計も併記。

検定:
  1. per-feature permutation test: AI 350文書のモデルラベルをシャッフル
     (n=1000, seed=20260717) して ratio の帰無分布を作り、
     p = (1 + #{null >= obs}) / (n + 1)。binary 語と連続リズム指標で
     帰無分布の位置が異なるため、標準化 z = (obs - mean_null) / sd_null
     も算出し、語彙 vs リズムの群間比較は raw ratio と z の両方で行う。
  2. 群間比較: 語彙特徴群の ratio 分布 vs リズム特徴群の ratio 分布を
     Mann-Whitney U (両側, scipy)。効果量は rank-biserial r = 2*AUC - 1
     (= Cliff's delta と同値)。
  3. 予備所見の検証: リズム13指標を「コアリズム」(文長・burstiness・
     段落・読点・ではなく: 11) と「文末修辞」(taigendome_rate,
     ending_mattr20: 2) に分け、ratio と permutation p を対比。

感度: llama3.2-1b 除外 (6モデル) での再計算を併記 (縮退文書の影響確認)。

出典: permutation test の p 値の +1 補正は Phipson & Smyth (2010,
Stat Appl Genet Mol Biol 9:39)。Mann-Whitney U: scipy.stats.mannwhitneyu。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import mannwhitneyu

SEED = 20260717
N_PERM = 1000

REPO = Path(__file__).resolve().parent.parent.parent
DATA = REPO / "results" / "discrimination" / "dataset"
P15 = REPO / "results" / "lexical"
OUT = REPO / "results" / "discrimination"

RHYTHM_FEATURES = [
    "sent_len_char_mean", "sent_len_char_sd", "sent_len_char_cv",
    "sent_len_mora_mean", "sent_len_mora_sd", "sent_len_mora_cv",
    "burstiness_char", "burstiness_mora",
    "para_sent_cv",
    "taigendome_rate", "dewanaku_per_sentence", "comma_per_sentence",
    "ending_mattr20",
]
ENDING_RHETORIC = {"taigendome_rate", "ending_mattr20"}


def ratios_matrix(M: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """列 = 特徴。モデルごとの平均・SD から between/within ratio を返す。

    NaN は列ごとに除外 (nanmean / nanstd)。
    """
    models = np.unique(labels)
    means = np.stack([np.nanmean(M[labels == m], axis=0) for m in models])
    withins = np.stack([np.nanstd(M[labels == m], axis=0, ddof=1) for m in models])
    between = np.std(means, axis=0, ddof=1)
    within = withins.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(within > 0, between / within, np.nan)


def permutation_ratios(M: np.ndarray, labels: np.ndarray,
                       rng: np.random.Generator, n_perm: int) -> np.ndarray:
    """(n_perm, n_features) の帰無 ratio。ラベルのみシャッフル。"""
    out = np.empty((n_perm, M.shape[1]))
    lab = labels.copy()
    for i in range(n_perm):
        rng.shuffle(lab)
        out[i] = ratios_matrix(M, lab)
    return out


def perm_stats(obs: np.ndarray, null: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """p (+1補正, 片側: 分化が帰無より大きいか) と z を返す。"""
    p = (1 + (null >= obs).sum(axis=0)) / (null.shape[0] + 1)
    mu = null.mean(axis=0)
    sd = null.std(axis=0, ddof=1)
    z = np.where(sd > 0, (obs - mu) / sd, np.nan)
    return p, z


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """rank-biserial r = 2*P(x>y) - 1 (Cliff's delta と同値)。"""
    u = mannwhitneyu(x, y, alternative="two-sided").statistic
    return float(2 * u / (len(x) * len(y)) - 1)


def analyze(tag: str, docs_ai: pd.DataFrame, Xw: np.ndarray,
            words: list[str], word_ai_counts: np.ndarray,
            rng: np.random.Generator) -> tuple[pd.DataFrame, list[dict]]:
    labels = docs_ai["model"].to_numpy()
    R = docs_ai[RHYTHM_FEATURES].to_numpy(dtype=np.float64)

    obs_r = ratios_matrix(R, labels)
    obs_w = ratios_matrix(Xw, labels)
    null_r = permutation_ratios(R, labels, rng, N_PERM)
    null_w = permutation_ratios(Xw, labels, rng, N_PERM)
    p_r, z_r = perm_stats(obs_r, null_r)
    p_w, z_w = perm_stats(obs_w, null_w)

    feat_rows = []
    for i, f in enumerate(RHYTHM_FEATURES):
        feat_rows.append(dict(
            dataset=tag, feature=f,
            feature_type="rhythm_ending" if f in ENDING_RHETORIC else "rhythm_core",
            ratio=obs_r[i], perm_p=p_r[i], perm_z=z_r[i], ai_doc_count=np.nan))
    for j, w in enumerate(words):
        feat_rows.append(dict(
            dataset=tag, feature=w, feature_type="lexical",
            ratio=obs_w[j], perm_p=p_w[j], perm_z=z_w[j],
            ai_doc_count=int(word_ai_counts[j])))
    feat_df = pd.DataFrame(feat_rows)

    comp_rows = []
    lex = feat_df[feat_df.feature_type == "lexical"]
    rhy = feat_df[feat_df.feature_type.str.startswith("rhythm")]
    subsets = {
        "lex_all_vs_rhythm13": (lex, rhy),
        "lex_all_vs_rhythm_core11": (lex, rhy[rhy.feature_type == "rhythm_core"]),
        "lex_min10_vs_rhythm13": (lex[lex.ai_doc_count >= 10], rhy),
        "lex_min10_vs_rhythm_core11": (lex[lex.ai_doc_count >= 10],
                                       rhy[rhy.feature_type == "rhythm_core"]),
    }
    for name, (a, b) in subsets.items():
        for metric in ("ratio", "perm_z"):
            x = a[metric].dropna().to_numpy()
            y = b[metric].dropna().to_numpy()
            res = mannwhitneyu(x, y, alternative="two-sided")
            comp_rows.append(dict(
                dataset=tag, comparison=name, metric=metric,
                n_lexical=len(x), n_rhythm=len(y),
                lexical_median=float(np.median(x)),
                rhythm_median=float(np.median(y)),
                U=float(res.statistic), p=float(res.pvalue),
                rank_biserial=rank_biserial(x, y)))
    return feat_df, comp_rows


def main() -> int:
    t0 = time.time()
    docs = pd.read_csv(DATA / "docs.csv")
    X = sparse.load_npz(DATA / "lexical_binary.npz").tocsr()
    vocab = (DATA / "vocab.txt").read_text(encoding="utf-8").splitlines()
    vidx = {w: i for i, w in enumerate(vocab)}

    ws = pd.read_csv(P15 / "word_stats.csv")
    sig_words = ws[ws.sig_fdr == True]["word"].tolist()  # noqa: E712
    sig_words = [w for w in sig_words if w in vidx]
    cols = [vidx[w] for w in sig_words]

    ai_mask = (docs.group == "ai").to_numpy()
    docs_ai = docs[ai_mask].reset_index(drop=True)
    Xw_ai = X[ai_mask][:, cols].toarray().astype(np.float64)
    word_ai_counts = Xw_ai.sum(axis=0)

    rng = np.random.default_rng(SEED)
    feat_all, comp_all = analyze("ai7", docs_ai, Xw_ai, sig_words,
                                 word_ai_counts, rng)

    # 感度: llama3.2-1b 除外
    m6 = (docs_ai.model != "llama3.2-1b").to_numpy()
    feat6, comp6 = analyze("ai6_nollama", docs_ai[m6].reset_index(drop=True),
                           Xw_ai[m6], sig_words, Xw_ai[m6].sum(axis=0), rng)

    feat_df = pd.concat([feat_all, feat6], ignore_index=True)
    comp_df = pd.DataFrame(comp_all + comp6)
    feat_df.to_csv(OUT / "h2_feature_ratios.csv", index=False)
    comp_df.to_csv(OUT / "h2_group_comparison.csv", index=False)

    meta = dict(seed=SEED, n_perm=N_PERM, n_sig_words=len(sig_words),
                n_ai_docs=int(ai_mask.sum()),
                python=sys.version.split()[0], numpy=np.__version__,
                scipy=__import__("scipy").__version__, pandas=pd.__version__,
                elapsed_sec=round(time.time() - t0, 1))
    (OUT / "h2_run_metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # コンソール要約
    for tag in ("ai7", "ai6_nollama"):
        sub = feat_df[feat_df.dataset == tag]
        print(f"--- {tag} ---")
        rhy = sub[sub.feature_type.str.startswith("rhythm")]
        print(rhy[["feature", "feature_type", "ratio", "perm_p", "perm_z"]]
              .sort_values("ratio").to_string(index=False))
        lex = sub[sub.feature_type == "lexical"]
        print(f"lexical: n={len(lex)} median ratio={lex.ratio.median():.3f} "
              f"sig(perm p<0.05)={100 * (lex.perm_p < 0.05).mean():.1f}%")
    print(comp_df.to_string(index=False))
    print(f"elapsed: {meta['elapsed_sec']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
