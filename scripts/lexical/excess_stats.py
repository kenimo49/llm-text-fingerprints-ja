#!/usr/bin/env python3
"""
excess_stats.py — document出現ベースの excess vocabulary 統計 (exp028 Phase 1.5)

phase1-methods-audit.md の P0/P1 修正リストを新パイプラインとして実装する:
  P0-1: 頻度単位を document 出現ベースに (token χ² の擬似反復を排除)
  P1-5: Kobak の頻度比 r = p/q と絶対差 δ = p−q を併記 (arXiv:2406.07016 本文 p.1)
  P1-6: Laplace 平滑化 p=(a+1)/(b+1) (arXiv:2406.07016 Methods §4.1)
  P2-11: 文書レベル bootstrap による 95% CI (B=1000, seed 固定)
  多重比較: FDR (Benjamini-Hochberg 1995) を主、Bonferroni 併記

旧手法 (token χ² + Bonferroni, excess-vocabulary-ja/results/statistical_tests.json の
651 有意語) が新手法で何語生き残るかを評価する。

入力 (READ ONLY):
  ../../results/lexical/doc_words_{ai,human}.jsonl  (build_doc_matrix.py の出力)
  ~/repos/excess-vocabulary-ja/results/statistical_tests.json  (旧結果)
出力:
  ../../results/lexical/word_stats.csv / word_stats.json / run_metadata.json / summary.md
"""

import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import scipy
from scipy import stats as sps

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / "results" / "lexical"
OLD_STATS = Path.home() / "repos" / "excess-vocabulary-ja" / "results" / "statistical_tests.json"

SEED = 20260717
N_BOOT = 1000
ALPHA = 0.05
JST = timezone(timedelta(hours=9))


# ── データ読み込み ──────────────────────────────────────────────

def load_docs(path: Path):
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def build_binary_matrix(docs, vocab_index):
    """文書 × 語 の binary 出現行列 (Kobak Methods §4.1 の CountVectorizer(binary=True) 相当)"""
    X = np.zeros((len(docs), len(vocab_index)), dtype=bool)
    for i, d in enumerate(docs):
        for w in d["words"]:
            j = vocab_index.get(w)
            if j is not None:
                X[i, j] = True
    return X


# ── 統計 ────────────────────────────────────────────────────────

def doc_chi2(a, n_ai, c, n_hum):
    """document 出現数の 2×2 分割表で χ² (Yates 補正は scipy デフォルトに従う)。
    各文書を独立試行として数えるため token χ² の擬似反復がない。"""
    table = np.array([[a, n_ai - a], [c, n_hum - c]])
    if table.min() < 0:
        return np.nan, np.nan
    try:
        chi2, p, _, _ = sps.chi2_contingency(table)
    except ValueError:
        # 期待度数 0 (行/列和が 0) → Fisher 正確検定にフォールバック
        _, p = sps.fisher_exact(table)
        chi2 = np.nan
    return chi2, p


def bh_fdr(pvals):
    """Benjamini-Hochberg (1995) の adjusted p (q値)。単調性を保証する標準実装。"""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.minimum(ranked, 1.0)
    return q


def smoothed_props(counts, n_docs):
    """Laplace 平滑化した document 頻度 p=(a+1)/(b+1)。
    出典: Kobak et al. arXiv:2406.07016 Methods §4.1 (監査md §1.1)。"""
    return (counts + 1.0) / (n_docs + 1.0)


def bootstrap_ci(X_ai, X_hum, word_cols, n_boot, seed):
    """文書レベル bootstrap で r=p/q と δ=p−q の percentile 95% CI。
    AI 文書と人間文書をそれぞれ復元抽出し、Laplace 平滑化込みで再計算する。
    word_cols: CI を計算する語の列 index 配列。"""
    rng = np.random.default_rng(seed)
    n_ai, n_hum = X_ai.shape[0], X_hum.shape[0]
    Xa = X_ai[:, word_cols].astype(np.float64)
    Xh = X_hum[:, word_cols].astype(np.float64)
    r_samples = np.empty((n_boot, len(word_cols)))
    d_samples = np.empty((n_boot, len(word_cols)))
    for b in range(n_boot):
        ai_idx = rng.integers(0, n_ai, n_ai)
        hum_idx = rng.integers(0, n_hum, n_hum)
        a = Xa[ai_idx].sum(axis=0)
        c = Xh[hum_idx].sum(axis=0)
        p = smoothed_props(a, n_ai)
        q = smoothed_props(c, n_hum)
        r_samples[b] = p / q
        d_samples[b] = p - q
        if (b + 1) % 100 == 0:
            print(f"  bootstrap {b + 1}/{n_boot}", flush=True)
    r_ci = np.percentile(r_samples, [2.5, 97.5], axis=0)
    d_ci = np.percentile(d_samples, [2.5, 97.5], axis=0)
    return r_ci, d_ci


# ── メイン ──────────────────────────────────────────────────────

def main():
    print("excess_stats: document-occurrence excess vocabulary pipeline", flush=True)
    ai_docs = load_docs(RESULTS / "doc_words_ai.jsonl")
    hum_docs = load_docs(RESULTS / "doc_words_human.jsonl")
    n_ai, n_hum = len(ai_docs), len(hum_docs)
    print(f"  corpora: AI={n_ai} docs, human={n_hum} docs", flush=True)

    old = json.loads(OLD_STATS.read_text(encoding="utf-8"))
    old_tests = old["chi2_tests"]
    old_by_word = {t["word"]: t for t in old_tests}
    old_sig = [t for t in old_tests if str(t["significant_bonferroni"]) == "True"]
    print(f"  old method: {len(old_tests)} tested, {len(old_sig)} Bonferroni-significant", flush=True)

    # 語ユニバース = 旧手法で検定された全 1684 語 (生き残り評価は 651 語サブセットで行う)
    vocab = [t["word"] for t in old_tests]
    vocab_index = {w: j for j, w in enumerate(vocab)}
    X_ai = build_binary_matrix(ai_docs, vocab_index)
    X_hum = build_binary_matrix(hum_docs, vocab_index)

    a_counts = X_ai.sum(axis=0).astype(int)   # 語を含む AI 文書数
    c_counts = X_hum.sum(axis=0).astype(int)  # 語を含む人間文書数
    p_hat = smoothed_props(a_counts.astype(float), n_ai)
    q_hat = smoothed_props(c_counts.astype(float), n_hum)
    r_hat = p_hat / q_hat          # Kobak excess frequency ratio
    d_hat = p_hat - q_hat          # Kobak excess frequency gap (δ)

    # document χ²
    print("  chi2 (document 2x2) ...", flush=True)
    chi2_arr = np.empty(len(vocab))
    pval_arr = np.empty(len(vocab))
    for j in range(len(vocab)):
        chi2_arr[j], pval_arr[j] = doc_chi2(a_counts[j], n_ai, c_counts[j], n_hum)
        if (j + 1) % 400 == 0:
            print(f"    {j + 1}/{len(vocab)}", flush=True)

    q_fdr = bh_fdr(pval_arr)                                   # 主: BH-FDR
    p_bonf = np.minimum(pval_arr * len(vocab), 1.0)            # 併記: Bonferroni
    sig_fdr = q_fdr < ALPHA
    sig_bonf = p_bonf < ALPHA

    # bootstrap CI: 全 1684 語で計算 (行列演算なので語数の影響は小さい)
    print(f"  bootstrap CI (B={N_BOOT}, seed={SEED}) ...", flush=True)
    all_cols = np.arange(len(vocab))
    r_ci, d_ci = bootstrap_ci(X_ai, X_hum, all_cols, N_BOOT, SEED)

    # 生き残り判定
    rows = []
    for j, w in enumerate(vocab):
        o = old_by_word[w]
        old_sig_flag = str(o["significant_bonferroni"]) == "True"
        direction = "excess" if r_hat[j] > 1 else ("deficit" if r_hat[j] < 1 else "equal")
        old_direction = "excess" if o["excess_score"] > 0 else "deficit"
        # CI ベースの判定: r の 95% CI が 1 を跨がない (方向つき)
        ci_excl = bool(r_ci[0, j] > 1.0) or bool(r_ci[1, j] < 1.0)
        survives_fdr = bool(sig_fdr[j]) and direction == old_direction
        survives_bonf = bool(sig_bonf[j]) and direction == old_direction
        survives_ci = bool(ci_excl and direction == old_direction)
        rows.append({
            "word": w,
            "old_excess_score": o["excess_score"],
            "old_sig_bonferroni": old_sig_flag,
            "old_direction": old_direction,
            "ai_doc_count": int(a_counts[j]),
            "human_doc_count": int(c_counts[j]),
            "p_ai": round(float(p_hat[j]), 6),
            "q_human": round(float(q_hat[j]), 6),
            "ratio_r": round(float(r_hat[j]), 4),
            "delta": round(float(d_hat[j]), 6),
            "r_ci_low": round(float(r_ci[0, j]), 4),
            "r_ci_high": round(float(r_ci[1, j]), 4),
            "delta_ci_low": round(float(d_ci[0, j]), 6),
            "delta_ci_high": round(float(d_ci[1, j]), 6),
            "chi2_doc": round(float(chi2_arr[j]), 4) if np.isfinite(chi2_arr[j]) else None,
            "p_value": float(pval_arr[j]),
            "q_fdr": float(q_fdr[j]),
            "p_bonferroni": float(p_bonf[j]),
            "sig_fdr": bool(sig_fdr[j]),
            "sig_bonferroni": bool(sig_bonf[j]),
            "new_direction": direction,
            "survives_fdr": survives_fdr,
            "survives_bonferroni": survives_bonf,
            "survives_ci": survives_ci,
        })

    # 保存: CSV + JSON
    csv_path = RESULTS / "word_stats.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS / "word_stats.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  saved {csv_path}", flush=True)

    # メタデータ (再現性: P1-8)
    def _git_head(repo):
        try:
            return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip()
        except Exception:
            return None
    meta = {
        "generated_at": datetime.now(JST).isoformat(),
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "alpha": ALPHA,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "mecab_dict": "unidic-lite",
        "corpus": {"ai_docs": n_ai, "human_docs": n_hum},
        "source_repo_head": _git_head(Path.home() / "repos" / "excess-vocabulary-ja"),
        "old_stats_file": str(OLD_STATS),
        "smoothing": "Laplace p=(a+1)/(b+1) per Kobak arXiv:2406.07016 Methods 4.1",
        "frequency_unit": "document occurrence (binary per doc)",
        "multiple_comparison": "BH-FDR primary, Bonferroni secondary",
        "ci_method": "percentile bootstrap over documents, both corpora resampled",
    }
    (RESULTS / "run_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # サマリ集計
    old651 = [r for r in rows if r["old_sig_bonferroni"]]
    surv_fdr = [r for r in old651 if r["survives_fdr"]]
    surv_bonf = [r for r in old651 if r["survives_bonferroni"]]
    surv_ci = [r for r in old651 if r["survives_ci"]]
    surv_all3 = [r for r in old651 if r["survives_fdr"] and r["survives_bonferroni"] and r["survives_ci"]]
    pos651 = [r for r in old651 if r["old_direction"] == "excess"]
    neg651 = [r for r in old651 if r["old_direction"] == "deficit"]
    summary_counts = {
        "old_651": len(old651),
        "old_651_excess": len(pos651),
        "old_651_deficit": len(neg651),
        "survive_fdr": len(surv_fdr),
        "survive_fdr_excess": sum(1 for r in surv_fdr if r["old_direction"] == "excess"),
        "survive_fdr_deficit": sum(1 for r in surv_fdr if r["old_direction"] == "deficit"),
        "survive_bonferroni": len(surv_bonf),
        "survive_ci": len(surv_ci),
        "survive_all3": len(surv_all3),
        "new_sig_fdr_total_1684": int(sig_fdr.sum()),
        "new_sig_bonf_total_1684": int(sig_bonf.sum()),
        # 「新たに有意」は素直に sig_fdr のみで数える (方向一致は旧発見の生存判定にのみ意味を持つ)
        "newly_sig_not_in_651": sum(1 for r in rows if r["sig_fdr"] and not r["old_sig_bonferroni"]),
        # 参考: 旧手法での方向 (非有意時の符号) と新手法の方向が一致するものに限った数
        "newly_sig_not_in_651_direction_matched": sum(1 for r in rows if r["survives_fdr"] and not r["old_sig_bonferroni"]),
    }
    (RESULTS / "summary_counts.json").write_text(
        json.dumps(summary_counts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary_counts, ensure_ascii=False, indent=2), flush=True)
    print("OK — write summary.md separately from these results", flush=True)


if __name__ == "__main__":
    sys.exit(main())
