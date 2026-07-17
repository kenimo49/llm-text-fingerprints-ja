#!/usr/bin/env python3
"""exp028 Phase 2: リズム指標の群間比較 (人間 vs AI全体 / モデル別)。

入力:  results/rhythm/rhythm_metrics.csv (rhythm_metrics.py の出力)
出力:  results/rhythm/group_comparison.csv   (人間 vs AI全体)
       results/rhythm/model_comparison.csv   (モデル別)
       results/rhythm/summary.md             (レポート)

統計:
- Cohen's d = (mean_AI − mean_human) / pooled SD (二標本、文書レベル)。
  pooled SD = sqrt(((n1−1)s1² + (n2−1)s2²) / (n1+n2−2))。標本SD (ddof=1)。
  正の d = AI の方が大きい。
- d の 95% CI は文書単位の bootstrap (n=2000, seed=20260717) percentile 法。
- Mann-Whitney U (両側) を参考値として併記 (分布の非正規性対策)。
  多重比較補正: Holm 法 (13指標)。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20260717
N_BOOT = 2000

RESULTS = Path(__file__).resolve().parent.parent.parent / "results" / "rhythm"

METRICS = [
    ("sent_len_char_mean", "文長平均 (文字)"),
    ("sent_len_char_sd", "文長SD (文字)"),
    ("sent_len_char_cv", "文長CV (文字)"),
    ("sent_len_mora_mean", "文長平均 (モーラ近似)"),
    ("sent_len_mora_sd", "文長SD (モーラ近似)"),
    ("sent_len_mora_cv", "文長CV (モーラ近似)"),
    ("burstiness_char", "burstiness (文字長)"),
    ("burstiness_mora", "burstiness (モーラ長)"),
    ("para_sent_cv", "段落あたり文数CV"),
    ("taigendome_rate", "体言止め率 (地の文)"),
    ("dewanaku_per_sentence", "「ではなく」/文"),
    ("comma_per_sentence", "読点/文"),
    ("ending_bigram_ttr", "文末bigram TTR (地の文)"),
]

MODEL_ORDER = [
    "claude-3-haiku", "claude-sonnet-4", "claude-opus-4",
    "gpt-3.5-turbo", "gpt-4o", "gpt-oss-20b", "llama3.2-1b",
]


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """d = (mean(x) − mean(y)) / pooled SD。x=AI, y=human。"""
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * x.std(ddof=1) ** 2 + (ny - 1) * y.std(ddof=1) ** 2)
                 / (nx + ny - 2))
    if sp == 0:
        return float("nan")
    return (x.mean() - y.mean()) / sp


def boot_ci_d(x: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    ds = np.empty(N_BOOT)
    for i in range(N_BOOT):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        ds[i] = cohens_d(xb, yb)
    return np.nanpercentile(ds, 2.5), np.nanpercentile(ds, 97.5)


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}"


def main() -> int:
    df = pd.read_csv(RESULTS / "rhythm_metrics.csv")
    df = df[df["included"] == 1].copy()
    print(f"included docs: {len(df)} "
          f"(human={len(df[df.group=='human'])}, ai={len(df[df.group=='ai'])})")

    rng = np.random.default_rng(SEED)
    human = df[df.group == "human"]
    ai = df[df.group == "ai"]

    # ---------- 人間 vs AI全体 ----------
    rows = []
    pvals = []
    for col, label in METRICS:
        h = human[col].dropna().to_numpy()
        a = ai[col].dropna().to_numpy()
        d = cohens_d(a, h)
        lo, hi = boot_ci_d(a, h, rng)
        u, p = stats.mannwhitneyu(a, h, alternative="two-sided")
        rows.append({
            "metric": col, "label": label,
            "n_human": len(h), "n_ai": len(a),
            "human_mean": h.mean(), "human_sd": h.std(ddof=1),
            "ai_mean": a.mean(), "ai_sd": a.std(ddof=1),
            "cohens_d": d, "d_ci_lo": lo, "d_ci_hi": hi,
            "mw_p": p,
        })
        pvals.append(p)
        print(f"  [human vs ai] {col}: d={d:+.3f} [{lo:+.3f},{hi:+.3f}]")

    # Holm 補正
    order = np.argsort(pvals)
    m = len(pvals)
    adj = [None] * m
    prev = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        prev = max(prev, val)
        adj[idx] = prev
    for r, p_adj in zip(rows, adj):
        r["mw_p_holm"] = p_adj

    comp = pd.DataFrame(rows)
    comp.to_csv(RESULTS / "group_comparison.csv", index=False)
    print("wrote group_comparison.csv")

    # ---------- モデル別 ----------
    mrows = []
    for col, label in METRICS:
        h = human[col].dropna().to_numpy()
        for model in MODEL_ORDER:
            sub = ai[ai.model == model][col].dropna().to_numpy()
            if len(sub) == 0:
                continue
            d = cohens_d(sub, h)
            mrows.append({
                "metric": col, "label": label, "model": model,
                "n": len(sub), "mean": sub.mean(), "sd": sub.std(ddof=1),
                "cohens_d_vs_human": d,
            })
        print(f"  [per-model] {col} done")
    mcomp = pd.DataFrame(mrows)
    mcomp.to_csv(RESULTS / "model_comparison.csv", index=False)
    print("wrote model_comparison.csv")

    # ---------- モデル間ばらつき (H2予備検分用の所見) ----------
    # 指標ごとに: モデル平均間SD (between) と モデル内SDの平均 (within)、
    # その比 (between/within)。比が小さいほど「モデル間で均質」。
    drows = []
    for col, label in METRICS:
        means = []
        withins = []
        for model in MODEL_ORDER:
            sub = ai[ai.model == model][col].dropna().to_numpy()
            if len(sub) >= 2:
                means.append(sub.mean())
                withins.append(sub.std(ddof=1))
        between = np.std(means, ddof=1)
        within = np.mean(withins)
        drows.append({
            "metric": col, "label": label,
            "between_model_sd": between,
            "within_model_sd_mean": within,
            "between_within_ratio": between / within if within else float("nan"),
            "model_means_min": np.min(means), "model_means_max": np.max(means),
        })
    disp = pd.DataFrame(drows)
    disp.to_csv(RESULTS / "model_dispersion.csv", index=False)
    print("wrote model_dispersion.csv")

    # ---------- summary.md ----------
    write_summary(df, comp, mcomp, disp)
    print("wrote summary.md")
    return 0


def write_summary(df, comp, mcomp, disp):
    import json
    env = {}
    envp = RESULTS / "environment.txt"
    if envp.exists():
        env = json.loads(envp.read_text())

    human = df[df.group == "human"]
    ai = df[df.group == "ai"]

    lines = []
    w = lines.append
    w("# exp028 Phase 2: リズム系文体指標の計測結果")
    w("")
    w("計測日: 2026-07-17 / スクリプト: `scripts/rhythm/`")
    w("")
    w("## データ")
    w("")
    w(f"- コーパス: `~/repos/excess-vocabulary-ja/data/` (読み取り専用で再利用)")
    w(f"- 人間: {len(human)} 文書 (qiita={len(human[human.source=='qiita'])}, "
      f"zenn={len(human[human.source=='zenn'])})")
    w(f"- AI: {len(ai)} 文書 (7モデル x 各50、最小文数フィルタ後)")
    excl = pd.read_csv(RESULTS / "rhythm_metrics.csv")
    w(f"- 除外: 文数<5 の文書 {len(excl[excl.included==0])} 件 "
      f"(除外内訳は rhythm_metrics.csv の included=0 行)")
    w("")
    w("## 手法")
    w("")
    w("- markdown 前処理: コードブロック・見出し・表・罫線・画像・URL を除去、"
      "リンクとインラインコードは中身を残す。箇条書きはマーカーを外して文として保持。")
    w("- 文分割: 。！？ + 改行 (coji/natural-japanese lint.py 準拠)。"
      "日本語文字を含まない文は除外。")
    w("- モーラ近似: MeCab + unidic-lite の pron フィールド (カタカナ発音) から、"
      "小書き仮名 (ャュョァィゥェォヮ) を直前拍に併合してカウント "
      "(natural-japanese 準拠)。ー・ッ・ンは各1モーラ。"
      "読みが得られない token (英語識別子等) は 0 モーラ。")
    w("- burstiness B = (σ−μ)/(σ+μ)、σ は母標準偏差 "
      "(出典: Goh & Barabási 2008, EPL 81:48002。文長系列への適用は natural-japanese に倣う)。")
    w("- 体言止め: 文末の実質形態素 (句読点・括弧除く) の pos1 が 名詞。"
      "リスト項目は名詞終わりが書式上普通なので **地の文のみ** で算出。"
      "文末bigram TTR も同様に地の文のみ。")
    w("- Cohen's d: 二標本 pooled SD、文書レベル。正の d = AI が大。"
      f"95% CI は bootstrap (n={N_BOOT}, seed={SEED})。"
      "Mann-Whitney U (両側) + Holm 補正 (13指標) を参考併記。")
    w("")
    w("### 依存バージョン")
    w("")
    w("```json")
    w(json.dumps(env, indent=2, ensure_ascii=False))
    w("```")
    w("")
    w("## 人間 vs AI全体 (文書レベル)")
    w("")
    w("| 指標 | 人間 mean±SD | AI mean±SD | Cohen's d [95% CI] | MW p (Holm) |")
    w("|---|---|---|---|---|")
    for _, r in comp.sort_values("cohens_d", key=lambda s: s.abs(), ascending=False).iterrows():
        w(f"| {r.label} | {fmt(r.human_mean)} ± {fmt(r.human_sd)} "
          f"| {fmt(r.ai_mean)} ± {fmt(r.ai_sd)} "
          f"| **{r.cohens_d:+.2f}** [{r.d_ci_lo:+.2f}, {r.d_ci_hi:+.2f}] "
          f"| {r.mw_p_holm:.2e} |")
    w("")
    w("## モデル別 (mean ± SD、括弧内は vs 人間の Cohen's d)")
    w("")
    header = "| 指標 | 人間 |" + "".join(f" {m} |" for m in MODEL_ORDER)
    w(header)
    w("|---" * (len(MODEL_ORDER) + 2) + "|")
    hmap = comp.set_index("metric")
    for col, label in METRICS:
        cells = [f"| {label} ", f"| {fmt(hmap.loc[col,'human_mean'])}±{fmt(hmap.loc[col,'human_sd'],2)} "]
        for m in MODEL_ORDER:
            sel = mcomp[(mcomp.metric == col) & (mcomp.model == m)]
            if len(sel):
                r = sel.iloc[0]
                cells.append(f"| {fmt(r['mean'])}±{fmt(r.sd,2)} (d={r.cohens_d_vs_human:+.1f}) ")
            else:
                cells.append("| — ")
        w("".join(cells) + "|")
    w("")
    w("## モデル間ばらつき (H2 予備検分)")
    w("")
    w("between = 7モデル平均値間のSD、within = モデル内SDの平均。"
      "ratio = between/within が小さいほどモデル間で均質。")
    w("")
    w("| 指標 | between | within | ratio | モデル平均の範囲 |")
    w("|---|---|---|---|---|")
    for _, r in disp.sort_values("between_within_ratio").iterrows():
        w(f"| {r.label} | {fmt(r.between_model_sd)} | {fmt(r.within_model_sd_mean)} "
          f"| {fmt(r.between_within_ratio,2)} | {fmt(r.model_means_min)} 〜 {fmt(r.model_means_max)} |")
    w("")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
