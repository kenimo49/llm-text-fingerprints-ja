#!/usr/bin/env python3
"""exp028 Phase 3: 文末bigram MATTR の計測 (Phase 2 の TTR 置き換え)。

Phase 2 要確認事項 #1 への対応:
  文末bigram TTR は文書長 (文数) に依存し、長い人間文書が不利に出る。
  moving-average TTR (MATTR) に置き換えて文書長を統制する。

手法出典:
  MATTR: Covington & McFall (2010) "Cutting the Gordian Knot: The
  Moving-Average Type-Token Ratio (MATTR)", Journal of Quantitative
  Linguistics 17(2), 94-100。原典は語トークン列への適用。本実験では
  文末bigram 系列 (地の文のみ) への適用に転用する (本実験の判断)。

窓の設計 (summary.md に明記):
  window = 20 文 (地の文の文末bigram 20個)。
  地の文文末が 20 個未満の文書は window = n (全系列の単純TTR) に縮小し、
  window_used 列に実際の窓幅を記録する。5 個未満は Phase 2 の
  MIN_SENTENCES に合わせて NaN (除外)。
  → 全文書に特徴値を持たせる必要 (H1 分類器) があるため「除外」ではなく
     「窓縮小」を採用。窓縮小文書 (n<20) の混入影響は、n>=20 の
     サブセット限定の群間比較を併記して確認する。

文末bigram の定義・前処理・文分割は Phase 2 rhythm_metrics.py をそのまま
import して再利用 (コーパス READ ONLY)。

出力: results/ending_mattr.csv
  doc_id, n_endings, window_used, mattr20
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PHASE2_SCRIPTS = Path(__file__).resolve().parent.parent / "rhythm"
sys.path.insert(0, str(PHASE2_SCRIPTS))

import rhythm_metrics as rm  # noqa: E402  (Phase 2 実装を再利用)

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "discrimination"
WINDOW = 20
MIN_ENDINGS = rm.MIN_SENTENCES  # 5 (Phase 2 と同一基準)


def ending_bigrams(raw: str) -> list[str]:
    """地の文の文末bigram 系列 (Phase 2 と同一定義、文書内の出現順)。"""
    paragraphs = rm.parse_markdown(raw)
    sentences = [s for p in paragraphs for s in p.sentences]
    prose = [s for s in sentences if not s.is_list]
    bigrams = []
    for s in prose:
        toks = rm.tokenize(s.text)
        finals = rm.final_morphemes(toks)
        if not finals:
            continue
        last2 = finals[-2:]
        bigrams.append("|".join(x[0] + "/" + x[1] for x in last2))
    return bigrams


def mattr(seq: list[str], window: int) -> tuple[float, int]:
    """MATTR。系列長 < window なら窓を系列長に縮小 (単純TTR)。

    返り値: (mattr値, 実際に使った窓幅)
    """
    n = len(seq)
    w = min(window, n)
    if w == 0:
        return float("nan"), 0
    if n <= w:
        return len(set(seq)) / n, n
    # sliding window の TTR 平均 (Covington & McFall 2010)
    total = 0.0
    from collections import Counter

    counts = Counter(seq[:w])
    total += len(counts) / w
    for i in range(n - w):
        out_tok, in_tok = seq[i], seq[i + w]
        counts[out_tok] -= 1
        if counts[out_tok] == 0:
            del counts[out_tok]
        counts[in_tok] += 1
        total += len(counts) / w
    return total / (n - w + 1), w


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "ending_mattr.csv"
    n_docs = 0
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["doc_id", "n_endings", "window_used", "mattr20"])
        for doc_id, group, model, source, theme, path in rm.iter_documents():
            raw = path.read_text(encoding="utf-8", errors="replace")
            seq = ending_bigrams(raw)
            n = len(seq)
            if n < MIN_ENDINGS:
                w.writerow([doc_id, n, "", ""])
            else:
                val, used = mattr(seq, WINDOW)
                w.writerow([doc_id, n, used, f"{val:.6f}"])
            n_docs += 1
            if n_docs % 200 == 0:
                print(f"  ... {n_docs} docs", flush=True)
    print(f"done: {n_docs} docs -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
