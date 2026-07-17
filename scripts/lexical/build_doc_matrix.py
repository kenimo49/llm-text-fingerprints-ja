#!/usr/bin/env python3
"""
build_doc_matrix.py — 文書ごとの語彙集合を構築する (exp028 Phase 1.5)

excess-vocabulary-ja の tokenize_mecab.py と同一のトークナイズ規則
(品詞フィルタ / ストップワード / unidic lemma / Markdownノイズ除去 / 100字未満スキップ)
を再実装し、コーパス集計を Counter 合算ではなく「文書ごとの unique 語集合」で出力する。

これは phase1-methods-audit.md P0-1 (頻度単位を document 出現ベースへ) の実装。
出典: Kobak et al., Science Advances 11(27), 2025 (arXiv:2406.07016) Methods §4.1 は
binary な文書×語の出現行列 (CountVectorizer(binary=True)) を用いる。

入力: ~/repos/excess-vocabulary-ja/data/{ai_samples,human_corpus} (READ ONLY)
出力: ../../results/lexical/doc_words_ai.jsonl, ../../results/lexical/doc_words_human.jsonl
      1行 = 1文書 {"path", "sub", "n_tokens", "words": [unique lemmas]}
"""

import json
import re
import sys
from pathlib import Path

import MeCab
import unidic_lite

# ── 定数: excess-vocabulary-ja/scripts/tokenize_mecab.py と同一 ──────────
SRC_REPO = Path.home() / "repos" / "excess-vocabulary-ja"
AI_DIR = SRC_REPO / "data" / "ai_samples"
HUMAN_DIR = SRC_REPO / "data" / "human_corpus"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "lexical"

TARGET_POS = {"名詞", "動詞", "形容詞", "副詞", "接続詞"}
EXCLUDE_NOUN_SUB = {"数詞", "非自立可能", "代名詞", "助数詞"}
STOPWORDS = {
    "する", "為る", "いる", "居る", "ある", "有る", "なる", "成る",
    "できる", "出来る", "れる", "られる",
    "こと", "事", "もの", "物", "ため", "為", "よう", "様",
    "それ", "これ", "ここ", "其れ", "此れ", "此処",
    "の", "に", "は", "を", "が", "と", "で", "も", "か",
    "や", "へ", "から", "まで", "より", "etc", "##",
}
MD_NOISE = re.compile(
    r'```[\s\S]*?```|`[^`]+`|!\[.*?\]\(.*?\)|\[.*?\]\(.*?\)|#{1,6}\s|[*_~`>|]|\|.*\|'
)
URL_PATTERN = re.compile(r'https?://\S+')
MIN_DOC_CHARS = 100  # tokenize_mecab.py:145 と同一


def clean_text(text: str) -> str:
    text = MD_NOISE.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    return re.sub(r'\s+', ' ', text).strip()


class MeCabTokenizer:
    def __init__(self):
        self.tagger = MeCab.Tagger(f"-d {unidic_lite.DICDIR}")
        self.tagger.parse("")

    def extract_words(self, text: str) -> list:
        """品詞フィルタ + ストップワード除外した原形(lemma)リスト。
        tokenize_mecab.py の extract_words と同一挙動。"""
        words = []
        node = self.tagger.parseToNode(text)
        while node:
            if node.surface:
                features = node.feature.split(",")
                pos = features[0] if len(features) > 0 else ""
                pos_sub = features[1] if len(features) > 1 else ""
                base = features[7] if len(features) > 7 else node.surface
                if base == "*" or not base:
                    base = node.surface
                ok = (
                    pos in TARGET_POS
                    and not (pos == "名詞" and pos_sub in EXCLUDE_NOUN_SUB)
                    and not (pos == "動詞" and pos_sub == "非自立可能")
                    and base not in STOPWORDS
                    and node.surface not in STOPWORDS
                    and not (len(base) == 1 and not base.isalpha())
                )
                if ok:
                    words.append(base)
            node = node.next
        return words


def process_corpus(directory: Path, tokenizer: MeCabTokenizer, out_path: Path, label: str) -> dict:
    md_files = sorted(directory.rglob("*.md"))  # sorted: 再現性のため走査順を固定
    n_docs = 0
    n_skipped = 0
    total_tokens = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for i, f in enumerate(md_files):
            text = f.read_text(encoding="utf-8", errors="ignore")
            if len(text) < MIN_DOC_CHARS:
                n_skipped += 1
                continue
            cleaned = clean_text(text)
            words = tokenizer.extract_words(cleaned)
            rec = {
                "path": str(f.relative_to(SRC_REPO)),
                "sub": f.parent.name,
                "n_tokens": len(words),
                "words": sorted(set(words)),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_docs += 1
            total_tokens += len(words)
            if (i + 1) % 100 == 0:
                print(f"  [{label}] {i + 1}/{len(md_files)} files...", flush=True)
    print(f"  [{label}] done: {n_docs} docs ({n_skipped} skipped <{MIN_DOC_CHARS} chars), "
          f"{total_tokens} filtered tokens -> {out_path.name}", flush=True)
    return {"docs": n_docs, "skipped": n_skipped, "tokens": total_tokens}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok = MeCabTokenizer()
    print("build_doc_matrix: per-document binary word sets", flush=True)
    stats = {}
    stats["ai"] = process_corpus(AI_DIR, tok, OUT_DIR / "doc_words_ai.jsonl", "AI")
    stats["human"] = process_corpus(HUMAN_DIR, tok, OUT_DIR / "doc_words_human.jsonl", "Human")
    (OUT_DIR / "doc_matrix_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK", flush=True)


if __name__ == "__main__":
    sys.exit(main())
