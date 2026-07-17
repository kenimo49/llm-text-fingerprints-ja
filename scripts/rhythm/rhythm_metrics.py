#!/usr/bin/env python3
"""exp028 Phase 2: 日本語リズム系文体指標の文書ごと計測。

対象コーパス (読み取り専用):
  ~/repos/excess-vocabulary-ja/data/ai_samples/<model>/*.md   (7LLM x 50文書)
  ~/repos/excess-vocabulary-ja/data/human_corpus/{qiita,zenn}/<theme>/*.md

出力:
  results/rhythm/rhythm_metrics.csv  (文書 x 指標)

指標定義の出典:
- burstiness B = (σ − μ) / (σ + μ):
    Goh & Barabási (2008) "Burstiness and memory in complex systems",
    EPL 81, 48002 の burstiness parameter。文長系列に適用する用法は
    coji/natural-japanese (MIT) scripts/lint.py に倣う。
- モーラ近似 (読みカタカナから小書き仮名を直前拍に併合してカウント):
    coji/natural-japanese scripts/lint.py の方式に準拠。
    読みは MeCab + unidic-lite の pron フィールド (feature index 9)。
    促音「ッ」・長音「ー」・撥音「ン」は各1モーラ (標準的なモーラ定義)。
- 文分割 (。！？ + 改行) / 文長CV / 段落あたり文数CV / 体言止め判定
    (文末形態素の pos1 == 名詞):
    coji/natural-japanese scripts/lint.py の定義に準拠。

再現性: 本スクリプトに乱数依存なし。依存バージョンは results/environment.txt
および summary.md に記録。
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import MeCab

# ---------------------------------------------------------------- paths

CORPUS_ROOT = Path.home() / "repos" / "excess-vocabulary-ja" / "data"
AI_DIR = CORPUS_ROOT / "ai_samples"
HUMAN_DIR = CORPUS_ROOT / "human_corpus"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "rhythm"

# ---------------------------------------------------------------- markdown preprocessing

FENCE_RE = re.compile(r"^(```|~~~)")
HEADING_RE = re.compile(r"^#{1,6}\s")
TABLE_RE = re.compile(r"^\s*\|")
HR_RE = re.compile(r"^\s*([-*_]\s*){3,}$")
LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d{1,3}[.)]\s+)")
QUOTE_RE = re.compile(r"^\s*>+\s?")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
INLINE_CODE_RE = re.compile(r"`([^`]*)`")
HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>|<!--.*?-->", re.DOTALL)
URL_RE = re.compile(r"https?://\S+")
JA_CHAR_RE = re.compile(r"[ぁ-んァ-ヶ一-龯々〆ヵヶ]")
SENT_SPLIT_RE = re.compile(r"[。！？]")

# モーラに数えない小書き仮名 (直前拍に併合)。natural-japanese lint.py 準拠。
SMALL_KANA = set("ァィゥェォャュョヮ")


@dataclass
class Line:
    text: str
    is_list: bool


@dataclass
class Sentence:
    text: str
    is_list: bool


@dataclass
class Paragraph:
    sentences: list = field(default_factory=list)


def clean_inline(text: str) -> str:
    """インライン markdown 装飾を除去して本文テキストだけ残す。

    インラインコード span は中身を残す (文の一部として読まれるため)。
    → 要確認: 中身が英語識別子の場合モーラ計数に乗らない (読みなし=0モーラ)。
    """
    text = IMAGE_RE.sub("", text)
    text = LINK_RE.sub(r"\1", text)
    text = INLINE_CODE_RE.sub(r"\1", text)
    text = HTML_TAG_RE.sub("", text)
    text = URL_RE.sub("", text)
    text = text.replace("**", "").replace("__", "")
    return text.strip()


def parse_markdown(raw: str) -> list:
    """markdown → 段落リスト。コードブロック/見出し/表/罫線は除外。

    - 段落 = 空行区切りのブロック。連続する箇条書きは1つの段落 (リストブロック)。
    - 箇条書き項目はマーカーを外して文として保持 (is_list=True)。
      natural-japanese lint.py が改行を文境界に含める設計に合わせ、
      リスト行も文単位に含める。
    """
    lines = raw.split("\n")
    blocks: list[list[Line]] = []
    cur: list[Line] = []
    in_fence = False
    for ln in lines:
        if FENCE_RE.match(ln.strip()):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        stripped = ln.strip()
        if not stripped:
            if cur:
                blocks.append(cur)
                cur = []
            continue
        if HEADING_RE.match(stripped) or TABLE_RE.match(ln) or HR_RE.match(stripped):
            # 見出し・表・罫線は文ではないので除外。段落境界としても扱う。
            if cur:
                blocks.append(cur)
                cur = []
            continue
        is_list = bool(LIST_RE.match(ln))
        text = LIST_RE.sub("", ln) if is_list else ln
        text = QUOTE_RE.sub("", text)
        text = clean_inline(text)
        if text:
            cur.append(Line(text=text, is_list=is_list))
    if cur:
        blocks.append(cur)

    paragraphs: list[Paragraph] = []
    for block in blocks:
        para = Paragraph()
        for line in block:
            # 改行 + 。！？ を文境界とする (natural-japanese 準拠)。
            for seg in SENT_SPLIT_RE.split(line.text):
                seg = seg.strip()
                # 日本語文字を1つ以上含むものだけ文と数える
                # (英語のみの行・記号行を除外)
                if seg and JA_CHAR_RE.search(seg):
                    para.sentences.append(Sentence(text=seg, is_list=line.is_list))
        if para.sentences:
            paragraphs.append(para)
    return paragraphs


# ---------------------------------------------------------------- mecab helpers

_tagger = MeCab.Tagger()

PRON_INDEX = 9  # unidic-lite: [pos1,pos2,pos3,pos4,cType,cForm,lForm,lemma,orth,pron,...]
PUNCT_POS = {"補助記号", "記号", "空白"}


def tokenize(sentence: str):
    """(surface, pos1, pron) のリストを返す。"""
    out = []
    node = _tagger.parseToNode(sentence)
    while node:
        if node.surface:
            f = node.feature.split(",")
            pos1 = f[0]
            pron = f[PRON_INDEX] if len(f) > PRON_INDEX else ""
            if pron == "*":
                pron = ""
            out.append((node.surface, pos1, pron))
        node = node.next
    return out


def mora_count(tokens) -> int:
    """読みカタカナからモーラ数を近似。

    小書き仮名 (ャュョァィゥェォヮ) は直前拍に併合 (カウントしない)。
    ー・ッ・ンは各1モーラ。読みが得られない token (英語・記号) は0モーラ。
    → 要確認: 英語混じり文はモーラ長が過小になる。文字長系指標と併読する。
    """
    n = 0
    for _, pos1, pron in tokens:
        if pos1 in PUNCT_POS:
            continue
        for ch in pron:
            if ch in SMALL_KANA:
                continue
            n += 1
    return n


TRAILING_CLUTTER = set("」』）)】＞>”\"'…‥・：:；;、，,")


def final_morphemes(tokens):
    """文末の実質形態素 (句読点・括弧類を除く) を後ろから返す。"""
    content = [t for t in tokens if t[1] not in PUNCT_POS and t[0] not in TRAILING_CLUTTER]
    return content


# ---------------------------------------------------------------- metrics

def cv(values) -> float:
    m = statistics.mean(values)
    if m == 0:
        return float("nan")
    return statistics.pstdev(values) / m


def burstiness(values) -> float:
    """Goh & Barabási (2008) burstiness parameter B = (σ−μ)/(σ+μ)。

    σ は母標準偏差 (pstdev)。natural-japanese lint.py と同じ規約。
    """
    m = statistics.mean(values)
    s = statistics.pstdev(values)
    if (s + m) == 0:
        return float("nan")
    return (s - m) / (s + m)


MIN_SENTENCES = 5  # これ未満の文書は集計から除外 (natural-japanese の最小文数に準拠)


def compute_doc_metrics(raw: str) -> dict:
    paragraphs = parse_markdown(raw)
    sentences = [s for p in paragraphs for s in p.sentences]
    prose = [s for s in sentences if not s.is_list]

    row: dict = {
        "n_paragraphs": len(paragraphs),
        "n_sentences": len(sentences),
        "n_sentences_prose": len(prose),
    }
    if len(sentences) < MIN_SENTENCES:
        row["included"] = 0
        return row
    row["included"] = 1

    # --- 文長 (文字数 / モーラ近似) --- 全文 (地の文 + リスト項目)
    char_lens = [len(s.text.replace(" ", "").replace("　", "")) for s in sentences]
    tokenized = [tokenize(s.text) for s in sentences]
    mora_lens = [mora_count(t) for t in tokenized]

    row["sent_len_char_mean"] = statistics.mean(char_lens)
    row["sent_len_char_sd"] = statistics.pstdev(char_lens)
    row["sent_len_char_cv"] = cv(char_lens)
    row["sent_len_mora_mean"] = statistics.mean(mora_lens)
    row["sent_len_mora_sd"] = statistics.pstdev(mora_lens)
    row["sent_len_mora_cv"] = cv(mora_lens)

    # --- burstiness (文長系列) ---
    row["burstiness_char"] = burstiness(char_lens)
    row["burstiness_mora"] = burstiness(mora_lens)

    # --- 段落構造: 段落あたり文数の変動係数 ---
    if len(paragraphs) >= 3:
        row["para_sent_cv"] = cv([len(p.sentences) for p in paragraphs])
    else:
        row["para_sent_cv"] = float("nan")

    # --- 読点密度: 文あたり読点数 (、と，を読点と数える) ---
    commas = [s.text.count("、") + s.text.count("，") for s in sentences]
    row["comma_per_sentence"] = statistics.mean(commas)

    # --- 「ではなく」出現比率: 出現数 / 文数 ---
    full_text = "".join(s.text for s in sentences)
    row["dewanaku_per_sentence"] = full_text.count("ではなく") / len(sentences)

    # --- 体言止め率 / 文末bigram TTR: 地の文のみ ---
    # リスト項目は名詞で終わるのが書式上普通なので、修辞としての
    # 体言止め・文末表現の多様性は地の文で測る (本実験の判断、summary に明記)。
    if len(prose) >= MIN_SENTENCES:
        prose_tok = [tokenize(s.text) for s in prose]
        finals = [final_morphemes(t) for t in prose_tok]
        finals = [f for f in finals if f]
        if finals:
            taigen = sum(1 for f in finals if f[-1][1] == "名詞")
            row["taigendome_rate"] = taigen / len(finals)
            bigrams = []
            for f in finals:
                last2 = f[-2:]
                bigrams.append("|".join(x[0] + "/" + x[1] for x in last2))
            row["ending_bigram_ttr"] = len(set(bigrams)) / len(bigrams)
        else:
            row["taigendome_rate"] = float("nan")
            row["ending_bigram_ttr"] = float("nan")
    else:
        row["taigendome_rate"] = float("nan")
        row["ending_bigram_ttr"] = float("nan")

    return row


# ---------------------------------------------------------------- corpus walk

def iter_documents():
    """(doc_id, group, model, source, theme, path) を列挙。"""
    for model_dir in sorted(AI_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for f in sorted(model_dir.glob("*.md")):
            theme = f.stem.rsplit("_", 1)[0]
            yield (f"ai/{model}/{f.stem}", "ai", model, "ai", theme, f)
    for source in ("qiita", "zenn"):
        src_dir = HUMAN_DIR / source
        if not src_dir.is_dir():
            continue
        for theme_dir in sorted(src_dir.iterdir()):
            if not theme_dir.is_dir():
                continue
            for f in sorted(theme_dir.glob("*.md")):
                yield (
                    f"human/{source}/{theme_dir.name}/{f.stem}",
                    "human",
                    "human",
                    source,
                    theme_dir.name,
                    f,
                )


METRIC_COLS = [
    "sent_len_char_mean", "sent_len_char_sd", "sent_len_char_cv",
    "sent_len_mora_mean", "sent_len_mora_sd", "sent_len_mora_cv",
    "burstiness_char", "burstiness_mora",
    "para_sent_cv",
    "taigendome_rate", "dewanaku_per_sentence", "comma_per_sentence",
    "ending_bigram_ttr",
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "rhythm_metrics.csv"

    cols = ["doc_id", "group", "model", "source", "theme",
            "n_paragraphs", "n_sentences", "n_sentences_prose", "included"] + METRIC_COLS

    n_docs = 0
    n_included = 0
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for doc_id, group, model, source, theme, path in iter_documents():
            raw = path.read_text(encoding="utf-8", errors="replace")
            row = compute_doc_metrics(raw)
            row.update(doc_id=doc_id, group=group, model=model, source=source, theme=theme)
            for c in METRIC_COLS:
                v = row.get(c)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    row[c] = ""
                else:
                    row[c] = f"{v:.6f}"
            w.writerow(row)
            n_docs += 1
            n_included += row["included"]
            if n_docs % 50 == 0:
                print(f"  ... {n_docs} docs processed (last: {doc_id})", flush=True)

    # 依存バージョン記録
    import numpy, scipy, pandas  # noqa: F401  (analyze側で使用、ここで記録)
    try:
        from importlib.metadata import version
        env = {
            "python": sys.version,
            "mecab-python3": version("mecab-python3"),
            "unidic-lite": version("unidic-lite"),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "pandas": pandas.__version__,
        }
    except Exception as e:  # pragma: no cover
        env = {"error": str(e)}
    (OUT_DIR / "environment.txt").write_text(
        json.dumps(env, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"documents processed: {n_docs}, included (>= {MIN_SENTENCES} sentences): {n_included}")
    print(f"wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
