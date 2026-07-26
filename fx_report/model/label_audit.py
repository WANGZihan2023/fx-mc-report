"""
证据人工标注（label_audit）—— 与流水线 direction / category 词表对齐。

方向约定（与 EvidenceItem.direction / CSV model_direction 一致）：
  up      = +1 = 推高分析报价路径最高值（看涨货币走强）
  down    = -1 = 压制峰值（看涨货币走弱）
  neutral =  0 = 中性 / 不影响峰值判断
  unclear = 人工无法判断（仅 human_direction）

agree：人工方向与模型方向是否一致 → yes / no / unsure
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from fx_report.news.classify import CATEGORY_RULES
from fx_report.news.llm import VALID_CATEGORY

# ---------------------------------------------------------------------------
# 允许值（UI / CSV / 文档共用，勿另造词表）
# ---------------------------------------------------------------------------

HUMAN_DIRECTIONS: tuple[str, ...] = ("up", "down", "neutral", "unclear")
MODEL_DIRECTIONS: tuple[str, ...] = ("up", "down", "neutral")
AGREE_VALUES: tuple[str, ...] = ("yes", "no", "unsure")

# 与 classify.CATEGORY_RULES + llm.VALID_CATEGORY 对齐，并保留模板偶发类
_EXTRA_CATEGORIES = ("unclassified", "dairy", "other")
LABEL_CATEGORIES: tuple[str, ...] = tuple(
    sorted(
        set(VALID_CATEGORY)
        | {cat for cat, _ in CATEGORY_RULES}
        | set(_EXTRA_CATEGORIES)
    )
)

LABEL_AUDIT_COLUMNS: tuple[str, ...] = (
    "statement_id",
    "title",
    "url",
    "model_category",
    "model_direction",
    "human_direction",
    "human_category",
    "agree",
)

DIRECTION_ZH: dict[str, str] = {
    "up": "看涨（推高分析报价）",
    "down": "看跌（压制峰值）",
    "neutral": "中性",
    "unclear": "无法判断",
}

AGREE_ZH: dict[str, str] = {
    "yes": "一致",
    "no": "不一致",
    "unsure": "不确定",
}

CATEGORY_ZH: dict[str, str] = {
    "geopolitics": "地缘风险",
    "oil": "油价",
    "cpi": "通胀/CPI",
    "fed": "美联储",
    "ecb": "欧央行",
    "boe": "英央行",
    "boj": "日央行",
    "rba": "澳储行",
    "rbnz": "纽储行",
    "boc": "加央行",
    "snb": "瑞央行",
    "pboc": "人行/中间价",
    "china_growth": "中国增长",
    "china_iron": "铁矿石",
    "yields": "收益率/利差",
    "growth": "增长/就业",
    "positioning": "仓位/情绪",
    "other": "其他",
    "unclassified": "未分类",
    "dairy": "乳制品",
}


def project_output_dir() -> Path:
    # fx_report/model/label_audit.py → repo root
    return Path(__file__).resolve().parents[2] / "output"


def label_audit_filename(pair: str, as_of: date | None = None) -> str:
    pair_safe = (pair or "PAIR").replace("/", "")
    d = (as_of or date.today()).isoformat()
    return f"label_audit_{pair_safe}_{d}.csv"


def label_audit_path(pair: str, as_of: date | None = None, out_dir: Path | None = None) -> Path:
    root = out_dir or project_output_dir()
    return root / label_audit_filename(pair, as_of)


def normalize_direction(raw: Any) -> str:
    """Map numeric / string direction → up|down|neutral|unclear|''."""
    if raw is None or raw == "":
        return ""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if raw > 0:
            return "up"
        if raw < 0:
            return "down"
        return "neutral"
    s = str(raw).strip().lower()
    aliases = {
        "up": "up",
        "+1": "up",
        "1": "up",
        "bullish": "up",
        "down": "down",
        "-1": "down",
        "bearish": "down",
        "neutral": "neutral",
        "0": "neutral",
        "unclear": "unclear",
        "unknown": "unclear",
        "n/a": "unclear",
    }
    return aliases.get(s, s if s in HUMAN_DIRECTIONS else "")


def compute_agree(model_direction: Any, human_direction: Any) -> str:
    """
    Auto-fill agree when both sides are set.
    unclear / empty human → unsure; exact match → yes; else no.
    """
    m = normalize_direction(model_direction)
    h = normalize_direction(human_direction)
    if not h or h == "unclear":
        return "unsure"
    if not m:
        return "unsure"
    return "yes" if m == h else "no"


def category_label(cat: str) -> str:
    zh = CATEGORY_ZH.get(cat, "")
    return f"{cat}（{zh}）" if zh else cat


def direction_label(d: str) -> str:
    zh = DIRECTION_ZH.get(d, "")
    return f"{d} — {zh}" if zh else d


def help_markdown(*, pair: str = "AUD/USD", bullish: str = "AUD") -> str:
    """Short Chinese guide for the expander / docs."""
    base = pair.split("/")[0] if "/" in pair else pair[:3]
    quote = pair.split("/")[1] if "/" in pair else ""
    return f"""
### 你在标什么？

每一行是模型实际用到的一条**证据语句**（标题/摘要）。你要判断：这条新闻对**分析报价**（看涨 **{bullish}** → 报价升高 = {bullish} 走强）意味着什么。

当前分析口径：**{pair}**｜看涨货币：**{bullish}**

### 只读列（模型猜测，不用改）

| 列 | 含义 |
|---|---|
| `model_direction` | 模型方向：`up` / `down` / `neutral` |
| `model_category` | 模型分类（驱动类别） |

### 你要填的列

**`human_direction`**（必填优先）— 你的方向判断，允许值：

| 值 | 含义 |
|---|---|
| `up` | 推高分析报价路径峰值（{bullish} 走强） |
| `down` | 压制峰值（{bullish} 走弱） |
| `neutral` | 中性，几乎不影响峰值 |
| `unclear` | 信息不足，你无法判断 |

**举例（看涨={bullish}，报价={pair}）：**

1. 「RBA 偏鹰 / 暗示加息」→ 澳元走强 → 若看涨 AUD 且报价为 AUD/USD → **`up`**；若报价为 USD/AUD → **`down`**
2. 「美联储偏鹰 / 加息预期升温」→ 美元走强 → 看涨 AUD 时对 AUD/USD → **`down`**
3. 「铁矿石大涨、中国需求回暖」→ 利多澳元 → AUD/USD 看涨 AUD → **`up`**
4. 「仅提到假期休市、无方向信息」→ **`neutral`** 或 **`unclear`**

**`human_category`** — 从与模型相同的类别表里选（如 `rba` / `fed` / `geopolitics` / `oil` / `china_growth` / `yields` …）。

**`agree`** — 人工方向是否与模型一致：`yes` / `no` / `unsure`。两边都填好后可自动计算（`unclear` → `unsure`）。

### 小技巧

- 先点「一键按模型预填再改」，只改你不同意的几条。
- 方向始终相对**分析报价升高**，不是相对「新闻标题里出现的货币名」单独判断。
- {base}/{quote} 时：BASE 走强 → `up`；QUOTE 走强 → `down`。
""".strip()


def demo_evidence_rows(*, pair: str = "AUD/USD", bullish: str = "AUD") -> list[dict[str, Any]]:
    """Practice statements when a run produced no news evidence."""
    # Model guesses intentionally imperfect so labeling practice is meaningful.
    samples = [
        {
            "statement_id": "DEMO-01",
            "title": "RBA holds rates, Bullock sounds hawkish on inflation persistence",
            "url": "https://www.rba.gov.au/",
            "category": "rba",
            "dir": 1 if bullish == "AUD" else (-1 if bullish == "USD" else 1),
        },
        {
            "statement_id": "DEMO-02",
            "title": "Fed officials signal fewer rate cuts as US data stays firm",
            "url": "https://www.federalreserve.gov/",
            "category": "fed",
            "dir": -1 if bullish == "AUD" else (1 if bullish == "USD" else -1),
        },
        {
            "statement_id": "DEMO-03",
            "title": "Iron ore prices jump on stronger China steel demand",
            "url": "",
            "category": "china_iron",
            "dir": 1 if bullish == "AUD" else (-1 if bullish == "USD" else 1),
        },
        {
            "statement_id": "DEMO-04",
            "title": "Markets closed for holiday; no policy surprises expected",
            "url": "",
            "category": "other",
            "dir": 0,
        },
        {
            "statement_id": "DEMO-05",
            "title": "Geopolitical tensions rise in Middle East shipping lanes",
            "url": "",
            "category": "geopolitics",
            "dir": -1 if bullish == "AUD" else (1 if bullish == "USD" else -1),
        },
    ]
    # Soft pair hint in title note via category only; keep generic for any pair.
    _ = pair
    return samples


def evidence_rows_to_audit_df(
    evidence_rows: Sequence[dict[str, Any]] | Sequence[Any],
) -> pd.DataFrame:
    """Build label_audit DataFrame (human columns empty). Includes url when present."""
    rows: list[dict[str, Any]] = []
    for item in evidence_rows:
        if hasattr(item, "id"):
            sid = getattr(item, "statement_id", "") or getattr(item, "id", "")
            title = getattr(item, "title", "") or ""
            cat = getattr(item, "category", "") or ""
            direction = getattr(item, "direction", "")
            url = getattr(item, "url", "") or ""
        elif isinstance(item, dict):
            sid = item.get("statement_id") or item.get("id") or ""
            title = item.get("title") or item.get("statement") or ""
            cat = item.get("category") or item.get("model_category") or ""
            direction = item.get("dir", item.get("direction", item.get("model_direction", "")))
            url = item.get("url") or ""
        else:
            continue
        rows.append(
            {
                "statement_id": sid,
                "title": title,
                "url": url,
                "model_category": cat,
                "model_direction": normalize_direction(direction) or "",
                "human_direction": "",
                "human_category": "",
                "agree": "",
            }
        )
    return pd.DataFrame(rows, columns=list(LABEL_AUDIT_COLUMNS))


def merge_human_labels(
    base_df: pd.DataFrame,
    labels: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """
    Apply per-statement_id human_* / agree from UI state.
    labels[sid] = {human_direction, human_category, agree}
    """
    out = base_df.copy()
    for col in ("human_direction", "human_category", "agree"):
        if col not in out.columns:
            out[col] = ""
    for i, row in out.iterrows():
        sid = str(row.get("statement_id") or "")
        lab = labels.get(sid) or {}
        hd = normalize_direction(lab.get("human_direction", row.get("human_direction", "")))
        hc = str(lab.get("human_category", row.get("human_category", "")) or "")
        ag = str(lab.get("agree", "") or "").strip().lower()
        if ag not in AGREE_VALUES:
            ag = compute_agree(row.get("model_direction", ""), hd)
        out.at[i, "human_direction"] = hd
        out.at[i, "human_category"] = hc
        out.at[i, "agree"] = ag
    return out


def prefill_from_model(df: pd.DataFrame) -> pd.DataFrame:
    """Copy model_* → human_* and recompute agree (all yes unless model empty)."""
    out = df.copy()
    for i, row in out.iterrows():
        md = normalize_direction(row.get("model_direction", ""))
        mc = str(row.get("model_category") or "")
        # Map model neutral → human neutral; never copy into unclear
        hd = md if md in ("up", "down", "neutral") else ""
        out.at[i, "human_direction"] = hd
        out.at[i, "human_category"] = mc if mc in LABEL_CATEGORIES else (mc or "other")
        out.at[i, "agree"] = compute_agree(md, hd)
    return out


def save_label_audit(df: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure column order
    cols = [c for c in LABEL_AUDIT_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in cols]
    out = df[cols + extra]
    out.to_csv(path, index=False)
    return path


def load_label_audit(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=list(LABEL_AUDIT_COLUMNS))
    df = pd.read_csv(path, dtype=str).fillna("")
    for c in LABEL_AUDIT_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[list(LABEL_AUDIT_COLUMNS)]


def empty_reason_message(
    *,
    evidence_n: int = 0,
    fetched: int = 0,
    quality: str = "",
    news_keys_present: bool = False,
) -> str:
    """Explain why labeling list is empty and what to do."""
    bits = [
        "本次运行没有可标注的证据语句。",
        f"evidence_n={evidence_n}，fetched={fetched}，quality=`{quality or 'n/a'}`。",
    ]
    if quality == "news_empty_no_prior" or evidence_n == 0:
        if not news_keys_present and fetched == 0:
            bits.append(
                "常见原因：未配置新闻 API Key（NewsAPI / Finnhub），或 RSS 未抓到相关头条。"
            )
            bits.append(
                "解决：侧栏「API 配置」填写 `NEWSAPI_KEY` 或 `FINNHUB_API_KEY` 后重新运行；"
                "也可先加载下方「练习样例」熟悉标注。"
            )
        else:
            bits.append(
                "已尝试抓取但未能分类出可用证据（相关度不足或方向无法判定）。"
                "可换货币对重跑，或先用练习样例练习标注流程。"
            )
    return " ".join(bits)
