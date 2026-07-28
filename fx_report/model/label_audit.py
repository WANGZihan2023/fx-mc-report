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


def direction_to_int(raw: Any) -> int | None:
    """
    Map direction token → EvidenceItem.direction int.
    unclear / empty → None (caller should keep model direction).
    """
    d = normalize_direction(raw)
    if d == "up":
        return 1
    if d == "down":
        return -1
    if d == "neutral":
        return 0
    return None


def agree_rate_stats(
    df: pd.DataFrame | None,
    *,
    is_demo: bool = False,
) -> dict[str, Any]:
    """
    Human vs model direction agree rate from a label_audit frame / session edits.

    agree_rate = n_yes / (n_yes + n_no); None when no decisive labels yet.
    Does not invent metrics for empty / practice-only rows.
    """
    empty: dict[str, Any] = {
        "n_rows": 0,
        "n_labeled": 0,
        "n_yes": 0,
        "n_no": 0,
        "n_unsure": 0,
        "agree_rate": None,
        "has_labels": False,
        "is_demo": bool(is_demo),
        "caption": "尚无人工方向标注，不显示同意率。",
    }
    if df is None or getattr(df, "empty", True):
        return empty

    work = df.copy()
    for col in ("model_direction", "human_direction", "agree"):
        if col not in work.columns:
            work[col] = ""

    # Refresh agree from directions when human_direction is set
    agrees: list[str] = []
    n_labeled = 0
    for _, row in work.iterrows():
        hd = normalize_direction(row.get("human_direction", ""))
        if not hd:
            agrees.append("")
            continue
        n_labeled += 1
        ag = str(row.get("agree") or "").strip().lower()
        if ag not in AGREE_VALUES:
            ag = compute_agree(row.get("model_direction", ""), hd)
        agrees.append(ag)

    n_yes = sum(1 for a in agrees if a == "yes")
    n_no = sum(1 for a in agrees if a == "no")
    n_unsure = sum(1 for a in agrees if a == "unsure")
    decisive = n_yes + n_no
    rate = (n_yes / decisive) if decisive > 0 else None

    if n_labeled == 0:
        caption = "尚无人工方向标注，不显示同意率。"
        if is_demo:
            caption = "练习样例未填方向 — 不计入真实同意率。"
    elif rate is None:
        caption = (
            f"已标 {n_labeled} 条，但均为 unsure/unclear，暂无同意率。"
            + ("（练习样例）" if is_demo else "")
        )
    else:
        caption = (
            f"同意率 {100 * rate:.0f}%（{n_yes}/{decisive} 条有明确对错"
            f"；另 unsure={n_unsure}）"
            + (" · 练习样例，非正式指标" if is_demo else "")
        )

    return {
        "n_rows": int(len(work)),
        "n_labeled": int(n_labeled),
        "n_yes": int(n_yes),
        "n_no": int(n_no),
        "n_unsure": int(n_unsure),
        "agree_rate": rate,
        "has_labels": n_labeled > 0,
        "is_demo": bool(is_demo),
        "caption": caption,
    }


def labels_dict_from_audit_df(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """statement_id → {human_direction, human_category, agree}."""
    out: dict[str, dict[str, str]] = {}
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        sid = str(row.get("statement_id") or "")
        if not sid:
            continue
        out[sid] = {
            "human_direction": normalize_direction(row.get("human_direction", "")),
            "human_category": str(row.get("human_category") or ""),
            "agree": str(row.get("agree") or "").strip().lower(),
        }
    return out


def apply_human_labels_to_evidence(
    evidence_rows: Sequence[dict[str, Any]],
    labels: dict[str, dict[str, str]] | pd.DataFrame,
) -> tuple[list[dict[str, Any]], int]:
    """
    Override direction (+ optional category) on evidence dicts using human labels.

    Matching key: statement_id, else id.
    human unclear/empty → keep model direction (no override).
    Returns (new_rows, n_overridden).
    """
    if isinstance(labels, pd.DataFrame):
        lab_map = labels_dict_from_audit_df(labels)
    else:
        lab_map = labels or {}

    out: list[dict[str, Any]] = []
    n_overridden = 0
    for raw in evidence_rows:
        row = dict(raw)
        sid = str(row.get("statement_id") or row.get("id") or "")
        lab = lab_map.get(sid) or {}
        hd = normalize_direction(lab.get("human_direction", ""))
        d_int = direction_to_int(hd)
        if d_int is not None:
            row["dir"] = d_int
            row["direction"] = d_int
            n_overridden += 1
        hc = str(lab.get("human_category") or "").strip()
        if hc and hc in LABEL_CATEGORIES:
            row["category"] = hc
        out.append(row)
    return out, n_overridden


def evidence_dicts_to_items(rows: Sequence[dict[str, Any]]) -> list[Any]:
    """Rebuild EvidenceItem list from session / diagnostics dicts."""
    from fx_report.model.weights import EvidenceItem

    items: list[EvidenceItem] = []
    for row in rows:
        direction = row.get("direction", row.get("dir", 0))
        try:
            direction_i = int(direction)
        except (TypeError, ValueError):
            direction_i = direction_to_int(direction) or 0
        items.append(
            EvidenceItem(
                id=str(row.get("id") or row.get("statement_id") or ""),
                title=str(row.get("title") or ""),
                direction=direction_i,
                strength=float(row.get("strength") or 0.0),
                freshness=float(row.get("freshness") if row.get("freshness") is not None else 1.0),
                unpriced=float(row.get("unpriced") if row.get("unpriced") is not None else 1.0),
                category=str(row.get("category") or row.get("model_category") or "other"),
                note=str(row.get("note") or ""),
                strength_label=str(row.get("strength_label") or row.get("label") or ""),
                strength_breakdown=dict(row.get("strength_breakdown") or {}),
                source_tier=str(row.get("source_tier") or ""),
                surprise=str(row.get("surprise") or ""),
                scope=str(row.get("scope") or ""),
                statement_id=str(row.get("statement_id") or row.get("id") or ""),
                url=str(row.get("url") or ""),
                is_prior=bool(row.get("is_prior", False)),
            )
        )
    return items


def recompute_score_and_scenarios(
    evidence: Sequence[Any],
    base_scenarios: Sequence[Any],
    *,
    score_to_mu_a: float,
    score_to_sigma_b: float,
    evidence_logit_scale: float,
    scenario_temperature: float,
    max_scenario_shift: float,
) -> dict[str, Any]:
    """Recompute S / μ / σ× / scenario weights after human direction overrides."""
    from fx_report.model.weights import (
        ScenarioSpec,
        apply_evidence_to_scenarios,
        evidence_score,
    )

    items = list(evidence)
    if items and isinstance(items[0], dict):
        items = evidence_dicts_to_items(items)  # type: ignore[arg-type]

    scenarios_in: list[ScenarioSpec] = []
    for s in base_scenarios:
        if isinstance(s, ScenarioSpec):
            scenarios_in.append(s)
        elif isinstance(s, dict):
            scenarios_in.append(
                ScenarioSpec(
                    name=str(s.get("name") or ""),
                    weight=float(s.get("weight") or 0.0),
                    mu_annual=float(s.get("mu_annual") or 0.0),
                    sigma_mult=float(s.get("sigma_mult") or 1.0),
                    expected_jumps=float(s.get("expected_jumps") or 0.0),
                    jump_mean=float(s.get("jump_mean") or 0.0),
                    jump_std=float(s.get("jump_std") or 0.0),
                    narrative=str(s.get("narrative") or ""),
                )
            )

    score = evidence_score(items)
    mu_shift = float(score_to_mu_a) * score
    sigma_extra = 1.0 + float(score_to_sigma_b) * abs(score)
    scenarios = apply_evidence_to_scenarios(
        scenarios_in,
        score,
        logit_scale=float(evidence_logit_scale),
        temperature=float(scenario_temperature),
        max_shift=float(max_scenario_shift),
    )
    return {
        "score_S": score,
        "mu_annual_shift": mu_shift,
        "sigma_mult_extra": sigma_extra,
        "scenarios_adjusted": scenarios,
        "evidence": items,
    }


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


def railway_env_checklist_markdown(*, news_keys_present: bool = False) -> str:
    """Checklist of Railway / vault env vars that unlock real news evidence."""
    news_mark = "✓ 本机/会话已检测到新闻 Key" if news_keys_present else "□ 建议填"
    return f"""
**Railway Variables / 本机 vault 建议检查：**

| 变量 | 用途 | 状态提示 |
|------|------|----------|
| `NEWSAPI_KEY` 或 `FINNHUB_API_KEY` | **References / 证据条数**的主要来源（无 Key 时仅靠央行+Google News RSS） | {news_mark} |
| `TAVILY_API_KEY` / `BRAVE_SEARCH_API_KEY` | AI 检索员网页搜索（给 LLM 更多原料） | 可选 |
| `GROQ_API_KEY` / `DEEPSEEK_API_KEY` / `LLM_API_KEY` | hybrid 证据判定与语句抽取（**不**单独增加 URL 条数） | 可选但推荐 |
| `LLM_BASE_URL` | DeepSeek 须为 `https://api.deepseek.com/v1`（通道选 DeepSeek 会自动填） | DeepSeek 必对 |
| `FRED_API_KEY` | 行情增强 | 可选 |
| `APP_PASSWORD`（或 `FX_REPORT_PASSWORD`） | 访问口令 | 云端强烈建议 |

无新闻 Key 时系统仍会抓 **Fed/RBA/ECB/BOE 等官方 RSS** + Google News 公开 RSS；
相关度过滤后可能只剩 0–1 条——属诚实空/稀薄证据，不是 LLM 坏了。
**填 DeepSeek 不会自动多出 References**；要更多链接请填 NewsAPI/Finnhub（或 Tavily）。
""".strip()


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
                "常见原因：未配置新闻 API Key（NewsAPI / Finnhub），且免费 RSS 未抓到相关头条。"
            )
            bits.append(
                "解决：在 Railway Variables 或侧栏「API / AI Key」填写 "
                "`NEWSAPI_KEY` / `FINNHUB_API_KEY`（LLM/DeepSeek 只做判定，不会凭空加链接），"
                "保存后重新运行；也可先加载下方「练习样例」熟悉标注。"
            )
        elif not news_keys_present and fetched > 0:
            bits.append(
                f"无新闻 API Key，但免费 RSS/公开源抓到 {fetched} 条头条；"
                "未能分类出可用证据（相关度不足或方向无法判定）。"
                "可换货币对、放宽侧栏最多头条数，或配置 Key 后再跑。"
            )
        else:
            bits.append(
                "已尝试抓取但未能分类出可用证据（相关度不足或方向无法判定）。"
                "可换货币对重跑，或先用练习样例练习标注流程。"
            )
    return " ".join(bits)


def thin_refs_message(
    *,
    evidence_n: int,
    fetched: int = 0,
    news_keys_present: bool = False,
    statements_n: int | None = None,
) -> str | None:
    """Hint when report has only one (or very few) references/evidence items."""
    if evidence_n > 1 and (statements_n is None or statements_n > 2):
        return None
    if evidence_n <= 0:
        return None
    parts = [
        f"本次证据/参考偏少（evidence_n={evidence_n}"
        + (f"，语句 {statements_n}" if statements_n is not None else "")
        + f"，抓取头条 fetched={fetched}）。"
    ]
    if not news_keys_present:
        parts.append(
            "未检测到 `NEWSAPI_KEY`/`FINNHUB_API_KEY`："
            "目前主要靠央行 RSS + Google News；相关度过滤后常只剩 1 条。"
            "要更多 References 请填新闻 Key（可选再加 Tavily）；"
            "DeepSeek/LLM 只判定与改写，不会虚构链接。"
        )
    else:
        parts.append(
            "已有新闻 Key 但仍偏少：可能是相关度过滤过严或当日头条与货币对匹配弱。"
            "可加大「最多头条证据条数」、换货币对，或加 Tavily/Brave 给 AI 检索员。"
        )
    return " ".join(parts)


def spotcheck_filename(pair: str, as_of: date | None = None) -> str:
    pair_safe = (pair or "PAIR").replace("/", "")
    d = (as_of or date.today()).isoformat()
    return f"label_spotcheck_{pair_safe}_{d}.json"


def spotcheck_path(pair: str, as_of: date | None = None, out_dir: Path | None = None) -> Path:
    root = out_dir or project_output_dir()
    return root / spotcheck_filename(pair, as_of)


def save_spotcheck_stats(
    stats: dict[str, Any],
    pair: str,
    *,
    as_of: date | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Persist 抽检准确率 (= agree_rate) next to label_audit CSV."""
    import json

    path = spotcheck_path(pair, as_of=as_of, out_dir=out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pair": pair,
        "as_of": (as_of or date.today()).isoformat(),
        "抽检准确率": stats.get("agree_rate"),
        "agree_rate": stats.get("agree_rate"),
        "n_yes": stats.get("n_yes"),
        "n_no": stats.get("n_no"),
        "n_unsure": stats.get("n_unsure"),
        "n_labeled": stats.get("n_labeled"),
        "n_rows": stats.get("n_rows"),
        "is_demo": stats.get("is_demo"),
        "caption": stats.get("caption"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_spotcheck_stats(
    pair: str,
    *,
    as_of: date | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any] | None:
    import json

    path = spotcheck_path(pair, as_of=as_of, out_dir=out_dir)
    if not path.exists():
        # Fall back to newest spotcheck for this pair
        root = out_dir or project_output_dir()
        pair_safe = (pair or "PAIR").replace("/", "")
        cands = sorted(root.glob(f"label_spotcheck_{pair_safe}_*.json"), reverse=True)
        if not cands:
            return None
        path = cands[0]
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def load_all_label_audits(out_dir: Path | None = None) -> pd.DataFrame:
    """Concatenate every output/label_audit_*.csv (for aggregate 抽检 / learning)."""
    root = out_dir or project_output_dir()
    frames: list[pd.DataFrame] = []
    if root.is_dir():
        for p in sorted(root.glob("label_audit_*.csv")):
            try:
                frames.append(load_label_audit(p))
            except Exception:
                continue
    if not frames:
        return pd.DataFrame(columns=list(LABEL_AUDIT_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def aggregate_spotcheck_stats(out_dir: Path | None = None) -> dict[str, Any]:
    """Agree-rate / 抽检准确率 across all saved label_audit CSVs (non-demo files)."""
    df = load_all_label_audits(out_dir)
    stats = agree_rate_stats(df, is_demo=False)
    stats["抽检准确率"] = stats.get("agree_rate")
    return stats
