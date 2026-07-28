"""
Parse boss/order PDFs (单子) into start-setup fields.

Heuristic regex first (works offline). Optional LLM assist when DeepSeek/LLM
is configured — never silently invent required start choices.
"""

from __future__ import annotations

import io
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping, Sequence

from fx_report.market.pairs import PAIR_CATALOG, list_pairs
from fx_report.ui.ux_helpers import START_REQUIRED_LABELS, missing_start_choices

# CCY codes we care about for bullish / pair matching
_KNOWN_CCY = frozenset(
    {
        "USD",
        "AUD",
        "EUR",
        "GBP",
        "JPY",
        "CNH",
        "CNY",
        "CAD",
        "NZD",
        "CHF",
        "HKD",
        "SGD",
        "KRW",
        "INR",
        "MXN",
        "NOK",
        "SEK",
        "ZAR",
    }
)

_CCY_ALIASES: dict[str, str] = {
    "澳元": "AUD",
    "美元": "USD",
    "欧元": "EUR",
    "英镑": "GBP",
    "日元": "JPY",
    "离岸人民币": "CNH",
    "在岸人民币": "CNY",
    "人民币": "CNY",
    "加元": "CAD",
    "纽元": "NZD",
    "瑞郎": "CHF",
    "美刀": "USD",
    "绿纸": "USD",
}

_PAIR_SLASH_RE = re.compile(
    r"\b([A-Z]{3})\s*[/\-−–]\s*([A-Z]{3})\b",
    re.IGNORECASE,
)
_PAIR_CONCAT_RE = re.compile(
    r"\b("
    + "|".join(re.escape(p.replace("/", "")) for p in PAIR_CATALOG)
    + r")\b",
    re.IGNORECASE,
)
_CN_PAIR_RE = re.compile(
    r"(澳元|美元|欧元|英镑|日元|离岸人民币|在岸人民币|人民币|加元|纽元|瑞郎)"
    r"\s*(?:兑|/|对)\s*"
    r"(澳元|美元|欧元|英镑|日元|离岸人民币|在岸人民币|人民币|加元|纽元|瑞郎)"
)

_BULLISH_RE = re.compile(
    r"(?:看涨|看多|偏多|做多|bullish|long)\s*"
    r"[:：]?\s*"
    r"([A-Z]{3}|澳元|美元|欧元|英镑|日元|离岸人民币|在岸人民币|人民币|加元|纽元|瑞郎)",
    re.IGNORECASE,
)
_BEARISH_RE = re.compile(
    r"(?:看跌|看空|偏空|做空|bearish|short)\s*"
    r"[:：]?\s*"
    r"([A-Z]{3}|澳元|美元|欧元|英镑|日元|离岸人民币|在岸人民币|人民币|加元|纽元|瑞郎)",
    re.IGNORECASE,
)

_LEVEL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("barrier", re.compile(r"(?:barrier|障碍(?:价|位)?|障碍水平)\s*[:：=]?\s*([0-9]+\.?[0-9]*)", re.I)),
    ("strike", re.compile(r"(?:strike|行权(?:价|位)?|执行价)\s*[:：=]?\s*([0-9]+\.?[0-9]*)", re.I)),
    ("spot", re.compile(r"(?:spot|现价|即期)\s*[:：=]?\s*([0-9]+\.?[0-9]*)", re.I)),
    ("target", re.compile(r"(?:target|目标(?:价|位)?)\s*[:：=]?\s*([0-9]+\.?[0-9]*)", re.I)),
]

_TENOR_RE = re.compile(
    r"(?:期限|tenor|horizon|到期)\s*[:：]?\s*"
    r"(\d+\s*个?\s*(?:天|日|周|月|年|d|w|m|y|day|days|week|weeks|month|months|year|years))",
    re.IGNORECASE,
)

_BUCKET_CUTS_RE = re.compile(
    r"(?:分档|切点|边界|bucket\s*cuts?|pct\s*cuts?)\s*[:：]?\s*"
    r"([+\-]?\d+(?:\.\d+)?(?:\s*[%％]?)(?:\s*[,/，、\s]+\s*[+\-]?\d+(?:\.\d+)?(?:\s*[%％]?)?){3})",
    re.IGNORECASE,
)


@dataclass
class OrderPdfParse:
    """Result of parsing a boss/order PDF (or plain text fixture)."""

    ok: bool
    error: str | None = None
    text_preview: str = ""
    pair: str | None = None
    pair_mode: str | None = None  # 目录 | 自定义
    bullish_currency: str | None = None
    strike: float | None = None
    barrier: float | None = None
    spot: float | None = None
    levels: list[float] = field(default_factory=list)
    bucket_mode: str | None = None  # 相对现价 | 绝对价位
    bucket_edges: list[float] | None = None  # 4 absolute cuts when confident
    bucket_pct_cuts: list[float] | None = None  # 4 relative % when confident
    tenor: str | None = None
    filled: list[str] = field(default_factory=list)
    still_needed: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source: str = "heuristic"  # heuristic | heuristic+llm

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_pdf_text(data: bytes, *, max_pages: int = 8) -> str:
    """
    Extract text from PDF bytes.

    Prefers pypdf; falls back to PyMuPDF (fitz) if installed.
    Raises ValueError with a Chinese message on failure / empty text.
    """
    if not data:
        raise ValueError("PDF 文件为空，请重新上传。")

    text = ""
    errors: list[str] = []

    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(io.BytesIO(data))
        chunks: list[str] = []
        for i, page in enumerate(reader.pages):
            if i >= max_pages:
                break
            try:
                chunks.append(page.extract_text() or "")
            except Exception as exc:  # pragma: no cover
                errors.append(f"pypdf page {i}: {exc}")
        text = "\n".join(chunks)
    except ImportError:
        errors.append("未安装 pypdf")
    except Exception as exc:
        errors.append(f"pypdf: {exc}")

    if not (text or "").strip():
        try:
            import fitz  # type: ignore

            doc = fitz.open(stream=data, filetype="pdf")
            chunks = [page.get_text() for i, page in enumerate(doc) if i < max_pages]
            doc.close()
            text = "\n".join(chunks)
        except ImportError:
            errors.append("未安装 PyMuPDF")
        except Exception as exc:
            errors.append(f"fitz: {exc}")

    cleaned = (text or "").strip()
    if not cleaned:
        detail = "；".join(errors) if errors else "无法识别文字"
        raise ValueError(
            f"无法从 PDF 提取文字（{detail}）。请改用可选中文字的 PDF，或继续手动填写开始设置。"
        )
    return cleaned


def _norm_ccy(token: str) -> str | None:
    t = (token or "").strip()
    if not t:
        return None
    if t in _CCY_ALIASES:
        return _CCY_ALIASES[t]
    u = t.upper()
    if u in _KNOWN_CCY:
        return u
    return None


def _norm_pair(base: str, quote: str) -> str | None:
    b = _norm_ccy(base)
    q = _norm_ccy(quote)
    if not b or not q or b == q:
        return None
    return f"{b}/{q}"


def _catalog_or_custom(pair: str) -> tuple[str, str]:
    """Return (pair, pair_mode). Prefer catalog spelling."""
    key = pair.replace(" ", "").upper()
    if key in PAIR_CATALOG:
        return key, "目录"
    if "/" not in key and len(key) == 6:
        key = f"{key[:3]}/{key[3:]}"
        if key in PAIR_CATALOG:
            return key, "目录"
    if "/" in key:
        b, q = key.split("/", 1)
        return f"{b}/{q}", ("目录" if f"{b}/{q}" in PAIR_CATALOG else "自定义")
    return pair, "自定义"


def _first_float(m: re.Match[str]) -> float | None:
    try:
        v = float(m.group(1))
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    return v


def _parse_pct_cuts(blob: str) -> list[float] | None:
    nums = re.findall(r"[+\-]?\d+(?:\.\d+)?", blob)
    if len(nums) < 4:
        return None
    vals = [float(x) for x in nums[:4]]
    # strip accidental % already handled by regex capture of number only
    if vals != sorted(vals):
        return None
    return vals


def parse_order_text(
    text: str,
    *,
    catalog: Sequence[str] | None = None,
) -> OrderPdfParse:
    """
    Heuristic parse of order/单子 plain text.

    Only fills fields when confident; never invents peak_engine / calib / HITL.
    """
    raw = (text or "").strip()
    if not raw:
        return OrderPdfParse(
            ok=False,
            error="单子内容为空，请检查 PDF 或改用手动填写。",
        )

    catalog_list = list(catalog) if catalog is not None else list_pairs()
    notes: list[str] = []
    pair: str | None = None
    pair_mode: str | None = None

    # 1) Slash / dash pairs
    for m in _PAIR_SLASH_RE.finditer(raw):
        cand = _norm_pair(m.group(1), m.group(2))
        if cand:
            pair, pair_mode = _catalog_or_custom(cand)
            break

    # 2) Concatenated catalog codes (AUDUSD)
    if pair is None:
        m = _PAIR_CONCAT_RE.search(raw)
        if m:
            code = m.group(1).upper()
            pair, pair_mode = _catalog_or_custom(f"{code[:3]}/{code[3:]}")

    # 3) Chinese 兑
    if pair is None:
        m = _CN_PAIR_RE.search(raw)
        if m:
            cand = _norm_pair(m.group(1), m.group(2))
            if cand:
                pair, pair_mode = _catalog_or_custom(cand)

    # Prefer catalog pair if inverted also exists and text mentions catalog form more
    if pair and pair not in catalog_list:
        inv = f"{pair.split('/')[1]}/{pair.split('/')[0]}" if "/" in pair else None
        if inv and inv in catalog_list:
            # keep custom only if neither in catalog; else prefer catalog invert
            if pair not in catalog_list:
                pair, pair_mode = inv, "目录"
                notes.append(f"已映射到目录货币对 {inv}")

    levels_named: dict[str, float] = {}
    for name, pat in _LEVEL_PATTERNS:
        m = pat.search(raw)
        if m:
            v = _first_float(m)
            if v is not None:
                levels_named[name] = v

    barrier = levels_named.get("barrier")
    strike = levels_named.get("strike")
    spot = levels_named.get("spot")
    levels = sorted({float(v) for v in levels_named.values()})

    bullish: str | None = None
    m_up = _BULLISH_RE.search(raw)
    if m_up:
        bullish = _norm_ccy(m_up.group(1))
    if bullish is None:
        m_dn = _BEARISH_RE.search(raw)
        if m_dn and pair and "/" in pair:
            # 看跌 X → 看涨 the other leg
            weak = _norm_ccy(m_dn.group(1))
            b, q = pair.split("/", 1)
            if weak == b:
                bullish = q
                notes.append(f"由「看跌 {weak}」推断看涨 {q}")
            elif weak == q:
                bullish = b
                notes.append(f"由「看跌 {weak}」推断看涨 {b}")

    # If pair known and bullish matches one leg, keep; else drop
    if bullish and pair and "/" in pair:
        b, q = pair.split("/", 1)
        if bullish not in (b, q):
            notes.append(f"忽略与货币对不符的看涨币 {bullish}")
            bullish = None

    tenor = None
    m_t = _TENOR_RE.search(raw)
    if m_t:
        tenor = re.sub(r"\s+", "", m_t.group(1))

    bucket_mode: str | None = None
    bucket_edges: list[float] | None = None
    bucket_pct: list[float] | None = None

    m_cuts = _BUCKET_CUTS_RE.search(raw)
    if m_cuts:
        pct = _parse_pct_cuts(m_cuts.group(1))
        if pct is not None:
            # Heuristic: values that look like percent moves (abs < 80) → relative
            if all(abs(x) <= 80 for x in pct):
                bucket_pct = pct
                bucket_mode = "相对现价"
            else:
                bucket_edges = pct
                bucket_mode = "绝对价位"

    # Absolute Barrier/Strike → absolute bucket mode; edges only if we can form 4
    if bucket_mode is None and (barrier is not None or strike is not None):
        bucket_mode = "绝对价位"
        edges_src = [v for v in (spot, barrier, strike, levels_named.get("target")) if v is not None]
        uniq = sorted({round(float(x), 6) for x in edges_src})
        if len(uniq) >= 4:
            bucket_edges = uniq[:4]
        elif len(uniq) == 3 and spot is not None:
            # insert a mid between barrier and strike if possible
            mid_candidates = sorted(uniq)
            lo, mid, hi = mid_candidates
            extra = round((mid + hi) / 2.0, 6)
            four = sorted({lo, mid, extra, hi})
            if len(four) == 4:
                bucket_edges = four
                notes.append("由现价/Barrier/Strike 插值得到 4 个绝对切点（请核对）")
        elif len(uniq) >= 2:
            notes.append(
                "已识别 Barrier/Strike 等绝对价位，但不足 4 个切点；"
                "已选「绝对价位」，请在主区补全边界。"
            )

    filled: list[str] = []
    if pair:
        filled.append("货币对")
    if bullish:
        filled.append("看涨货币")
    if bucket_mode:
        filled.append("分档边界方式")
    if bucket_edges or bucket_pct:
        filled.append("分档切点")
    if tenor:
        filled.append("期限")

    # Still needed among start-required (PDF never fills engine/calib/HITL)
    draft = {
        "pair": pair,
        "bullish_currency": bullish,
        "peak_engine": None,
        "use_calibrated": None,
        "human_review": None,
        "bucket_mode": bucket_mode,
    }
    still = missing_start_choices(draft)
    # Also note cuts incomplete when absolute mode without 4 edges
    if bucket_mode == "绝对价位" and not bucket_edges:
        if "分档切点" not in still:
            still.append("分档切点（需手动补全）")
    if bucket_mode == "相对现价" and not bucket_pct:
        if "分档切点" not in still:
            still.append("分档切点（需手动补全）")

    notes.append(
        "峰值引擎 / 是否使用校准参数 / 不确定证据人工确认：单子通常不含，请手动选择。"
    )

    ok = bool(pair or bullish or barrier or strike or bucket_mode)
    if not ok:
        return OrderPdfParse(
            ok=False,
            error="未能从单子识别货币对或价位，请手动填写开始设置。",
            text_preview=raw[:400],
            notes=notes,
        )

    return OrderPdfParse(
        ok=True,
        text_preview=raw[:600],
        pair=pair,
        pair_mode=pair_mode,
        bullish_currency=bullish,
        strike=strike,
        barrier=barrier,
        spot=spot,
        levels=levels,
        bucket_mode=bucket_mode,
        bucket_edges=bucket_edges,
        bucket_pct_cuts=bucket_pct,
        tenor=tenor,
        filled=filled,
        still_needed=still,
        notes=notes,
        source="heuristic",
    )


def _merge_llm_fields(base: OrderPdfParse, llm: Mapping[str, Any]) -> OrderPdfParse:
    """Fill only unset fields from LLM JSON; never overwrite confident heuristic."""
    if not base.ok:
        return base
    notes = list(base.notes)
    pair = base.pair
    pair_mode = base.pair_mode
    bullish = base.bullish_currency
    barrier = base.barrier
    strike = base.strike
    spot = base.spot
    bucket_mode = base.bucket_mode
    bucket_edges = list(base.bucket_edges) if base.bucket_edges else None
    bucket_pct = list(base.bucket_pct_cuts) if base.bucket_pct_cuts else None
    tenor = base.tenor

    llm_pair = str(llm.get("pair") or "").strip().upper().replace(" ", "")
    if not pair and llm_pair:
        if "/" not in llm_pair and len(llm_pair) == 6:
            llm_pair = f"{llm_pair[:3]}/{llm_pair[3:]}"
        if re.match(r"^[A-Z]{3}/[A-Z]{3}$", llm_pair):
            pair, pair_mode = _catalog_or_custom(llm_pair)
            notes.append("货币对由 LLM 补全（请核对）")

    llm_bull = _norm_ccy(str(llm.get("bullish_currency") or ""))
    if not bullish and llm_bull and pair and "/" in pair:
        b, q = pair.split("/", 1)
        if llm_bull in (b, q):
            bullish = llm_bull
            notes.append("看涨货币由 LLM 补全（请核对）")

    def _f(key: str) -> float | None:
        v = llm.get(key)
        try:
            if v is None or v == "":
                return None
            x = float(v)
            return x if x > 0 else None
        except (TypeError, ValueError):
            return None

    if barrier is None:
        barrier = _f("barrier")
    if strike is None:
        strike = _f("strike")
    if spot is None:
        spot = _f("spot")
    if not tenor and llm.get("tenor"):
        tenor = str(llm.get("tenor")).strip() or None

    if bucket_mode is None and llm.get("bucket_mode") in ("相对现价", "绝对价位"):
        bucket_mode = str(llm["bucket_mode"])
        notes.append("分档方式由 LLM 建议（请核对）")

    # Rebuild still_needed / filled
    levels = sorted({x for x in (barrier, strike, spot) if x is not None})
    filled: list[str] = []
    if pair:
        filled.append("货币对")
    if bullish:
        filled.append("看涨货币")
    if bucket_mode:
        filled.append("分档边界方式")
    if bucket_edges or bucket_pct:
        filled.append("分档切点")
    if tenor:
        filled.append("期限")
    draft = {
        "pair": pair,
        "bullish_currency": bullish,
        "peak_engine": None,
        "use_calibrated": None,
        "human_review": None,
        "bucket_mode": bucket_mode,
    }
    still = missing_start_choices(draft)
    return OrderPdfParse(
        ok=True,
        text_preview=base.text_preview,
        pair=pair,
        pair_mode=pair_mode,
        bullish_currency=bullish,
        strike=strike,
        barrier=barrier,
        spot=spot,
        levels=levels,
        bucket_mode=bucket_mode,
        bucket_edges=bucket_edges,
        bucket_pct_cuts=bucket_pct,
        tenor=tenor,
        filled=filled,
        still_needed=still,
        notes=notes,
        source="heuristic+llm",
    )


def llm_assist_order_text(text: str, *, llm_cfg: Any | None = None) -> dict[str, Any] | None:
    """Optional LLM JSON extraction. Returns None if unavailable / fails."""
    if llm_cfg is None:
        try:
            from fx_report.news.llm import resolve_llm_config

            llm_cfg = resolve_llm_config()
        except Exception:
            return None
    if llm_cfg is None:
        return None
    try:
        from fx_report.news.llm import _chat_json
    except Exception:
        return None

    system = (
        "你是外汇单子解析助手。只根据用户文本抽取字段，不要编造。"
        "返回 JSON："
        '{"pair":"USD/AUD或null","bullish_currency":"USD或null",'
        '"barrier":数字或null,"strike":数字或null,"spot":数字或null,'
        '"tenor":"字符串或null","bucket_mode":"相对现价|绝对价位|null"}'
    )
    user = f"单子文本：\n{text[:3500]}"
    try:
        return _chat_json(llm_cfg, system, user)
    except Exception:
        return None


def parse_order_pdf(
    data: bytes,
    *,
    use_llm: bool = True,
    llm_cfg: Any | None = None,
) -> OrderPdfParse:
    """Extract PDF text then parse. On extract failure returns ok=False Chinese error."""
    try:
        text = extract_pdf_text(data)
    except ValueError as exc:
        return OrderPdfParse(ok=False, error=str(exc))
    except Exception as exc:  # pragma: no cover
        return OrderPdfParse(
            ok=False,
            error=f"读取 PDF 失败：{exc}。请改用手动填写开始设置。",
        )

    result = parse_order_text(text)
    if not result.ok:
        return result

    if use_llm:
        llm_json = llm_assist_order_text(text, llm_cfg=llm_cfg)
        if llm_json:
            result = _merge_llm_fields(result, llm_json)
    return result


def order_pdf_from_dict(data: Mapping[str, Any] | None) -> OrderPdfParse | None:
    """Rebuild OrderPdfParse from session_state dict."""
    if not data:
        return None
    allowed = {f.name for f in fields(OrderPdfParse)}
    kwargs = {k: v for k, v in data.items() if k in allowed}
    for list_key in ("levels", "filled", "still_needed", "notes"):
        if kwargs.get(list_key) is None and list_key in kwargs:
            kwargs[list_key] = []
    try:
        return OrderPdfParse(**kwargs)
    except TypeError:
        return None


def preview_lines(result: OrderPdfParse) -> list[str]:
    """Chinese bullet lines for UI: filled vs still needed."""
    lines: list[str] = []
    if not result.ok:
        lines.append(result.error or "解析失败")
        return lines
    if result.filled:
        lines.append("已自动填入：" + "、".join(result.filled))
    else:
        lines.append("已自动填入：（无）")
    if result.still_needed:
        lines.append("仍需你选择：" + "、".join(result.still_needed))
    detail_bits: list[str] = []
    if result.pair:
        detail_bits.append(f"货币对={result.pair}")
    if result.bullish_currency:
        detail_bits.append(f"看涨={result.bullish_currency}")
    if result.barrier is not None:
        detail_bits.append(f"Barrier={result.barrier:g}")
    if result.strike is not None:
        detail_bits.append(f"Strike={result.strike:g}")
    if result.bucket_mode:
        detail_bits.append(f"分档={result.bucket_mode}")
    if result.tenor:
        detail_bits.append(f"期限={result.tenor}")
    if detail_bits:
        lines.append("识别明细：" + " · ".join(detail_bits))
    for n in result.notes:
        lines.append(n)
    return lines


# Labels that PDF may auto-fill (for docs / tests)
ORDER_PDF_FILLABLE = (
    "货币对",
    "看涨货币",
    "分档边界方式",
    "分档切点",
)
ORDER_PDF_ALWAYS_MANUAL = tuple(
    START_REQUIRED_LABELS[k]
    for k in ("peak_engine", "use_calibrated", "human_review")
)
