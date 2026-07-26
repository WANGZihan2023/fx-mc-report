"""Build evidence from headlines: rules and/or LLM full-text."""

from __future__ import annotations

from typing import Any, Literal

from fx_report.news.classify import headlines_to_evidence as rules_headlines_to_evidence
from fx_report.news.fetch import Headline
from fx_report.news.llm import LLMConfig, classify_headlines_llm, resolve_llm_config
from fx_report.market.pairs import PairSpec
from fx_report.model.weights import EvidenceItem

Mode = Literal["rules", "llm", "hybrid"]


def build_evidence_from_news(
    headlines: list[Headline],
    pair: PairSpec | str,
    *,
    mode: Mode = "hybrid",
    max_items: int = 12,
    unpriced_cap: float = 0.75,
    llm_cfg: LLMConfig | None = None,
    fetch_fulltext: bool = True,
) -> tuple[list[EvidenceItem], dict[str, Any]]:
    """
    mode:
      rules  — keyword only
      llm    — LLM only (fallback to rules if LLM fails/empty)
      hybrid — LLM first; if empty/error, rules
    """
    meta: dict[str, Any] = {"mode": mode, "llm": None, "rules_used": False}

    use_llm = mode in {"llm", "hybrid"}
    cfg = llm_cfg or (resolve_llm_config() if use_llm else None)

    if use_llm and cfg is not None:
        items, llm_meta = classify_headlines_llm(
            headlines,
            pair,
            cfg,
            max_items=max_items,
            fetch_fulltext=fetch_fulltext,
            unpriced_cap=unpriced_cap,
        )
        meta["llm"] = llm_meta
        if items:
            return items, meta
        if mode == "llm" and llm_meta.get("error"):
            # hard fail path still falls back so the report can run
            meta["rules_used"] = True
            return (
                rules_headlines_to_evidence(
                    headlines, pair, max_items=max_items, unpriced_cap=unpriced_cap
                ),
                meta,
            )

    meta["rules_used"] = True
    return (
        rules_headlines_to_evidence(
            headlines, pair, max_items=max_items, unpriced_cap=unpriced_cap
        ),
        meta,
    )
