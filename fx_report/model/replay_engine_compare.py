"""Auto-scan historical as_of dates with news evidence, then compare peak engines A vs C."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fx_report.market.pairs import get_pair
from fx_report.model.replay_backtest import run_replay_backtest
from fx_report.news.classify import headlines_to_evidence
from fx_report.news.fetch import fetch_historical_headlines_for_pair

# Combo labels from docs/auto_tune_USDAUD.md
ENGINE_COMBOS: dict[str, dict[str, str]] = {
    "A": {
        "peak_engine": "path_max",
        "jump_model": "merton",
        "variance_reduction": "antithetic",
    },
    "C": {
        "peak_engine": "brownian_bridge",
        "jump_model": "none",
        "variance_reduction": "antithetic",
    },
}


def _as_date(value: date | datetime | str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value).strip()
    if "T" in text:
        text = text.split("T", 1)[0]
    if " " in text:
        text = text.split(" ", 1)[0]
    return date.fromisoformat(text)


def _parse_dates_list(raw: str | list[str] | None) -> list[date] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        parts = raw
    else:
        parts = [p.strip() for p in str(raw).replace(";", ",").split(",") if p.strip()]
    if not parts:
        return None
    return [_as_date(p) for p in parts]


def _rules_evidence_n(
    headlines: list[Any],
    pair: str,
    *,
    as_of: date,
    max_items: int = 12,
) -> int:
    reference_now = datetime(
        as_of.year, as_of.month, as_of.day, 23, 59, 59, tzinfo=timezone.utc
    )
    items, _counts = headlines_to_evidence(
        headlines,
        pair,
        max_items=max_items,
        unpriced_cap=0.75,
        reference_now=reference_now,
    )
    return len(items)


@dataclass
class CandidateDate:
    as_of: date
    evidence_n: int
    quality: str
    newsapi_hits: int
    from_cache: bool
    http_status: int | None = None


@dataclass
class EngineCompareRow:
    as_of: str
    evidence_n: int
    quality: str
    skill_a: float
    skill_c: float
    winner: str
    path_a: str
    path_c: str


@dataclass
class EngineCompareResult:
    pair: str
    candidates: list[CandidateDate]
    rows: list[EngineCompareRow]
    summary_path: Path
    stopped_on_429: bool
    scan_notes: list[str]


def scan_candidate_as_of(
    pair: str,
    *,
    start_date: date | datetime | str,
    end_date: date | datetime | str,
    step_days: int = 3,
    lookback: int = 14,
    max_dates: int = 3,
    max_news: int = 25,
    verbose: bool = True,
) -> tuple[list[CandidateDate], bool, list[str]]:
    """
    Walk [start, end] by step and keep dates where historical news looks usable:
    evidence_n > 0 OR historical_news_quality == date_filtered.

    Stops early on NewsAPI HTTP 429 (quota). Disk/memory cache is used by fetch_newsapi.
    """
    start = _as_date(start_date)
    end = _as_date(end_date)
    if end < start:
        raise ValueError("end_date must be >= start_date")
    if step_days <= 0:
        raise ValueError("step_days must be positive")
    if max_dates <= 0:
        raise ValueError("max_dates must be positive")

    spec = get_pair(pair)
    candidates: list[CandidateDate] = []
    notes: list[str] = []
    stopped_429 = False
    cur = start
    while cur <= end and len(candidates) < max_dates:
        headlines, meta = fetch_historical_headlines_for_pair(
            spec,
            as_of_date=cur,
            lookback_days=lookback,
            max_items=max_news,
        )
        http_status = meta.get("newsapi_http_status")
        err = str(meta.get("newsapi_error") or "")
        if http_status == 429 or "429" in err or "rateLimited" in err:
            stopped_429 = True
            notes.append(f"NewsAPI 429 at as_of={cur.isoformat()}; stop scan early.")
            if verbose:
                print(notes[-1])
            break

        quality = str(meta.get("historical_news_quality") or "limited")
        evidence_n = _rules_evidence_n(headlines, spec.pair, as_of=cur)
        newsapi_hits = int(meta.get("newsapi_hits") or 0)
        from_cache = bool(meta.get("newsapi_from_cache"))
        ok = evidence_n > 0 or quality == "date_filtered"
        if verbose:
            print(
                f"scan {cur.isoformat()}: quality={quality} evidence_n={evidence_n} "
                f"newsapi_hits={newsapi_hits} cache={from_cache}"
                + (" → candidate" if ok else "")
            )
        if ok:
            candidates.append(
                CandidateDate(
                    as_of=cur,
                    evidence_n=evidence_n,
                    quality=quality,
                    newsapi_hits=newsapi_hits,
                    from_cache=from_cache,
                    http_status=http_status if isinstance(http_status, int) else None,
                )
            )
        cur += timedelta(days=step_days)

    if not candidates and not stopped_429:
        notes.append(
            "未找到 evidence_n>0 或 date_filtered 的 as_of；可缩小步长或改日期窗。"
        )
    return candidates, stopped_429, notes


def _skill_from_replay(result: Any) -> float:
    table = result.table
    if table is None or table.empty or "skill_brier" not in table.columns:
        return float("nan")
    return float(table["skill_brier"].iloc[0])


def _winner(skill_a: float, skill_c: float) -> str:
    a_ok = math.isfinite(skill_a)
    c_ok = math.isfinite(skill_c)
    if a_ok and c_ok:
        if skill_a > skill_c:
            return "A"
        if skill_c > skill_a:
            return "C"
        return "tie"
    if a_ok and not c_ok:
        return "A"
    if c_ok and not a_ok:
        return "C"
    return "n/a"


def _write_engine_json(
    path: Path,
    *,
    label: str,
    as_of: date,
    combo: dict[str, str],
    candidate: CandidateDate,
    replay: Any,
) -> None:
    rows = replay.table.to_dict(orient="records") if replay.table is not None else []
    payload = {
        "engine": label,
        "combo": combo,
        "as_of": as_of.isoformat(),
        "scan_evidence_n": candidate.evidence_n,
        "scan_quality": candidate.quality,
        "summary": replay.summary,
        "rows": rows,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # Drop auto-named replay files to keep engine_compare/ tidy.
    for p in (getattr(replay, "json_path", None), getattr(replay, "csv_path", None)):
        if p is None:
            continue
        try:
            Path(p).unlink(missing_ok=True)
        except OSError:
            pass


def run_engine_compare(
    pair: str = "USD/AUD",
    *,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    step_days: int = 3,
    dates: str | list[str] | None = None,
    max_dates: int = 3,
    sims: int = 800,
    days: int = 20,
    seed: int = 42,
    lookback: int = 14,
    max_news: int = 10,
    mode: str = "rules",
    out_dir: str | Path = "output/engine_compare",
    bullish_currency: str | None = None,
    calibrated_params_path: str | Path | None = None,
    verbose: bool = True,
) -> EngineCompareResult:
    """
    1) Scan (unless --dates) for as_of with historical news evidence
    2) Replay A vs C on each candidate
    3) Write A_{asof}.json / C_{asof}.json + summary table
    """
    today = date.today()
    if start_date is None:
        start_date = today - timedelta(days=25)
    if end_date is None:
        # Leave room for realized horizon after as_of.
        end_date = today - timedelta(days=max(days + 1, 7))

    explicit = _parse_dates_list(dates)
    stopped_429 = False
    notes: list[str] = []

    if explicit is not None:
        candidates = [
            CandidateDate(
                as_of=d,
                evidence_n=-1,
                quality="user_dates",
                newsapi_hits=-1,
                from_cache=False,
            )
            for d in explicit[: max(1, int(max_dates))]
        ]
        notes.append(f"跳过新闻扫描，使用 --dates（n={len(candidates)}）。")
        if verbose:
            print(notes[-1])
    else:
        candidates, stopped_429, notes = scan_candidate_as_of(
            pair,
            start_date=start_date,
            end_date=end_date,
            step_days=step_days,
            lookback=lookback,
            max_dates=max_dates,
            max_news=max(max_news, 25),
            verbose=verbose,
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[EngineCompareRow] = []

    for cand in candidates:
        as_of = cand.as_of
        if verbose:
            print(f"\n=== replay compare as_of={as_of.isoformat()} ===")
        paths: dict[str, Path] = {}
        skills: dict[str, float] = {}
        for label, combo in ENGINE_COMBOS.items():
            if verbose:
                print(
                    f"  Engine {label}: {combo['peak_engine']}+{combo['jump_model']}"
                    f"+{combo['variance_reduction']}"
                )
            replay = run_replay_backtest(
                pair,
                bullish_currency=bullish_currency,
                start_date=as_of,
                end_date=as_of,
                step_days=1,
                out_dir=out,
                sims=sims,
                days=days,
                seed=seed,
                lookback=lookback,
                peak_engine=combo["peak_engine"],
                variance_reduction=combo["variance_reduction"],
                jump_model=combo["jump_model"],
                jump_compensate=False,
                mode=mode,
                max_news=max_news,
                keep_templates=False,
                template_policy="off",
                no_news=False,
                no_fulltext=True,
                ai_research=False,
                calibrated_params_path=calibrated_params_path,
                use_label_learned_strength=False,
                max_dates=1,
                verbose=verbose,
            )
            # Prefer replay-row evidence when user skipped scan.
            if cand.evidence_n < 0 and not replay.table.empty:
                cand.evidence_n = int(replay.table["evidence_n"].iloc[0])
                cand.quality = str(replay.table["historical_news_quality"].iloc[0])
            path = out / f"{label}_{as_of.isoformat()}.json"
            _write_engine_json(
                path,
                label=label,
                as_of=as_of,
                combo=combo,
                candidate=cand,
                replay=replay,
            )
            paths[label] = path
            skills[label] = _skill_from_replay(replay)

        skill_a = skills.get("A", float("nan"))
        skill_c = skills.get("C", float("nan"))
        row = EngineCompareRow(
            as_of=as_of.isoformat(),
            evidence_n=int(cand.evidence_n),
            quality=cand.quality,
            skill_a=skill_a,
            skill_c=skill_c,
            winner=_winner(skill_a, skill_c),
            path_a=str(paths["A"]),
            path_c=str(paths["C"]),
        )
        rows.append(row)

    summary_path = out / "summary.json"
    summary_payload = {
        "pair": pair,
        "combos": ENGINE_COMBOS,
        "defaults": {
            "sims": sims,
            "days": days,
            "lookback": lookback,
            "max_dates": max_dates,
            "step_days": step_days,
            "mode": mode,
        },
        "stopped_on_429": stopped_429,
        "notes": notes,
        "rows": [
            {
                "as_of": r.as_of,
                "evidence_n": r.evidence_n,
                "quality": r.quality,
                "skill_A": r.skill_a,
                "skill_C": r.skill_c,
                "winner": r.winner,
                "path_A": r.path_a,
                "path_C": r.path_c,
            }
            for r in rows
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return EngineCompareResult(
        pair=pair,
        candidates=candidates,
        rows=rows,
        summary_path=summary_path,
        stopped_on_429=stopped_429,
        scan_notes=notes,
    )


def print_chinese_summary(result: EngineCompareResult) -> None:
    print(
        "\n===== 引擎对比汇总（A=path_max+merton+antithetic，"
        "C=brownian_bridge+none+antithetic）====="
    )
    if result.stopped_on_429:
        print("注意：扫描因 NewsAPI 429 提前停止（已用缓存尽量省配额）。")
    for note in result.scan_notes:
        print(f"备注：{note}")
    if not result.rows:
        print("无对比行。")
        return
    header = (
        f"{'as_of':<12} {'evidence_n':>10} {'quality':<16} "
        f"{'skill_A':>10} {'skill_C':>10} {'winner':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in result.rows:
        sa = f"{r.skill_a:.4f}" if r.skill_a == r.skill_a else "nan"
        sc = f"{r.skill_c:.4f}" if r.skill_c == r.skill_c else "nan"
        print(
            f"{r.as_of:<12} {r.evidence_n:>10} {r.quality:<16} "
            f"{sa:>10} {sc:>10} {r.winner:>6}"
        )
    print(f"\n明细 JSON → {result.summary_path}")
