"""Shared FX rate display formatting (spot, edges, peaks, targets).

Default precision is 6 decimal places (e.g. 0.650123). Percentages /
probabilities are out of scope — keep their own conventions.
"""

from __future__ import annotations

from typing import Any

RATE_DISPLAY_DECIMALS = 6


def format_rate(
    x: Any,
    *,
    decimals: int = RATE_DISPLAY_DECIMALS,
    signed: bool = False,
    na: str = "—",
) -> str:
    """Format an FX price / rate for UI and report display."""
    if x is None:
        return na
    try:
        v = float(x)
    except (TypeError, ValueError):
        return na
    if v != v:  # NaN
        return na
    if signed:
        return f"{v:+.{decimals}f}"
    return f"{v:.{decimals}f}"


def rate_input_format(decimals: int = RATE_DISPLAY_DECIMALS) -> str:
    """Streamlit ``number_input`` / slider ``format`` string for rate fields."""
    return f"%.{decimals}f"


def rate_input_step(spot: float | None, *, decimals: int = RATE_DISPLAY_DECIMALS) -> float:
    """Reasonable spinner step: fine for FX quotes, coarser for JPY-scale."""
    if spot is not None and float(spot) >= 10:
        return 0.01
    return 10.0 ** (-decimals)
