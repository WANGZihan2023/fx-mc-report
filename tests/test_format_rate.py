"""Tests for shared FX rate display formatting (default 6 decimal places)."""

from __future__ import annotations

from fx_report.format_rate import (
    RATE_DISPLAY_DECIMALS,
    format_rate,
    rate_input_format,
    rate_input_step,
)
from fx_report.model.monte_carlo import bucket_labels_from_edges


def test_default_decimals_is_six():
    assert RATE_DISPLAY_DECIMALS == 6


def test_format_rate_six_dp():
    assert format_rate(0.6501234) == "0.650123"
    assert format_rate(1.5) == "1.500000"
    assert format_rate(150.12) == "150.120000"


def test_format_rate_signed_and_na():
    assert format_rate(0.001234, signed=True) == "+0.001234"
    assert format_rate(-0.001234, signed=True) == "-0.001234"
    assert format_rate(None) == "—"
    assert format_rate(float("nan"), na="N/A") == "N/A"


def test_rate_input_helpers():
    assert rate_input_format() == "%.6f"
    assert rate_input_step(0.65) == 1e-6
    assert rate_input_step(150.0) == 0.01


def test_bucket_labels_use_six_dp():
    labels = bucket_labels_from_edges((1.4, 1.43, 1.46, 1.49))
    assert labels[0] == "< 1.400000"
    assert labels[1] == "1.400000 to 1.430000"
    assert labels[-1] == ">= 1.490000"
