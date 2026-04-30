"""Tests for ``rvi.config``."""

from __future__ import annotations

import pytest

from rvi.config import (
    DEFAULT_NAIROBI_BBOX,
    DEFAULT_SEVERITY_INT,
    Config,
    get_config,
    reset_config,
    set_config,
)


def test_defaults_match_proposal() -> None:
    cfg = Config()
    # §2.3.4 weights
    assert cfg.rvi_alpha == pytest.approx(0.4)
    assert cfg.rvi_beta == pytest.approx(0.3)
    assert cfg.rvi_gamma == pytest.approx(0.3)
    # §5.3 buffer triple
    assert cfg.buffer_widths_m == (6, 10, 30)
    # §5.5 upstream radius
    assert cfg.upstream_radius_m == 50_000.0
    # §5.3 segment length
    assert cfg.segment_length_m == 500.0
    # §2.4.1 severity oracle
    assert cfg.severity_int["EXTREME"] == 4
    assert cfg.severity_int["NO_FLOODING"] == 1
    assert cfg.severity_int["UNKNOWN"] == 0


def test_weight_sum_validation_rejects_bad_triple() -> None:
    with pytest.raises(ValueError, match="weights"):
        Config(rvi_alpha=0.5, rvi_beta=0.5, rvi_gamma=0.5)


def test_buffer_widths_must_be_positive() -> None:
    with pytest.raises(ValueError, match="buffer width"):
        Config(buffer_widths_m=(0, 10, 30))


def test_strahler_lookup_clamps_unknown_orders() -> None:
    cfg = Config()
    assert cfg.half_width_for_strahler(2) == 3.0
    # Order 99 is out of range — clamps to the largest known order (4).
    assert cfg.half_width_for_strahler(99) == cfg.half_width_for_strahler(4)
    # Order -5 clamps to 1.
    assert cfg.half_width_for_strahler(-5) == cfg.half_width_for_strahler(1)


def test_strahler_for_waterway_defaults_to_one() -> None:
    cfg = Config()
    assert cfg.strahler_for_waterway("river") == 4
    assert cfg.strahler_for_waterway("ditch") == 1
    assert cfg.strahler_for_waterway("not-a-real-tag") == 1


def test_severity_to_int_handles_aliases_and_unknown() -> None:
    cfg = Config()
    assert cfg.severity_to_int("SEVERE") == 3
    assert cfg.severity_to_int("severe") == 3
    assert cfg.severity_to_int(None) == 0
    assert cfg.severity_to_int("WHATEVER") == 0
    assert cfg.severity_to_int("SEVERITY_UNSPECIFIED") == 0


def test_ensure_dirs_is_idempotent(tmp_path) -> None:
    cfg = Config(
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        interim_dir=tmp_path / "data" / "interim",
        processed_dir=tmp_path / "data" / "processed",
        cache_dir=tmp_path / "data" / "cache",
        outputs_dir=tmp_path / "outputs",
    )
    cfg.ensure_dirs()
    cfg.ensure_dirs()  # second call must not raise
    assert (tmp_path / "data" / "cache").exists()
    assert (tmp_path / "outputs").exists()


def test_from_env_picks_up_overrides(monkeypatch) -> None:
    reset_config()
    monkeypatch.setenv("FLOODHUB_API_KEY", "test-key-123")
    monkeypatch.setenv("RVI_REQUEST_TIMEOUT", "42")
    cfg = Config.from_env()
    assert cfg.floodhub_api_key == "test-key-123"
    assert cfg.request_timeout_s == pytest.approx(42.0)


def test_set_and_get_config_round_trip() -> None:
    custom = Config(rvi_alpha=0.5, rvi_beta=0.25, rvi_gamma=0.25)
    set_config(custom)
    try:
        assert get_config() is custom
    finally:
        reset_config()


def test_severity_constant_export_matches_dataclass_default() -> None:
    cfg = Config()
    for k, v in DEFAULT_SEVERITY_INT.items():
        assert cfg.severity_int[k] == v


def test_nairobi_bbox_constant_is_well_formed() -> None:
    w, s, e, n = DEFAULT_NAIROBI_BBOX
    assert w < e
    assert s < n
