from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from rvi.cli import main


def test_pilot_cli_uses_ascii_correlation_output(monkeypatch, tmp_path) -> None:
    result_obj = SimpleNamespace(
        run_dir=tmp_path / "outputs" / "pilot_run",
        map_path=tmp_path / "outputs" / "pilot_run" / "map.html",
        correlation={
            "30m_p75": SimpleNamespace(
                rho=0.5,
                n=2,
                ci_low=0.1,
                ci_high=0.8,
                pvalue=0.04,
            )
        },
        manifest_path=tmp_path / "outputs" / "pilot_run" / "manifest.json",
    )

    monkeypatch.setattr("rvi.cli.get_config", lambda: SimpleNamespace())
    monkeypatch.setattr("rvi.pipeline.run_pilot", lambda **_kwargs: result_obj)

    runner = CliRunner()
    result = runner.invoke(main, ["pilot", "--area", "ascii"])

    assert result.exit_code == 0
    assert "Spearman rho - upstream RVI vs Flood Hub severity" in result.output
    assert "rho=+0.500" in result.output
    assert "ρ" not in result.output
