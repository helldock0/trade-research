from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prop.ftmo_simulator import FTMOSimResult, PhaseResult
from src.strategy.registry import KNOWN_STRATEGIES, list_strategies


def test_list_strategies_canonical():
    assert list_strategies() == [
        "cssm",
        "orb",
        "csmo",
        "htf_pullback",
        "d1_mom",
        "fail_fade",
        "vwap_fade",
        "msnr",
        "donchian_atr",
        "xau_htf_pullback",
        "htf_adx_pullback",
        "event_drift",
        "macro_swing",
        "tom",
        "imom",
        "dip",
    ]
    assert tuple(list_strategies()) == KNOWN_STRATEGIES


def test_parse_strategies_arg_default_and_subset():
    from scripts.run_all import parse_strategies_arg

    assert parse_strategies_arg(None) == list_strategies()
    assert parse_strategies_arg("") == list_strategies()
    assert parse_strategies_arg("cssm, orb") == ["cssm", "orb"]


def test_parse_strategies_arg_unknown_raises():
    from scripts.run_all import parse_strategies_arg

    with pytest.raises(ValueError, match="Unknown"):
        parse_strategies_arg("cssm,not_a_real_strat")


def test_run_batch_summary_shape_with_mocks(tmp_path, monkeypatch):
    from scripts import run_all

    monkeypatch.setattr(run_all, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_all, "ensure_dirs", lambda: None)

    ok_metrics = {"n_trades": 2, "profit_factor": 1.5}
    ok_result = MagicMock()
    ok_result.metrics = ok_metrics
    ok_result.trades = [object(), object()]
    ok_result.trades_frame.return_value = MagicMock(empty=True)

    ok_sim = FTMOSimResult(
        phase1=PhaseResult(
            phase="phase1",
            passed=False,
            breach="max_loss",
            ending_balance=90000.0,
            max_dd_pct=10.0,
            worst_daily_loss_pct=5.0,
            trading_days=1,
            profit_pct=-10.0,
            trades_used=2,
        ),
        phase2=None,
        overall_pass=False,
        notes=["Phase1 failed: max_loss"],
    )

    strategy_cfg = {
        "strategy": "cssm",
        "universe": ["EURUSD"],
    }

    call_count = {"n": 0}

    def fake_load_strategy(name: str):
        return {**strategy_cfg, "strategy": name.replace("strategy_", "").replace(".yaml", "")}

    def fake_run_backtest(*_a, **_k):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("boom")
        return ok_result

    with (
        patch.object(run_all, "load_costs_config", return_value={}),
        patch.object(run_all, "load_ftmo_config", return_value={}),
        patch.object(run_all, "load_strategy_config", side_effect=fake_load_strategy),
        patch.object(run_all, "resolve_strategy_config_name", side_effect=lambda s, c: f"strategy_{s}.yaml"),
        patch.object(run_all, "load_symbol_bars", return_value=MagicMock()),
        patch.object(run_all, "resolve_signal_fn", return_value=lambda *_a, **_k: []),
        patch.object(run_all, "run_backtest", side_effect=fake_run_backtest),
        patch.object(run_all, "simulate_ftmo_path", return_value=ok_sim),
        patch.object(run_all, "strategy_id", side_effect=lambda cfg: cfg["strategy"]),
    ):
        summary = run_all.run_batch(["cssm", "orb"])

    assert summary["strategies"] == ["cssm", "orb"]
    assert len(summary["results"]) == 2

    ok_row, err_row = summary["results"]
    for key in (
        "strategy",
        "config",
        "n_signals",
        "n_trades",
        "metrics",
        "overall_pass",
        "notes",
        "phase1_breach",
        "phase2_breach",
        "error",
    ):
        assert key in ok_row
        assert key in err_row

    assert ok_row["strategy"] == "cssm"
    assert ok_row["n_trades"] == 2
    assert ok_row["metrics"] == ok_metrics
    assert ok_row["overall_pass"] is False
    assert ok_row["error"] is None
    assert err_row["strategy"] == "orb"
    assert err_row["error"] == "boom"
    assert err_row["metrics"] is None
    assert (tmp_path / "reports" / "batch_summary.json").is_file()
