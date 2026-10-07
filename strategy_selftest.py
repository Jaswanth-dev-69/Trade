"""Strategy-level checks for agent.py.

No network, no private engine, no third-party packages. These are not the
official builderr evals; they exercise the real decide() contract (well-formed
orders, caps, regime behaviour, determinism, bad-input safety) before submission.

Run:
    python strategy_selftest.py
"""
from __future__ import annotations

import importlib.util
import math
import time
from datetime import date, timedelta
from pathlib import Path

AGENT_PATH = Path(__file__).with_name("agent.py")
BETA = {"TQQQ": 3.0, "SOXL": 3.0, "UPRO": 3.0, "SPXL": 3.0, "QLD": 2.0, "SSO": 2.0}
UNIVERSE = (
    "SPY", "QQQ", "DIA", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLC",
    "SMH", "SOXX", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "AVGO", "MU", "TSLA", "JPM",
    "QLD", "SSO", "TQQQ", "SOXL",
)


def fresh_agent():
    """A freshly loaded module per test, like the live runner (no state leaks between tests)."""
    spec = importlib.util.spec_from_file_location(f"agent_under_test_{time.perf_counter_ns()}", AGENT_PATH)
    assert spec is not None and spec.loader is not None, AGENT_PATH
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bars(start: float, returns: list[float]) -> list[dict]:
    out, px, d = [], start, date(2024, 1, 1)
    for r in returns:
        px *= 1.0 + r
        out.append({"ts": d.isoformat(), "open": px, "high": px * 1.01, "low": px * 0.99, "close": px, "volume": 1_000_000})
        d += timedelta(days=1)
    return out


def market(kind: str, n: int = 150) -> dict[str, list[dict]]:
    if kind == "crash":   # steady uptrend, then a sharp 5-day fall
        path = [0.002] * (n - 5) + [-0.03] * 5
        return {t: bars(100.0, [x * BETA.get(t, 1.0) for x in path]) for t in UNIVERSE}
    if kind == "trend_break":   # indexes roll over slowly (no crash) while a few names keep trending
        idx = [0.002] * (n - 25) + [-0.005] * 25
        data = {t: bars(100.0, [x * BETA.get(t, 1.0) for x in idx]) for t in UNIVERSE}
        for t in ("NVDA", "AMD", "AVGO", "MU"):
            data[t] = bars(100.0, [0.004] * n)
        return data
    data = {t: bars(100.0, [0.001] * n) for t in UNIVERSE}            # calm risk-on
    for t in ("NVDA", "AMD", "AVGO", "MU", "SMH", "SOXX"):              # semis lead strongly
        data[t] = bars(100.0, [0.004] * n)
    for t in ("META", "AAPL"):
        data[t] = bars(100.0, [0.003] * n)
    data["QQQ"] = bars(100.0, [0.0025] * n)
    data["SPY"] = bars(100.0, [0.0018] * n)
    data["SOXL"] = bars(100.0, [0.012] * n)
    data["TQQQ"] = bars(100.0, [0.0075] * n)
    if kind == "nasdaq_leads":   # QQQ trends harder than semiconductors
        for t in ("NVDA", "AMD", "AVGO", "MU", "SMH", "SOXX"):
            data[t] = bars(100.0, [0.0015] * n)
        data["SOXL"] = bars(100.0, [0.0045] * n)
    return data


def portfolio(m, cash=100_000.0, positions=()):
    return {"cash": cash, "positions": list(positions), "last_prices": {t: b[-1]["close"] for t, b in m.items()}}


def well_formed(orders, m) -> bool:
    return isinstance(orders, list) and all(
        isinstance(o, dict) and o.get("side") in ("buy", "sell") and o.get("ticker") in m
        and isinstance(o.get("quantity"), (int, float)) and math.isfinite(o["quantity"]) and o["quantity"] > 0
        for o in orders)


def test_bad_inputs_never_raise() -> None:
    a = fresh_agent()
    m = market("risk_on")
    for args in (({}, {"cash": 1e5, "positions": []}, 1e5), (None, None, None),
                 (m, {"cash": float("nan"), "positions": []}, float("nan")), (m, "garbage", 1e5)):
        out = a.decide(*args)
        assert isinstance(out, list), args
        assert all(o["side"] == "sell" for o in out) or not out, out


def test_short_history_trades_nothing() -> None:
    a = fresh_agent()
    m = market("risk_on", n=30)
    assert a.decide(m, portfolio(m), 100_000.0) == []


def test_risk_on_deploys_inside_caps() -> None:
    a = fresh_agent()
    m = market("risk_on")
    orders = a.decide(m, portfolio(m), 100_000.0)
    assert well_formed(orders, m) and 0 < len(orders) <= 8, orders
    assert all(o["side"] == "buy" for o in orders), orders
    px = {t: b[-1]["close"] for t, b in m.items()}
    w = {o["ticker"]: o["quantity"] * px[o["ticker"]] / 100_000.0 for o in orders}
    assert a.CFG["CLAMP_NAME"] <= 0.25 and a.CFG["CLAMP_GROSS"] <= 1.40, a.CFG   # well inside 30% / 1.5x
    assert max(w.values()) <= a.CFG["CLAMP_NAME"] + 1e-9, w
    assert sum(v * BETA.get(t, 1.0) for t, v in w.items()) <= a.CFG["CLAMP_GROSS"] + 1e-9, w
    assert sum(w.values()) <= 1.0, w
    assert "SOXL" in w and "TQQQ" not in w, w                     # semiconductors trend harder here


def test_sleeve_follows_the_stronger_index() -> None:
    a = fresh_agent()
    m = market("nasdaq_leads")
    orders = a.decide(m, portfolio(m), 100_000.0)
    bought = {o["ticker"] for o in orders if o["side"] == "buy"}
    assert "TQQQ" in bought and "SOXL" not in bought, orders


def test_trend_break_halves_the_leaders() -> None:
    a = fresh_agent()
    m = market("trend_break")
    px = {t: b[-1]["close"] for t, b in m.items()}
    held = {t: 19_000.0 / px[t] for t in ("NVDA", "AMD", "AVGO", "MU")}
    orders = a.decide(m, portfolio(m, cash=24_000.0, positions=[
        {"ticker": t, "quantity": q, "avg_cost": px[t]} for t, q in held.items()]), 24_000.0)
    assert well_formed(orders, m) and orders and all(o["side"] == "sell" for o in orders), orders
    for o in orders:   # about half of each leader is sold, not all of it
        assert 0.35 <= o["quantity"] / held[o["ticker"]] <= 0.65, (o, held[o["ticker"]])


def test_crash_exits_to_cash() -> None:
    a = fresh_agent()
    m = market("crash")
    px = {t: b[-1]["close"] for t, b in m.items()}
    held = [{"ticker": t, "quantity": 190.0 / px[t] * 100, "avg_cost": px[t]} for t in ("NVDA", "META", "AAPL", "MSFT")]
    orders = a.decide(m, portfolio(m, cash=24_000.0, positions=held), 24_000.0)
    assert well_formed(orders, m), orders
    assert orders and all(o["side"] == "sell" for o in orders), orders
    assert {o["ticker"] for o in orders} == {"NVDA", "META", "AAPL", "MSFT"}, orders


def test_identical_inputs_identical_orders() -> None:
    a = fresh_agent()
    m = market("risk_on")
    first = a.decide(m, portfolio(m), 100_000.0)
    for _ in range(12):
        assert a.decide(m, portfolio(m), 100_000.0) == first


def test_unreadable_position_blocks_buys() -> None:
    a = fresh_agent()
    m = market("risk_on")
    orders = a.decide(m, portfolio(m, cash=80_000.0, positions=[{"ticker": "META", "avg_cost": 100.0}]), 80_000.0)
    assert not any(o["side"] == "buy" for o in orders), orders


def test_fast_with_large_universe() -> None:
    a = fresh_agent()
    m = market("risk_on", n=300)
    big = dict(m)
    for k in range(1000):
        big[f"X{k:04d}"] = m["SPY"]
    start = time.perf_counter()
    orders = a.decide(big, portfolio(big), 100_000.0)
    assert time.perf_counter() - start < 1.0
    assert len(orders) <= 8, orders


def run() -> None:
    tests = [
        test_bad_inputs_never_raise,
        test_short_history_trades_nothing,
        test_risk_on_deploys_inside_caps,
        test_sleeve_follows_the_stronger_index,
        test_trend_break_halves_the_leaders,
        test_crash_exits_to_cash,
        test_identical_inputs_identical_orders,
        test_unreadable_position_blocks_buys,
        test_fast_with_large_universe,
    ]
    for test in tests:
        test()
    print(f"✓ {len(tests)} strategy checks passed.")


if __name__ == "__main__":
    run()
