"""Convex Leader Momentum — builderr Trading Round 2 agent.

One sentence: in a confirmed uptrend hold the four strongest trending leaders,
park the remaining cash in QQQ and, in a calm uptrend, add a 3x sleeve on
whichever index is stronger (TQQQ for the Nasdaq-100, SOXL for semiconductors);
on a trend break keep the leaders at half size, step fully to cash on a fast
crash signal, and re-enter as soon as QQQ is back above a rising 20-day average.

Design rules
  * decide() depends only on its inputs: no clock, no randomness, no file or
    network access. The only module state is a per-session order counter and a
    memo of the orders already returned for identical inputs, keyed by the date
    of the latest QQQ bar. A repeated identical call returns identical orders
    and is not counted twice; past MAX_DAY_ORDERS only sells pass. With one call
    per session (live scoring) none of this binds. The orders themselves are a
    pure function of the inputs: when a call trades anyway it also trims names
    within 0.3 points of a trim line, so a re-call after the fills stays idle.
  * Garbage-in safety: unparseable bars are skipped, a held position that cannot
    be read blocks all buys, and a latest close that jumps implausibly (or
    disagrees with last_prices) freezes that ticker for the session.
  * Regime hysteresis is rebuilt from the price history on every call, and
    current holdings get a small rank buffer, so the bot does not churn.
  * Hard safety clamp: every order list is checked against a projected
    post-trade book; per-name weight <= 21% and beta-adjusted gross <= 1.32x,
    well inside the 30% / 1.5x rules, and names are trimmed back once they
    drift past 21% (so even a +36% one-day jump stays under 27%); levered
    names are cut first whenever held gross drifts above 1.34x (so a 3x +55% /
    1x +12% day stays near 1.45x). At most 8 orders per call.
    Standard library only.
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------- parameters
CFG = {
    # regime (QQQ and SPY versus their 50-day averages, with hysteresis)
    "TREND_SMA": 50,
    "BAND_IN": 0.030,        # need +3% above trend to switch risk on
    "BAND_OUT": 0.030,       # switch off only on a -3% break below trend
    "REGIME_LOOKBACK": 60,   # days replayed to rebuild the hysteresis state
    # fast crash brake (any one triggers; stays on for BRAKE_HOLD days)
    "BRAKE_R3": -0.05,       # QQQ 3-day return
    "BRAKE_R5": -0.07,       # QQQ 5-day return
    "BRAKE_VOL10": 0.45,     # QQQ 10-day annualised volatility
    "BRAKE_HOLD": 3,
    # V-recovery re-entry: strong 10-day QQQ thrust re-enables risk early
    "THRUST_R10": 0.08,
    # fast re-entry after a trend-break exit: QQQ above a rising 20-day average
    "REENTRY_SMA": 20,
    "REENTRY_SLOPE": 5,      # "rising" = SMA20 above its value 5 sessions earlier
    # leader selection
    "MOM_LONG": 63,
    "MOM_SHORT": 21,
    "W_LONG": 0.5,
    "W_SHORT": 0.3,
    "W_GAP": 0.2,
    "NAME_SMA": 50,
    "TOP_N": 4,
    "HOLD_BUFFER": 2,        # a held name stays while ranked <= TOP_N + buffer
    "NAME_W": 0.19,          # target weight per leader
    "STOCKS": 1,             # 1 = rank single stocks too, 0 = ETFs only
    "VOL_ADJ": 0,            # 1 = divide the momentum score by 20-day volatility
    # strong state and the 3x sleeve
    "STRONG_VOL20": 0.28,    # QQQ 20-day annualised vol must be below this
    "SLEEVE_W": 0.15,        # dollar weight of the 3x sleeve (beta-gross 0.45)
    "STRONG_NAME_W": 0.19,
    "FILL_TO": 0.97,         # ON/STRONG: top up with QQQ (SPY if QQQ is a leader) to this invested fraction
    "OFF_W": 0.5,            # trend-break OFF (not a crash BRAKE): leaders held at this fraction of size
    # trading hygiene
    "REBAL_BAND": 0.05,      # ignore weight changes smaller than this
    "SLEEVE_BAND": 0.03,     # tighter band for 2x/3x names (they drift 3x faster)
    "TRIM_AT": 0.21,         # always trim a name above this weight
    "MIN_HISTORY": 64,
    # hard clamp (inside the official 0.30 / 1.50 limits)
    "CLAMP_NAME": 0.21,
    "CLAMP_GROSS": 1.32,     # buys never take projected beta-gross above this
    "GROSS_TRIM": 1.34,      # held beta-gross above this -> cut levered names first
    "BUY_PRICE_PAD": 0.01,   # size buys as if they fill 1% above last close
    "CASH_KEEP": 0.005,      # never spend the last 0.5% of equity
    "MAX_ORDERS": 8,         # per call (sells first, riskiest first, so a cut defers buys and small sells)
    "MAX_DAY_ORDERS": 40,    # per-session order budget (rule: <= 50 trades a day)
    "HARD_DAY_ORDERS": 48,   # past the budget only sells pass, up to this many
}

INDEXES = ("SPY", "QQQ")
STOCK_LEADERS = (
    "NVDA", "MSFT", "AAPL", "META", "AMZN", "GOOGL", "AVGO", "AMD", "MU", "MRVL",
    "NFLX", "TSLA", "PLTR", "ORCL", "CRM", "JPM", "V", "MA", "COST", "LLY",
)
ETF_LEADERS = (
    "SMH", "SOXX", "XLK", "XLC", "XLY", "XLF", "XLI", "XLE", "XLV", "XLP", "XLU",
    "XLRE", "IWM", "QQQ", "SPY", "DIA",
)
FILLS = ("QQQ", "SPY")       # where spare cash is parked, in order of preference
BETA_3X = frozenset({"TQQQ", "SOXL", "UPRO", "SPXL", "TNA", "FAS", "TECL", "LABU",
                     "CURE", "DRN", "UDOW", "NAIL"})
BETA_2X = frozenset({"QLD", "SSO", "DDM", "ROM", "UWM", "AGQ"})


def _beta(t: str) -> float:
    return 3.0 if t in BETA_3X else 2.0 if t in BETA_2X else 1.0


# ---------------------------------------------------------------- data helpers
def _is_day(x: str) -> bool:
    return len(x) >= 10 and x[4] == x[7] == "-" and (x[:4] + x[5:7] + x[8:10]).isdigit()


def _daily_closes(bars) -> list[float]:
    """Close per calendar date, oldest first. Collapses intraday bars to daily.

    Unparseable bars are skipped; a ticker whose LATEST bar is bad counts as missing."""
    if not isinstance(bars, list) or not bars:
        return []
    try:
        keys = [str(b.get("ts", "")) for b in bars]
    except Exception:
        return []
    valid = [_is_day(k) for k in keys]
    clean = any(valid)
    if clean and not all(valid):   # drop bars whose timestamp is not a clean date
        bars = [b for b, ok in zip(bars, valid) if ok]
        keys = [k for k, ok in zip(keys, valid) if ok]
    if clean and any(keys[k] > keys[k + 1] for k in range(len(keys) - 1)):
        bars = [bars[k] for k in sorted(range(len(bars)), key=keys.__getitem__)]
    out: list[float] = []
    last_day = None
    ok = False
    for b in bars:
        try:
            c = float(b["close"])
            day = str(b.get("ts", ""))[:10]
        except Exception:
            ok = False
            continue
        ok = math.isfinite(c) and c > 0.0
        if not ok:
            continue
        if day and day == last_day:
            out[-1] = c
        else:
            out.append(c)
            last_day = day
    return out if ok else []


def _sma(v: list[float], n: int, end: int | None = None):
    end = len(v) if end is None else end
    if n <= 0 or end < n:
        return None
    return sum(v[end - n:end]) / n


def _ret(v: list[float], n: int, end: int | None = None):
    end = len(v) if end is None else end
    if end - 1 - n < 0:
        return None
    a = v[end - 1 - n]
    return v[end - 1] / a - 1.0 if a > 0 else None


def _vol(v: list[float], n: int, end: int | None = None):
    end = len(v) if end is None else end
    if end - 1 - n < 0 or n < 2:
        return None
    r = [v[i] / v[i - 1] - 1.0 for i in range(end - n, end)]
    m = sum(r) / n
    return math.sqrt(sum((x - m) * (x - m) for x in r) / n) * math.sqrt(252.0)


# ---------------------------------------------------------------- regime
def _brake_at(q: list[float], end: int, c: dict) -> bool:
    r3, r5, v10 = _ret(q, 3, end), _ret(q, 5, end), _vol(q, 10, end)
    return bool((r3 is not None and r3 < c["BRAKE_R3"])
                or (r5 is not None and r5 < c["BRAKE_R5"])
                or (v10 is not None and v10 > c["BRAKE_VOL10"]))


def _regime(q: list[float], s: list[float], c: dict) -> str:
    """Return 'BRAKE', 'OFF', 'ON' or 'STRONG' for the latest bar.

    The hysteresis state is rebuilt by replaying the last REGIME_LOOKBACK days,
    so the answer depends only on the price history (pure, restart-safe)."""
    n = min(len(q), len(s))
    q, s = q[-n:], s[-n:]
    first = max(c["TREND_SMA"] + 11, n - c["REGIME_LOOKBACK"])
    if n < first + 1:
        return "OFF"
    on = None        # first replayed bar: start from where QQQ/SPY sit vs their trend
    brake_left = 0
    for end in range(first, n + 1):
        sq, ss = _sma(q, c["TREND_SMA"], end), _sma(s, c["TREND_SMA"], end)
        qc, sc = q[end - 1], s[end - 1]
        if _brake_at(q, end, c):
            brake_left = c["BRAKE_HOLD"]
            on = False
            continue
        if brake_left > 0:
            brake_left -= 1
        broken = qc < sq * (1 - c["BAND_OUT"]) or sc < ss * (1 - c["BAND_OUT"])
        if on is None:
            on = not broken and qc > sq and sc > ss
            continue
        if on:
            if qc < sq * (1 - c["BAND_OUT"]) or sc < ss * (1 - c["BAND_OUT"]):
                on = False
        else:
            above = qc > sq * (1 + c["BAND_IN"]) and sc > ss * (1 + c["BAND_IN"])
            r10 = _ret(q, 10, end)
            sq20 = _sma(q, 20, end)
            thrust = r10 is not None and r10 > c["THRUST_R10"] and sq20 is not None and qc > sq20
            a1 = _sma(q, c["REENTRY_SMA"], end)
            a0 = _sma(q, c["REENTRY_SMA"], end - c["REENTRY_SLOPE"])
            fast = a1 is not None and a0 is not None and qc > a1 and a1 > a0
            if brake_left == 0 and not broken and (above or thrust or fast):
                on = True
    if brake_left > 0 and not on:
        return "BRAKE"
    if not on:
        return "OFF"
    sq20, sq50 = _sma(q, 20), _sma(q, c["TREND_SMA"])
    v20 = _vol(q, 20)
    strong = (sq20 is not None and sq50 is not None and sq20 > sq50
              and v20 is not None and v20 < c["STRONG_VOL20"])
    return "STRONG" if strong else "ON"


# ---------------------------------------------------------------- selection
def _score(v: list[float], c: dict):
    if len(v) < c["MIN_HISTORY"]:
        return None
    rl, rs = _ret(v, c["MOM_LONG"]), _ret(v, c["MOM_SHORT"])
    sma = _sma(v, c["NAME_SMA"])
    if rl is None or rs is None or sma is None or sma <= 0:
        return None
    if v[-1] <= sma or rl <= 0:
        return None
    sc = c["W_LONG"] * rl + c["W_SHORT"] * rs + c["W_GAP"] * (v[-1] / sma - 1.0)
    if c["VOL_ADJ"]:
        vol = _vol(v, 20)
        if not vol:
            return None
        sc /= max(vol, 0.05)
    return sc


def _safe_score(v, c: dict):
    try:
        return _score(v, c) if v else None
    except Exception:
        return None


def _sane(v: list[float], lp, levered: bool) -> bool:
    """False if the latest close looks like a bad tick: an implausible one-day
    jump, or a clash with the engine's last_prices (buys are sized off it)."""
    if len(v) >= 2:
        r = v[-1] / v[-2]
        if r > (1.75 if levered else 1.5) or r < (0.35 if levered else 0.5):
            return False
    try:
        lp = float(lp)
    except Exception:
        return True
    return not (lp > 0 and math.isfinite(lp)) or abs(v[-1] / lp - 1.0) <= 0.2


def _targets(closes: dict, held: set, regime: str, c: dict) -> dict:
    if regime == "BRAKE" or (regime == "OFF" and not c["OFF_W"]):
        return {}
    pool = (STOCK_LEADERS + ETF_LEADERS) if c["STOCKS"] else ETF_LEADERS
    ranked = sorted(((sc, t) for t in pool if t in closes
                     for sc in [_safe_score(closes[t], c)] if sc is not None),
                    key=lambda x: (-x[0], x[1]))
    names = [t for _, t in ranked]
    keep = [t for t in names[: c["TOP_N"] + c["HOLD_BUFFER"]] if t in held and t not in FILLS]
    picks: list = []
    for t in keep + names:
        if len(picks) >= c["TOP_N"]:
            break
        if t in picks:
            continue
        picks.append(t)
    w = c["STRONG_NAME_W"] if regime == "STRONG" else c["NAME_W"]
    if regime == "OFF":   # trend break: keep the leaders at reduced size, no sleeve, no fill
        return {t: min(w * c["OFF_W"], c["CLAMP_NAME"]) for t in picks}
    tgt = {t: w for t in picks}
    sleeve = _sleeve(closes, c)
    if regime == "STRONG" and c["SLEEVE_W"] > 0 and sleeve:
        tgt[sleeve] = c["SLEEVE_W"]
    rest = c["FILL_TO"] - sum(tgt.values())
    fill = next((f for f in FILLS if f not in tgt and f in closes), None)
    if rest > 0.01 and fill:
        tgt[fill] = min(rest, c["NAME_W"])   # never stacked on a pick, never at the trim line
    return {t: min(w, c["CLAMP_NAME"]) for t, w in tgt.items()}


def _sleeve(closes: dict, c: dict):
    """TQQQ, or SOXL when the semiconductor index (SMH) trends more strongly than QQQ."""
    s_smh, s_qqq = _safe_score(closes.get("SMH"), c), _safe_score(closes.get("QQQ"), c)
    if s_smh is not None and (s_qqq is None or s_smh > s_qqq) and "SOXL" in closes:
        return "SOXL"
    return "TQQQ" if "TQQQ" in closes else None


# ---------------------------------------------------------------- orders
def _orders(tgt: dict, qty: dict, px: dict, cash: float, c: dict, allow_buys: bool = True,
            frozen: frozenset | set = frozenset(), tight: bool = False) -> list[dict]:
    """Orders that move the book toward tgt. Tickers in `frozen` are never bought.
    tight=True lowers every sell-side trigger slightly (used when trading anyway)."""
    ts_, tg_ = (0.003, 0.005) if tight else (0.0, 0.0)
    equity = cash + sum(q * px[t] for t, q in qty.items() if t in px)
    if equity <= 0:
        return []
    cur = {t: q * px[t] / equity for t, q in qty.items() if t in px}
    sells: dict[str, float] = {}
    buys: dict[str, float] = {}
    for t, w in cur.items():
        goal = tgt.get(t, 0.0)
        if goal == 0.0:
            sells[t] = qty[t]
        elif w > c["TRIM_AT"] - ts_ or w - goal > (c["SLEEVE_BAND"] if _beta(t) > 1 else c["REBAL_BAND"]) - ts_:
            sells[t] = min(qty[t], (w - min(goal, c["TRIM_AT"])) * equity / px[t])
    for t, goal in tgt.items():
        if t not in px or not allow_buys or t in frozen:
            continue
        w = cur.get(t, 0.0)
        if (w == 0.0 and goal > 0) or goal - w > c["REBAL_BAND"]:
            buys[t] = goal - w
    # leverage drift guard: if held beta gross is high, trim levered names first
    gross_now = sum(w * _beta(t) for t, w in cur.items())
    if gross_now > c["GROSS_TRIM"] - tg_:
        excess = gross_now - (c["GROSS_TRIM"] - 0.05)
        for t in sorted(cur, key=lambda x: (-_beta(x), x)):
            if excess <= 0 or _beta(t) == 1.0:
                break
            cut = min(cur[t] - (sells.get(t, 0.0) * px[t] / equity), excess / _beta(t))
            if cut > 0:
                sells[t] = sells.get(t, 0.0) + cut * equity / px[t]
                excess -= cut * _beta(t)
                buys.pop(t, None)
    # project the post-sell book, then clamp each buy
    post = {t: qty[t] - sells.get(t, 0.0) for t in qty}
    post_val = {t: q * px[t] for t, q in post.items() if t in px and q > 0}
    proceeds = sum(sells[t] * px[t] for t in sells) * 0.995
    spend = max(0.0, cash + proceeds - c["CASH_KEEP"] * equity)
    gross_val = sum(v * _beta(t) for t, v in post_val.items())
    # full exits sell the exact holding (no dust left behind); partial sells are floored.
    # Riskiest first (levered, then largest), so a cut at MAX_ORDERS only defers small sells and buys.
    out = [{"ticker": t, "side": "sell", "quantity": q if q >= qty[t] else _floor(q)}
           for t, q in sorted(sells.items(), key=lambda kv: (-_beta(kv[0]), -cur.get(kv[0], 0.0), kv[0]))
           if (q if q >= qty[t] else _floor(q)) > 0]
    for t in sorted(buys, key=lambda x: (-buys[x], x)):
        p = px[t] * (1 + c["BUY_PRICE_PAD"])
        room_name = c["CLAMP_NAME"] * equity - post_val.get(t, 0.0) * (1 + c["BUY_PRICE_PAD"])
        room_gross = (c["CLAMP_GROSS"] * equity - gross_val) / _beta(t)
        dollars = min(buys[t] * equity, room_name, room_gross, spend)
        q = _floor(dollars / p)
        if q <= 0 or q * p < 0.01 * equity:
            continue
        out.append({"ticker": t, "side": "buy", "quantity": q})
        spend -= q * p
        post_val[t] = post_val.get(t, 0.0) + q * p
        gross_val += q * p * _beta(t)
    return out[: c["MAX_ORDERS"]]


def _floor(x: float) -> float:
    y = max(x, 0.0) * 1000.0
    return math.floor(y) / 1000.0 if math.isfinite(y) else 0.0


# ---------------------------------------------------------------- entry point
_SESSION = {"day": None, "used": 0, "memo": {}}   # see design rules


def decide(market_state, portfolio_state, cash):
    """Return long-only orders [{ticker, side, quantity}] for the next session."""
    try:
        day = _session_day(market_state)
        if day is None:   # no clean session date: stay stateless
            return _decide(market_state, portfolio_state, cash, CFG)
        if day != _SESSION["day"]:
            _SESSION.update(day=day, used=0, memo={})
        key = _fingerprint(market_state, portfolio_state, cash)
        if key is not None and key in _SESSION["memo"]:
            return [dict(o) for o in _SESSION["memo"][key]]
        orders = _budget(_decide(market_state, portfolio_state, cash, CFG), CFG)
        if key is not None:
            _SESSION["memo"][key] = [dict(o) for o in orders]
        return orders
    except Exception:  # never crash the harness; hold position for this tick
        return []


def _session_day(market_state):
    """Date of the latest QQQ (else SPY) bar, only if it is a clean YYYY-MM-DD."""
    if not isinstance(market_state, dict):
        return None
    for want in ("QQQ", "SPY"):
        for t, v in market_state.items():
            if str(t).upper() == want and isinstance(v, list) and v:
                days = [str(b.get("ts", ""))[:10] for b in v if isinstance(b, dict)]
                days = [d for d in days if _is_day(d)]
                if days:
                    return max(days)
    return None


def _fingerprint(market_state, portfolio_state, cash):
    """Hashable summary of everything decide() reads (None if it cannot be built)."""
    try:
        mk = tuple(sorted((repr(t), len(v), repr(v[-1].get("ts")),
                           hash(tuple((b.get("ts"), b.get("close")) if isinstance(b, dict) else repr(b) for b in v)))
                          for t, v in market_state.items() if isinstance(v, list) and v))
        ps = portfolio_state if isinstance(portfolio_state, dict) else {}
        pos = ps.get("positions")
        pk = tuple(sorted(repr(sorted(p.items(), key=repr)) if isinstance(p, dict) else repr(p) for p in pos)) \
            if isinstance(pos, (list, tuple)) else repr(pos)
        lp = ps.get("last_prices")
        lk = tuple(sorted((repr(k), repr(x)) for k, x in lp.items())) if isinstance(lp, dict) else repr(lp)
        return (mk, pk, lk, repr(ps.get("cash")), repr(cash), type(portfolio_state).__name__)
    except Exception:
        return None


def _budget(orders: list, c: dict) -> list:
    """At most MAX_DAY_ORDERS orders per session; past that only sells pass
    (levered names first), up to HARD_DAY_ORDERS."""
    used = _SESSION["used"]
    if used + len(orders) > c["MAX_DAY_ORDERS"]:
        sells = sorted((o for o in orders if o["side"] == "sell"), key=lambda o: -_beta(o["ticker"]))
        orders = sells[:max(0, c["HARD_DAY_ORDERS"] - used)]
    _SESSION["used"] = used + len(orders)
    return orders


def _decide(market_state, portfolio_state, cash, c):
    if not isinstance(market_state, dict) or not market_state:
        return []
    book_ok = isinstance(portfolio_state, dict)   # False -> equity is uncertain: sells only
    ps = portfolio_state if book_ok else {}
    closes = {}
    for t, bars in market_state.items():
        v = _daily_closes(bars)
        if v:
            closes[str(t).upper()] = v
    try:
        cash_v = float(ps.get("cash", cash))
    except Exception:
        return []
    if not math.isfinite(cash_v):
        return []
    lp_raw = ps.get("last_prices")
    last_prices = {str(k).upper(): x for k, x in lp_raw.items()} if isinstance(lp_raw, dict) else {}
    suspect = {t for t, v in closes.items() if not _sane(v, last_prices.get(t), _beta(t) > 1.0)}
    qty: dict[str, float] = {}
    px: dict[str, float] = {t: v[-1] for t, v in closes.items()}
    positions = ps.get("positions", [])
    if not isinstance(positions, (list, tuple)):
        book_ok, positions = False, []
    for p in positions:
        try:
            t = str(p["ticker"]).upper()
            q = float(p["quantity"])
        except Exception:
            book_ok = False
            continue
        if not math.isfinite(q) or q < 0:
            book_ok = False
            continue
        if q > 0:
            qty[t] = qty.get(t, 0.0) + q
            if t in suspect or t not in closes:   # bad tick or no live bar: equity uncertain
                book_ok = False
            if t not in px:
                try:
                    lp = float(last_prices.get(t) or p.get("avg_cost") or 0.0)
                except Exception:
                    lp = 0.0
                if lp > 0 and math.isfinite(lp):
                    px[t] = lp
                else:
                    book_ok = False
    q_, s_ = closes.get("QQQ"), closes.get("SPY")
    if (not q_ or not s_ or len(q_) < c["MIN_HISTORY"] or len(s_) < c["MIN_HISTORY"]
            or "QQQ" in suspect or "SPY" in suspect):
        # Index data missing or suspect: hold everything, but still trim anything above TRIM_AT.
        return _orders({t: c["TRIM_AT"] for t in qty}, qty, px, cash_v,
                       dict(c, REBAL_BAND=1.0, SLEEVE_BAND=1.0), allow_buys=False, frozen=suspect) if qty else []
    regime = _regime(q_, s_, c)
    tgt = _targets(closes, set(qty), regime, c)
    out = _orders(tgt, qty, px, cash_v, c, allow_buys=book_ok, frozen=suspect)
    if out:   # trading anyway: also trim names sitting just under a trim line
        out = _orders(tgt, qty, px, cash_v, c, allow_buys=book_ok, frozen=suspect, tight=True)
    return out
