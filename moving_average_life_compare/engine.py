"""Causal, long-only EMA episodes with optional fictional life costs.

Prices and trades use floating point BTC quantities, with no lot-size or tax
assumptions. Signals are fixed at the previous close; execution is at today's
open, so an existing position bears the overnight gap. Every segment supplies
its own rows and consequently resets both EMA indicators.
"""

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Config:
    initial_cash: float = 100.0
    initial_life: float = 90.0
    seconds_per_bar: float = 5.0
    pack_cost: float = 8.0
    pack_seconds: float = 15.0
    max_packs: int = 6
    commission: float = 0.001
    slippage: float = 0.0005
    fast_period: int = 15
    slow_period: int = 30
    buy_tolerance: float = 0.00015
    horizon_bars: int = 30

    def __post_init__(self):
        for field in ("initial_cash", "initial_life", "seconds_per_bar", "pack_seconds"):
            if not math.isfinite(getattr(self, field)) or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be finite and positive")
        if not math.isfinite(self.pack_cost) or self.pack_cost < 0:
            raise ValueError("pack_cost must be finite and nonnegative")
        if not 0 <= self.commission < 1 or not 0 <= self.slippage < 1:
            raise ValueError("commission and slippage must be in [0, 1)")
        if self.fast_period < 1 or self.slow_period < self.fast_period:
            raise ValueError("EMA periods must satisfy 1 <= fast <= slow")
        if self.max_packs < 0 or self.horizon_bars < 1 or self.buy_tolerance < 0:
            raise ValueError("invalid pack, horizon or tolerance configuration")


def ema_values(closes: Sequence[float], period: int) -> list:
    """LEAN-style EMA: first ready value is the first period's SMA.

    A period-n EMA is ready on close n-1, hence first usable at open n.
    """
    if period < 1:
        raise ValueError("period must be positive")
    values = [float(x) for x in closes]
    if any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("close prices must be finite and positive")
    out = [None] * len(values)
    if len(values) < period:
        return out
    state = sum(values[:period]) / period
    out[period - 1] = state
    alpha = 2.0 / (period + 1)
    for i in range(period, len(values)):
        state = alpha * values[i] + (1.0 - alpha) * state
        out[i] = state
    return out


def _date(row, index):
    return str(row.get("date", row.get("open_time_utc", row.get("open_time", index))))


def _settle_death(cash, qty, life, price):
    """Forfeit an exhausted account; this is destruction, not a market sale.

    Default safe exits prevent this state. Keep a defined settlement for an
    externally exhausted account, including the pre-forfeit marked wealth.
    """
    if life is None or life > 0:
        raise ValueError("a living account cannot be settled as dead")
    return {"cash": 0.0, "qty": 0.0, "life": 0.0,
            "destroyed_wealth": cash + qty * price}


def run_episode(rows: Sequence[Mapping], start: int, variant: str,
                cfg: Config = None, trace: bool = False,
                horizon_bars: int = None) -> dict:
    """Replay one episode; variant is original, life_only, or life_buy.

    Life-only agents leave safely at an open when remaining life <= one step.
    Life-buy agents plan a pack at <= two steps if the previous-close entry
    signal is bullish. At execution, a pack may require a partial BTC sale;
    affordability is checked only then. A failed purchase exits safely.
    There are no terminal purchases. Cash waits following ordinary sell signals.

    ``trace`` contains open execution records, a close mark for each consumed
    bar, and a final close liquidation when the common horizon is reached.
    ``action`` is planned before today's open; ``executed_actions`` documents
    execution including failed purchases. All end values are after liquidation.
    """
    cfg = cfg or Config()
    if variant not in ("original", "life_only", "life_buy"):
        raise ValueError(f"unknown variant: {variant}")
    horizon = cfg.horizon_bars if horizon_bars is None else horizon_bars
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon_bars must be a positive integer")
    if start < cfg.slow_period or start + horizon > len(rows):
        raise ValueError("episode needs complete prior EMA warmup and horizon")
    closes = [float(r["close"]) for r in rows]
    opens = [float(r["open"]) for r in rows]
    if any(not math.isfinite(x) or x <= 0 for x in opens):
        raise ValueError("open prices must be finite and positive")
    fast = ema_values(closes, cfg.fast_period)
    slow = ema_values(closes, cfg.slow_period)
    cash, qty = cfg.initial_cash, 0.0
    life = None if variant == "original" else cfg.initial_life
    packs = 0
    life_spend = 0.0
    fees = 0.0
    slip_cost = 0.0
    trade_count = 0
    bars = 0
    elapsed = 0.0
    destroyed_wealth = 0.0
    records = []
    reason = None
    end_index = start
    event_fees = event_slip = 0.0

    def sell(amount, mid):
        nonlocal cash, qty, fees, slip_cost, trade_count, event_fees, event_slip
        amount = min(qty, max(0.0, amount))
        if amount <= 1e-14:
            return
        execution = mid * (1.0 - cfg.slippage)
        fee = amount * execution * cfg.commission
        slip = amount * mid * cfg.slippage
        cash += amount * execution - fee
        qty -= amount
        if abs(qty) < 1e-12:
            qty = 0.0
        fees += fee
        slip_cost += slip
        event_fees += fee
        event_slip += slip
        trade_count += 1

    def buy(mid):
        nonlocal cash, qty, fees, slip_cost, trade_count, event_fees, event_slip
        if cash <= 1e-12:
            return
        execution = mid * (1.0 + cfg.slippage)
        amount = cash / (execution * (1.0 + cfg.commission))
        fee = amount * execution * cfg.commission
        slip = amount * mid * cfg.slippage
        qty += amount
        cash = 0.0
        fees += fee
        slip_cost += slip
        event_fees += fee
        event_slip += slip
        trade_count += 1

    def record(index, event, action, executed, mid, f, s, life_before,
               seconds=0.0, signal_index=None):
        if not trace:
            return
        records.append({
            "date": _date(rows[index], index), "bar_index": index,
            "event": event, "action": action,
            "executed_actions": list(executed), "price": mid,
            "signal_bar_index": signal_index,
            "signal_date": _date(rows[signal_index], signal_index) if signal_index is not None else None,
            "fast_ema": f, "slow_ema": s,
            "cash": cash, "qty": qty, "wealth": cash + qty * mid,
            "life_before": life_before, "life": life,
            "remaining_life": life, "seconds_spent": seconds,
            "elapsed_seconds": elapsed, "bars_elapsed": bars,
            "time_packs": packs, "life_spend": life_spend,
            "commission_paid": event_fees, "slippage_cost": event_slip,
            "cumulative_commission": fees, "cumulative_slippage": slip_cost,
        })

    for t in range(start, start + horizon):
        end_index = t
        f, s = fast[t - 1], slow[t - 1]
        bullish = f > s * (1.0 + cfg.buy_tolerance)
        bearish = f < s
        life_before = life
        # These plans depend only on the prior close and previous account state.
        want_pack = (variant == "life_buy" and life <= 2 * cfg.seconds_per_bar
                     and bullish and packs < cfg.max_packs)
        safe_exit = life is not None and life <= cfg.seconds_per_bar and not want_pack
        market_plan = "buy" if bullish and qty == 0 else "sell" if bearish and qty > 0 else "hold"
        plan = "safe_exit" if safe_exit else ("buy_time+" + market_plan if want_pack else market_plan)
        executed = []
        event_fees = event_slip = 0.0
        open_price = opens[t]

        if safe_exit:
            sell(qty, open_price)
            reason = "safe_life_exit"
            executed.append("safe_exit")
            record(t, "open", plan, executed, open_price, f, s, life_before, signal_index=t - 1)
            break

        if want_pack:
            net_unit = open_price * (1.0 - cfg.slippage) * (1.0 - cfg.commission)
            available = cash + qty * net_unit
            if available + 1e-10 < cfg.pack_cost:
                sell(qty, open_price)
                executed.append("unaffordable_time_exit")
                reason = "unaffordable_time_exit"
                record(t, "open", plan, executed, open_price, f, s, life_before, signal_index=t - 1)
                break
            if cash < cfg.pack_cost:
                sell((cfg.pack_cost - cash) / net_unit, open_price)
                executed.append("sell_for_time")
            cash = max(0.0, cash - cfg.pack_cost)
            life += cfg.pack_seconds
            life_spend += cfg.pack_cost
            packs += 1
            executed.append("buy_time")

        # Do not top up an existing position after funding a time purchase.
        if market_plan == "sell":
            sell(qty, open_price)
            executed.append("sell")
        elif market_plan == "buy":
            buy(open_price)
            executed.append("buy")
        else:
            executed.append("hold")
        record(t, "open", plan, executed, open_price, f, s, life_before, signal_index=t - 1)

        bars += 1
        elapsed += cfg.seconds_per_bar
        if life is not None:
            life -= cfg.seconds_per_bar
        event_fees = event_slip = 0.0
        record(t, "close", "mark", [], closes[t], f, s, life_before,
               cfg.seconds_per_bar, signal_index=t - 1)
        if life is not None and life <= 0:
            settlement = _settle_death(cash, qty, life, closes[t])
            cash, qty, life = settlement["cash"], settlement["qty"], settlement["life"]
            destroyed_wealth = settlement["destroyed_wealth"]
            reason = "death"
            record(t, "death_settlement", "death", ["forfeit_assets"], closes[t], f, s, life,
                   signal_index=t - 1)
            if trace:
                records[-1]["destroyed_wealth"] = destroyed_wealth
            break
    else:
        reason = "horizon_end"
        event_fees = event_slip = 0.0
        had_qty = qty > 0
        sell(qty, closes[end_index])
        record(end_index, "close_liquidation", "horizon_exit",
               ["sell"] if had_qty else ["hold"], closes[end_index],
               fast[end_index - 1], slow[end_index - 1], life,
               signal_index=end_index - 1)

    if cash < -1e-8 or qty < -1e-12:
        raise ArithmeticError("negative cash or short position")
    if qty != 0:
        raise ArithmeticError("episode must finish flat")
    return {
        "variant": variant, "start_index": start, "end_index": end_index,
        "start_date": _date(rows[start], start), "end_date": _date(rows[end_index], end_index),
        "cash": cash, "wealth": cash, "remaining_life": life,
        "elapsed_seconds": elapsed, "bars_elapsed": bars,
        "time_packs": packs, "life_spend": life_spend,
        "trading_pnl": cash + life_spend - cfg.initial_cash,
        "trades": trade_count, "commission_paid": fees,
        "slippage_cost": slip_cost, "reason": reason,
        "death": reason == "death", "destroyed_wealth": destroyed_wealth,
        "trace": records if trace else [],
    }
