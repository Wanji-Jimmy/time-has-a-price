"""Finite lookahead with cash, holdings and remaining life as decision states.

This is planning under a finite empirical market kernel, not a future oracle.
Orders are chosen before seeing the next opening gap. Costs and terminal life
conversion are identical in every branch. The terminal utility scales (90 and
100) remain fixed even when the initial-life counterfactual changes.

Kernel arrays have shape (market_states, events): probabilities, gaps,
intraday, next_state. Holding is its value at the previous daily close. A
continuation observes a gap, executes the preselected controls at that open,
observes the intraday return, and consumes dt game seconds. A depth-limited
leaf retires at the following unknown open, except that the real finite
horizon liquidates at the current close. Every future state gets its own
decision, so this is contingent planning rather than an open-loop action list.
Optional retirement_gaps/retirement_probabilities replace only the retirement
integration distribution. Using a single conditional mean gap is an explicit
approximation and is not equivalent to full expected utility integration.

score_life is a separate preference state for the life-blind ablation. Actual
life always governs survival and controls. A caller may reset score_life to
90 at each observed decision; inside each lookahead it evolves with purchases
and elapsed game time. Actual outcomes are always evaluated with actual life.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Mapping

import numpy as np


ACTION_NAMES = (
    "retire", "cash", "half", "full",
    "buy_time_cash", "buy_time_half", "buy_time_full",
)
ACTION_FRACTIONS = np.array((0., 0., .5, 1., 0., .5, 1.))
ACTION_BUY_TIME = np.array((False, False, False, False, True, True, True))


@dataclass(frozen=True)
class PlannerConfig:
    initial_cash: float = 100.0
    initial_life: float = 90.0
    dt: float = 5.0
    horizon_steps: int = 30
    time_pack: float = 15.0
    pack_cost: float = 8.0
    max_packs: int = 6
    commission: float = .001
    slippage: float = .0005
    age_scale: float = 90.0
    wealth_scale: float = 100.0

    def __post_init__(self):
        for name in ("initial_life", "dt", "time_pack", "age_scale", "wealth_scale"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (self.initial_cash < 0 or self.pack_cost < 0 or self.max_packs < 0
                or self.horizon_steps < 1):
            raise ValueError("invalid resource or horizon configuration")
        if not (0 <= self.commission < 1 and 0 <= self.slippage < 1):
            raise ValueError("commission and slippage must lie in [0, 1)")


Config = PlannerConfig


def _config(config=None):
    if config is None:
        return PlannerConfig()
    if isinstance(config, PlannerConfig):
        return config
    return PlannerConfig(**(asdict(config) if is_dataclass(config) else dict(config)))


def _kernel(kernel):
    arrays = {key: np.asarray(kernel[key], dtype=int if key == "next_state" else float)
              for key in ("probabilities", "gaps", "intraday", "next_state")}
    p = arrays["probabilities"]
    if p.ndim != 2 or not p.size or any(x.shape != p.shape for x in arrays.values()):
        raise ValueError("all kernel arrays must have the same nonempty (M, E) shape")
    if not np.all(np.isfinite(p)) or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1., atol=1e-9, rtol=0):
        raise ValueError("kernel probabilities must be finite, nonnegative, and sum to one per state")
    for key in ("gaps", "intraday"):
        if np.any(~np.isfinite(arrays[key])) or np.any(arrays[key] <= 0):
            raise ValueError(f"{key} must be finite and positive, including padded outcomes")
    ns = arrays["next_state"]
    if np.any(ns < 0) or np.any(ns >= p.shape[0]):
        raise ValueError("next_state is outside the kernel state range")
    if np.any(np.asarray(kernel["next_state"]) != ns):
        raise ValueError("next_state must contain integer state identifiers")
    optional = ("retirement_gaps", "retirement_probabilities")
    if any(key in kernel for key in optional):
        if not all(key in kernel for key in optional):
            raise ValueError("retirement_gaps and retirement_probabilities must be supplied together")
        rg, rp = (np.asarray(kernel[key], float) for key in optional)
        if (rg.ndim != 2 or rg.shape[0] != p.shape[0] or rp.shape != rg.shape
                or not rg.size or np.any(~np.isfinite(rg)) or np.any(rg <= 0)
                or np.any(~np.isfinite(rp)) or np.any(rp < 0)
                or not np.allclose(rp.sum(axis=1), 1., atol=1e-9, rtol=0)):
            raise ValueError("invalid optional retirement-gap distribution")
        arrays.update(dict(zip(optional, (rg, rp))))
    return arrays


def utility(life, wealth, alpha, config=None):
    cfg = _config(config)
    life, wealth, alpha = np.broadcast_arrays(np.asarray(life, float),
                                              np.asarray(wealth, float),
                                              np.asarray(alpha, float))
    if np.any((alpha < 0) | (alpha > 1)):
        raise ValueError("alpha must lie in [0, 1]")
    value = _preference_utility(life, wealth, alpha, cfg)
    return np.where(life > 0, value, 0.)


def _preference_utility(score_life, wealth, alpha, cfg):
    # A shadow preference state is not a survival state. Clamp negative shadow
    # remaining time to zero for its preference term, while physical life alone
    # determines death. The ordinary depth-2 blind reset at 90 stays positive.
    return (alpha * np.log1p(np.maximum(0, score_life) / cfg.age_scale)
            + (1 - alpha) * np.log1p(np.maximum(0, wealth) / cfg.wealth_scale))


def _terminal_conversion(cash, life, packs_used, alpha, cfg, score_life=None):
    """Exact discrete cash-for-life optimization, after liquidation costs."""
    score_life = life if score_life is None else score_life
    cash, life, packs_used, alpha, score_life = np.broadcast_arrays(cash, life, packs_used, alpha, score_life)
    best = np.where(life > 0, _preference_utility(score_life, cash, alpha, cfg), 0.)
    best_count = np.zeros(cash.shape, dtype=int)
    for count in range(1, cfg.max_packs + 1):
        cost = count * cfg.pack_cost
        eligible = ((life > 0) & (packs_used + count <= cfg.max_packs)
                    & (cash + 1e-9 >= cost))
        candidate = _preference_utility(score_life + count * cfg.time_pack, cash - cost, alpha, cfg)
        improve = eligible & (candidate > best + 1e-12)
        best = np.where(improve, candidate, best)
        best_count = np.where(improve, count, best_count)
    alive = life > 0
    return {
        "cash": np.where(alive, np.maximum(0, cash - best_count * cfg.pack_cost), 0.),
        "life": np.where(alive, life + best_count * cfg.time_pack, 0.),
        "packs_used": packs_used + best_count,
        "score_life": score_life + best_count * cfg.time_pack,
        "planned_utility": best,
        "utility": utility(life + best_count * cfg.time_pack,
                           np.maximum(0, cash - best_count * cfg.pack_cost), alpha, cfg),
        "life_spend": best_count * cfg.pack_cost,
        "packs_bought": best_count,
        "death": ~alive,
    }


def _terminal_utility_only(cash, life, packs_used, alpha, cfg, score_life=None):
    """Same optimum as enumeration, using concavity to score at most 3 counts.

    Monetary pack cost and added life are constant, so terminal utility is
    concave in a continuous pack count. Its integer maximum is at the clipped
    stationary point's floor or ceiling. Actual settlements still enumerate
    all counts to retain the public smallest-count tie convention.
    """
    score_life = life if score_life is None else score_life
    cash, life, packs_used, alpha, score_life = np.broadcast_arrays(cash, life, packs_used, alpha, score_life)
    base = _preference_utility(score_life, cash, alpha, cfg)
    if cfg.pack_cost == 0:
        cap = np.maximum(0, cfg.max_packs - packs_used)
        best = np.maximum(base, _preference_utility(score_life + cap * cfg.time_pack, cash, alpha, cfg))
        return np.where(life > 0, best, 0.)
    cap = np.maximum(0, np.minimum(cfg.max_packs - packs_used,
                                   np.floor((cash + 1e-9) / cfg.pack_cost)))
    stationary = (alpha * (cfg.wealth_scale + cash) / cfg.pack_cost
                  - (1 - alpha) * (cfg.age_scale + score_life) / cfg.time_pack)
    clipped = np.clip(stationary, 0., cap)
    low, high = np.floor(clipped), np.ceil(clipped)
    best = base
    if np.any(low > 0):
        best = np.maximum(best, _preference_utility(score_life + low * cfg.time_pack,
                                                    cash - low * cfg.pack_cost, alpha, cfg))
    if np.any(high != low):
        best = np.maximum(best, _preference_utility(score_life + high * cfg.time_pack,
                                                    cash - high * cfg.pack_cost, alpha, cfg))
    # A negative shadow state introduces a clamp kink, so use exact enumeration
    # for those unusual callers instead of invoking the concavity shortcut.
    negative = score_life < 0
    if np.any(negative):
        exact = _terminal_conversion(cash[negative], life[negative], packs_used[negative],
                                      alpha[negative], cfg, score_life[negative])
        best = np.array(best, copy=True)
        best[negative] = exact["planned_utility"]
    return np.where(life > 0, best, 0.)


def settle_retirement(cash, holding_at_fill, life, packs_used, alpha, config=None, score_life=None):
    """Execute liquidation at an already observed fill mark, then buy time.

    holding_at_fill is a marked monetary value, not a share count. Use it at
    open for a retirement order and at close for actual horizon settlement.
    """
    cfg = _config(config)
    score_life = life if score_life is None else score_life
    cash, holding, life, packs, alpha, score_life = np.broadcast_arrays(
        np.asarray(cash, float), np.asarray(holding_at_fill, float),
        np.asarray(life, float), np.asarray(packs_used, int), np.asarray(alpha, float), np.asarray(score_life, float))
    sell_factor = (1 - cfg.slippage) * (1 - cfg.commission)
    result = _terminal_conversion(cash + holding * sell_factor, life, packs, alpha, cfg, score_life)
    live = life > 0
    commission = np.where(live, holding * (1 - cfg.slippage) * cfg.commission, 0.)
    slippage = np.where(live, holding * cfg.slippage, 0.)
    result.update({"holding": np.zeros(cash.shape), "wealth": result["cash"],
                   "commission": commission, "slippage_cost": slippage,
                   "fees": commission + slippage,
                   "turnover": np.where(live, holding, 0.),
                   "trades": (live & (holding > 1e-10)).astype(int),
                   "terminal_reason": np.where(live, "retire", "death")})
    return result


def apply_controls(cash, holding_at_open, life, packs_used, action_index, config=None, score_life=None):
    """Execute one continuation control (actions 1..6), before the daily bar.

    An unavailable/unaffordable time pack returns valid=False and leaves the
    state and all costs unchanged. A replay may then apply its prespecified
    order-rejection protection. The planner disallows a time-pack action if
    any positive-probability kernel branch would reject it.
    """
    cfg = _config(config)
    score_life = life if score_life is None else score_life
    cash, holding, life, packs, action, score_life = np.broadcast_arrays(
        np.asarray(cash, float), np.asarray(holding_at_open, float),
        np.asarray(life, float), np.asarray(packs_used, int), np.asarray(action_index, int),
        np.asarray(score_life, float))
    if np.any((action < 1) | (action > 6)):
        raise ValueError("apply_controls accepts continuation action indices 1..6")
    cash0, holding0, life0, packs0 = cash.copy(), holding.copy(), life.copy(), packs.copy()
    target, buy_time = ACTION_FRACTIONS[action], ACTION_BUY_TIME[action]
    buy_factor = (1 + cfg.slippage) * (1 + cfg.commission)
    sell_factor = (1 - cfg.slippage) * (1 - cfg.commission)
    valid = ((life > 0) & (~buy_time | ((packs < cfg.max_packs)
              & (cash + holding * sell_factor + 1e-9 >= cfg.pack_cost))))
    pay_pack = buy_time & valid
    pack_sale = np.where(pay_pack, np.maximum(0, cfg.pack_cost - cash) / sell_factor, 0.)
    pack_sale = np.minimum(pack_sale, holding)
    cash = cash + pack_sale * sell_factor - np.where(pay_pack, cfg.pack_cost, 0.)
    holding = holding - pack_sale
    life = life + np.where(pay_pack, cfg.time_pack, 0.)
    packs = packs + pay_pack.astype(int)
    # Solve for the desired post-cost portfolio weight, not a pre-cost weight.
    change = target * (cash + holding) - holding
    buy_notional = np.where(valid & (change > 0), change / (1 + target * (buy_factor - 1)), 0.)
    sell_notional = np.where(valid & (change < 0), -change / (1 - target * (1 - sell_factor)), 0.)
    cash = cash - buy_notional * buy_factor + sell_notional * sell_factor
    holding = holding + buy_notional - sell_notional
    commission = ((pack_sale + sell_notional) * (1 - cfg.slippage)
                  + buy_notional * (1 + cfg.slippage)) * cfg.commission
    slippage = (pack_sale + sell_notional + buy_notional) * cfg.slippage
    return {"cash": np.where(valid, np.maximum(0, cash), cash0),
            "holding": np.where(valid, np.maximum(0, holding), holding0),
            "life": np.where(valid, life, life0),
            "score_life": score_life + np.where(pay_pack, cfg.time_pack, 0.),
            "packs_used": np.where(valid, packs, packs0), "valid": valid,
            "commission": commission, "slippage_cost": slippage,
            "fees": commission + slippage,
            "turnover": pack_sale + sell_notional + buy_notional,
            "trades": ((pack_sale > 1e-10).astype(int)
                       + (buy_notional > 1e-10).astype(int)
                       + (sell_notional > 1e-10).astype(int)),
            "life_spend": np.where(pay_pack, cfg.pack_cost, 0.),
            "packs_bought": pay_pack.astype(int)}


def _retirement_value(cash, holding, life, packs, market_state, alpha, kernel, cfg, score_life=None):
    score_life = life if score_life is None else score_life
    cash, holding, life, packs, state, alpha, score_life = np.broadcast_arrays(
        cash, holding, life, packs, np.asarray(market_state, int), alpha, score_life)
    shape = cash.shape
    result = np.empty(shape)
    no_assets = holding == 0
    if np.any(no_assets):
        result[no_assets] = _terminal_utility_only(cash[no_assets], life[no_assets],
                                                  packs[no_assets], alpha[no_assets], cfg, score_life[no_assets])
    have_assets = ~no_assets
    if np.any(have_assets):
        selected_state = state[have_assets]
        p = kernel.get("retirement_probabilities", kernel["probabilities"])[selected_state]
        gap = kernel.get("retirement_gaps", kernel["gaps"])[selected_state]
        net_cash = (cash[have_assets, None] + holding[have_assets, None] * gap
                    * (1 - cfg.slippage) * (1 - cfg.commission))
        values = _terminal_utility_only(net_cash, life[have_assets, None],
                                        packs[have_assets, None], alpha[have_assets, None], cfg,
                                        score_life[have_assets, None])
        result[have_assets] = np.sum(p * values, axis=-1)
    return result


def retirement_value(cash, holding, life, packs_used, market_state, alpha, kernel,
                     config=None, score_life=None):
    """Expected retirement utility before knowing the next opening quote."""
    cfg, ker = _config(config), _kernel(kernel)
    args = np.broadcast_arrays(np.asarray(cash, float), np.asarray(holding, float),
                               np.asarray(life, float), np.asarray(packs_used, int),
                               np.asarray(market_state, int), np.asarray(alpha, float))
    return _retirement_value(*args, ker, cfg, score_life)


def transition(cash, holding, life, packs_used, steps_remaining, action_index,
               gap, intraday, next_state, alpha, config=None, score_life=None):
    """Execute one selected continuation under an observed market outcome.

    This helper rejects invalid controls unchanged; callers must implement the
    declared rejection protection before advancing the clock. Valid controls
    consume dt, with death preceding horizon settlement. At the actual horizon
    this liquidates at the close, without another random gap.
    """
    cfg = _config(config)
    cash, holding, life, packs, n, action, gap, body, ns, alpha = np.broadcast_arrays(
        np.asarray(cash, float), np.asarray(holding, float), np.asarray(life, float),
        np.asarray(packs_used, int), np.asarray(steps_remaining, int),
        np.asarray(action_index, int), np.asarray(gap, float), np.asarray(intraday, float),
        np.asarray(next_state, int), np.asarray(alpha, float))
    control = apply_controls(cash, holding * gap, life, packs, action, cfg, score_life)
    valid = control["valid"]
    life_after = np.where(valid, control["life"] - cfg.dt, life)
    score_after = control["score_life"] - np.where(valid, cfg.dt, 0.)
    holding_after = np.where(valid, control["holding"] * body, holding)
    n_after = n - valid.astype(int)
    dead = valid & (life_after <= 0)
    horizon = valid & ~dead & (n_after <= 0)
    result = dict(control)
    result.update({"cash": np.where(dead, 0., control["cash"]),
                   "holding": np.where(dead, 0., holding_after),
                   "life": np.where(dead, 0., life_after),
                   "score_life": score_after,
                   "steps_remaining": n_after, "market_state": np.where(valid, ns, 0),
                   "death": dead, "done": dead | horizon,
                   "utility": np.zeros(cash.shape),
                   "planned_utility": np.zeros(cash.shape),
                   "terminal_reason": np.where(dead, "death", np.where(horizon, "horizon", "continue"))})
    terminal = settle_retirement(control["cash"], holding_after, life_after,
                                 control["packs_used"], alpha, cfg, score_after)
    for key in ("cash", "holding", "life", "score_life", "packs_used", "utility", "planned_utility"):
        result[key] = np.where(horizon, terminal[key], result[key])
    for key in ("commission", "slippage_cost", "fees", "turnover", "trades", "life_spend", "packs_bought"):
        result[key] = result[key] + np.where(horizon, terminal[key], 0)
    return result


def _values(cash, holding, life, packs, remaining, state, alpha, score_life, kernel, depth, cfg):
    """Batch Bellman expansion; all incoming states are flattened arrays."""
    n = len(cash)
    retire = _retirement_value(cash, holding, life, packs, state, alpha, kernel, cfg, score_life)
    p = kernel["probabilities"][state][:, None, :]
    gap = kernel["gaps"][state][:, None, :]
    body = kernel["intraday"][state][:, None, :]
    ns = kernel["next_state"][state][:, None, :]
    action = np.arange(1, 7)[None, :, None]
    control = apply_controls(cash[:, None, None], holding[:, None, None] * gap,
                             life[:, None, None], packs[:, None, None], action, cfg,
                             score_life[:, None, None])
    shape = control["cash"].shape
    next_life = control["life"] - cfg.dt
    next_score_life = control["score_life"] - cfg.dt
    next_holding = control["holding"] * body
    next_remaining = np.broadcast_to(remaining[:, None, None] - 1, shape)
    next_state = np.broadcast_to(ns, shape)
    next_alpha = np.broadcast_to(alpha[:, None, None], shape)
    alive = control["valid"] & (next_life > 0)
    terminal_at_close = alive & (next_remaining <= 0)
    continuing = alive & ~terminal_at_close
    branch_value = np.zeros(shape)
    if np.any(terminal_at_close):
        sel = terminal_at_close
        settled = settle_retirement(control["cash"][sel], next_holding[sel], next_life[sel],
                                     control["packs_used"][sel], next_alpha[sel], cfg, next_score_life[sel])
        branch_value[sel] = settled["planned_utility"]
    if np.any(continuing):
        sel = continuing
        args = (control["cash"][sel], next_holding[sel], next_life[sel],
                control["packs_used"][sel], next_state[sel], next_alpha[sel])
        if depth > 1:
            future_q = _values(args[0], args[1], args[2], args[3],
                               next_remaining[sel], args[4], args[5], next_score_life[sel],
                               kernel, depth - 1, cfg)
            branch_value[sel] = np.max(future_q, axis=1)
        else:
            branch_value[sel] = _retirement_value(*args, kernel, cfg, next_score_life[sel])
    expected = np.sum(p * branch_value, axis=-1)
    # A rejected pack cannot be financed with borrowing or ignored. The market
    # kernel's zero-probability padding does not invalidate any action.
    robust_valid = ~np.any((p > 0) & ~control["valid"], axis=-1)
    expected = np.where(robust_valid, expected, -np.inf)
    q = np.column_stack((retire, expected))
    # A dead state is terminal and cannot be resurrected with a time purchase.
    q[life <= 0, :] = 0.
    return q


def choose(cash, holding, life, packs_used, steps_remaining, market_state,
           alpha, kernel, depth=2, config=None, chunk_size=32, score_life=None):
    """Select a depth-limited, contingent action for broadcastable state arrays.

    Returns q_values (..., 7), action_index, action_name, target_fraction,
    buy_time, retire, and planned_utility. Ties within 1e-12 prefer retirement,
    then lower unextended exposure, then life-purchasing actions. This is a
    limited-depth optimum under the supplied empirical kernel, not a claim of
    global or real-market optimality.
    """
    cfg, ker = _config(config), _kernel(kernel)
    if not isinstance(depth, int) or not 1 <= depth <= 3:
        raise ValueError("depth must be an integer from one to three")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    score_life = life if score_life is None else score_life
    args = np.broadcast_arrays(np.asarray(cash, float), np.asarray(holding, float),
                               np.asarray(life, float), np.asarray(packs_used, int),
                               np.asarray(steps_remaining, int), np.asarray(market_state, int),
                               np.asarray(alpha, float), np.asarray(score_life, float))
    shape = args[0].shape
    flat = [x.ravel() for x in args]
    if any(np.any(~np.isfinite(x)) for x in flat):
        raise ValueError("planner states and preferences must be finite")
    b, h, l, k, remaining, state, a, scoring_life = flat
    if (np.any(b < -1e-8) or np.any(h < -1e-8) or np.any(l < 0)
            or np.any(k < 0) or np.any(k > cfg.max_packs)
            or np.any(remaining < 1) or np.any(state < 0)
            or np.any(state >= ker["probabilities"].shape[0])
            or np.any((a < 0) | (a > 1))):
        raise ValueError("invalid planner state")
    q = np.empty((len(b), 7))
    for first in range(0, len(b), chunk_size):
        last = min(len(b), first + chunk_size)
        q[first:last] = _values(*(x[first:last] for x in flat), ker, depth, cfg)
    maximum = np.max(q, axis=1)
    action = np.argmax(q >= maximum[:, None] - 1e-12, axis=1)
    selected_value = q[np.arange(len(q)), action]
    action = action.reshape(shape)
    return {"action_index": action,
            "action_name": np.asarray(ACTION_NAMES, dtype=object)[action],
            "target_fraction": ACTION_FRACTIONS[action],
            "buy_time": ACTION_BUY_TIME[action], "retire": action == 0,
            "q_values": q.reshape(shape + (7,)),
            "planned_utility": selected_value.reshape(shape)}


def action_value_table(result):
    """Readable action rows for a single state returned by choose()."""
    q = np.asarray(result["q_values"])
    if q.shape != (7,):
        raise ValueError("action_value_table requires a single scalar planner state")
    chosen = int(result["action_index"])
    return [{"action_index": i, "action": name,
             "target_fraction": float(ACTION_FRACTIONS[i]),
             "buy_time": bool(ACTION_BUY_TIME[i]),
             "q_value": float(q[i]) if np.isfinite(q[i]) else None,
             "valid": bool(np.isfinite(q[i])), "selected": i == chosen}
            for i, name in enumerate(ACTION_NAMES)]
