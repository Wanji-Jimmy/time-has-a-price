"""Full finite-horizon empirical Bellman planner, with one-dimensional gridding.

The original model's controls, financing, terminal conversion, costs and market
kernel are unchanged.  This module changes *only* the numerical solution method.
It recombines the contingent tree instead of truncating it after two decisions.

With a fixed episode initial life L0 and horizon H, every live decision satisfies
    life = L0 - dt * (H - steps_remaining) + time_pack * packs_used.
Consequently life is exact, not a gridded extra dimension.  After a continuation
the portfolio fraction is 0, 1, or body/(1+body).  The finite empirical kernel
therefore also provides an exact finite set of internal portfolio fractions.
Only wealth is linearly interpolated; no implicit rebalance is introduced.

Orders precede the unknown opening gap.  Each action's expectation is computed
BEFORE the maximum over actions.  Future decisions may condition on the next
observed market state.  R=0 uses current-close settlement; retirement at R>0
integrates the complete next-opening-gap distribution, as in the old planner.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from time_budget_policy_v2 import planner as original

ACTION_NAMES = original.ACTION_NAMES
ACTION_FRACTIONS = original.ACTION_FRACTIONS
ACTION_BUY_TIME = original.ACTION_BUY_TIME
PlannerConfig = original.PlannerConfig


def default_wealth_grid(fine: bool = False, ceiling: float = 640.) -> np.ndarray:
    """A priori coarse/fine grids; no holdout outcome is used to choose nodes."""
    if ceiling < 256:
        raise ValueError("wealth ceiling must be at least 256")
    if fine:
        central = np.unique(np.r_[np.arange(0., 40., 1.),
                                  np.arange(40., 180. + .25, .5),
                                  np.arange(181., 257., 1.)])
    else:
        central = np.arange(0., 257., 1.)
    tail = np.arange(272., ceiling, 16.)
    return np.unique(np.r_[central, tail, ceiling])


class FullHorizonPlanner:
    """Solve all actual remaining horizons, from zero through the episode end.

    Build one instance per fixed (kernel, alpha, initial_life, config, wealth
    grid).  ``choose`` accepts arbitrary *current* cash/holding, executes its
    first decision layer exactly, and looks up subsequent endogenous states.
    It does not interpolate the root Q values or silently clamp wealth.

    Default wealth grids have a time-dependent reachable upper bound.  Every
    positive-probability internal transition stays inside its child grid.
    Actual queries outside that domain raise instead of silently clamping or
    extrapolating.  Increase ``initial_wealth_cap`` if a historical replay needs
    a larger reachable domain. Custom fixed grids retain explicit extrapolation
    diagnostics, and should be used only for numerical convergence checks.
    """

    def __init__(self, kernel: dict[str, Any], alpha: float,
                 initial_life: float | None = None, config: Any = None,
                 wealth_grid: Any = None, *, fine: bool = False,
                 ceiling: float = 640., initial_wealth_cap: float = 120.,
                 allow_mid_buy: bool = True):
        self.config = original._config(config)
        if initial_life is not None:
            self.config = replace(self.config, initial_life=float(initial_life))
        self.initial_life = float(self.config.initial_life)
        self.alpha = float(alpha)
        self.allow_mid_buy = bool(allow_mid_buy)
        if not np.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be finite and in [0, 1]")
        self.kernel = original._kernel(kernel)
        self.horizon_steps = int(self.config.horizon_steps)
        self._growth_bound = (max(1., float(self.kernel["gaps"].max()))
                              * max(1., float(self.kernel["intraday"].max())))
        self.dynamic_bounds = wealth_grid is None
        self.initial_wealth_cap = float(initial_wealth_cap)
        if not np.isfinite(self.initial_wealth_cap) or self.initial_wealth_cap <= 0:
            raise ValueError("initial wealth cap must be finite and positive")
        if self.dynamic_bounds:
            step = .5 if fine else 1.
            self.wealth_grids = []
            for r in range(self.horizon_steps + 1):
                cap = self.initial_wealth_cap * self._growth_bound ** (self.horizon_steps - r)
                self.wealth_grids.append(np.unique(np.r_[np.arange(0., cap, step), cap]))
            self.wealth_grid = self.wealth_grids[0]
        else:
            self.wealth_grid = np.asarray(wealth_grid, float)
            self.wealth_grids = [self.wealth_grid] * (self.horizon_steps + 1)
        if (self.wealth_grid.ndim != 1 or len(self.wealth_grid) < 2
                or self.wealth_grid[0] != 0
                or np.any(~np.isfinite(self.wealth_grid))
                or np.any(np.diff(self.wealth_grid) <= 0)):
            raise ValueError("wealth grid must be finite, strictly increasing, and start at zero")
        self.values: list[np.ndarray] = []
        self.build_seconds = 0.
        self.query_diagnostics = {
            "queries": 0, "root_children_above_grid": 0,
            "root_children_below_grid": 0, "future_reachability_failures": 0,
            "maximum_root_child_wealth": 0., "maximum_conservative_future_wealth": 0.,
        }
        self._make_fraction_nodes()
        self.maximum_internal_extrapolation = 0.

    def life_at(self, remaining: Any, packs_used: Any) -> np.ndarray:
        return (self.initial_life - self.config.dt * (self.horizon_steps - np.asarray(remaining))
                + self.config.time_pack * np.asarray(packs_used))

    def _make_fraction_nodes(self) -> None:
        ker = self.kernel
        m_count, events = ker["probabilities"].shape
        fractions = []
        for market in range(m_count):
            mask = (ker["next_state"] == market) & (ker["probabilities"] > 0)
            bodies = ker["intraday"][mask]
            fractions.append(np.unique(np.r_[0., bodies / (1. + bodies), 1.]))
        self.fractions_by_state = fractions
        self.node_offsets = np.r_[0, np.cumsum([len(f) for f in fractions])]
        self.node_market = np.repeat(np.arange(m_count), [len(f) for f in fractions])
        self.node_fraction = np.concatenate(fractions)
        self.node_count = len(self.node_fraction)
        child = np.empty((3, m_count, events), dtype=int)
        for target_i, target in enumerate((0., .5, 1.)):
            for market in range(m_count):
                for event in range(events):
                    ns = int(ker["next_state"][market, event])
                    body = ker["intraday"][market, event]
                    fraction = target * body / (1. - target + target * body)
                    candidates = fractions[ns]
                    index = int(np.argmin(np.abs(candidates - fraction)))
                    # A zero-probability padding branch never enters expectations.
                    if ker["probabilities"][market, event] > 0 and abs(candidates[index] - fraction) > 1e-13:
                        raise AssertionError("finite portfolio-fraction state closure failed")
                    child[target_i, market, event] = self.node_offsets[ns] + index
        self._child_nodes = child

    def _wealth_coordinates(self, wealth: np.ndarray, remaining: int = 0):
        """Linear interpolation, explicit linear extrapolation at the top."""
        grid = self.wealth_grids[int(remaining)]
        lo = np.searchsorted(grid, wealth, side="right") - 1
        lo = np.clip(lo, 0, len(grid) - 2)
        weight = (wealth - grid[lo]) / (grid[lo + 1] - grid[lo])
        return lo, weight

    def _make_transitions(self, remaining: int) -> None:
        cfg, ker = self.config, self.kernel
        grid = self.wealth_grids[remaining][None, :, None]
        fraction = self.node_fraction[:, None, None]
        cash, holding = grid * (1. - fraction), grid * fraction
        market = self.node_market
        gap = ker["gaps"][market, None, :]
        body = ker["intraday"][market, None, :]
        self._probability = ker["probabilities"][market, None, :]
        self._transitions = []
        for action in range(1, 7):
            control = original.apply_controls(cash, holding * gap, 1., 0, action, cfg)
            child_wealth = control["cash"] + control["holding"] * body
            lo, weight = self._wealth_coordinates(child_wealth, remaining - 1)
            excess = float(np.maximum(0., child_wealth - self.wealth_grids[remaining - 1][-1]).max())
            self.maximum_internal_extrapolation = max(self.maximum_internal_extrapolation, excess)
            if self.dynamic_bounds and excess > 1e-8:
                raise AssertionError("internal transition exceeds the reachable child wealth grid")
            target_index = (action - 1) % 3
            child_node = self._child_nodes[target_index, market, :][:, None, :]
            base = child_node * len(self.wealth_grids[remaining - 1])
            robust = ~np.any((self._probability > 0) & ~control["valid"], axis=-1)
            self._transitions.append({
                "low": base + lo, "weight": weight, "valid": robust,
                "buy": bool(ACTION_BUY_TIME[action]),
                "max_child_wealth": float(child_wealth.max()),
                "cash": control["cash"], "holding": control["holding"] * body,
            })

    def _retirement_grid(self, remaining: int, *, at_close: bool = False):
        cfg, ker = self.config, self.kernel
        packs = np.arange(cfg.max_packs + 1)[:, None, None, None]
        life = self.life_at(remaining, packs)
        wealth = self.wealth_grids[remaining][None, None, :, None]
        fraction = self.node_fraction[None, :, None, None]
        cash, holding = wealth * (1. - fraction), wealth * fraction
        if at_close:
            gap, probability = 1., 1.
        else:
            gap = ker.get("retirement_gaps", ker["gaps"])[self.node_market][None, :, None, :]
            probability = ker.get("retirement_probabilities", ker["probabilities"])[self.node_market][None, :, None, :]
        sell_factor = (1. - cfg.slippage) * (1. - cfg.commission)
        net_cash = cash + holding * gap * sell_factor
        utility = original._terminal_utility_only(net_cash, life, packs, self.alpha, cfg)
        return np.sum(probability * utility, axis=-1)

    def _continuation(self, previous: np.ndarray, remaining: int, action: int):
        cfg = self.config
        tr = self._transitions[action - 1]
        k = np.arange(cfg.max_packs + 1)
        next_k = np.minimum(cfg.max_packs, k + tr["buy"])
        # Flat axes are (internal market/fraction node, wealth).  The endogenous
        # next fraction is exact; interpolation below concerns wealth alone.
        life = self.life_at(remaining, k)
        next_life = life + tr["buy"] * cfg.time_pack - cfg.dt
        if remaining == 1:
            branches = original._terminal_utility_only(
                tr["cash"][None, ...] + tr["holding"][None, ...]
                * (1. - cfg.slippage) * (1. - cfg.commission),
                next_life[:, None, None, None], next_k[:, None, None, None],
                self.alpha, cfg)
        else:
            source = previous[next_k].reshape(len(k), -1)
            low = source[:, tr["low"]]
            high = source[:, tr["low"] + 1]
            branches = low + (high - low) * tr["weight"][None, ...]
        q = np.sum(branches * self._probability[None, ...], axis=-1)
        q = np.where(next_life[:, None, None] > 0, q, 0.)
        valid = tr["valid"][None, ...] & (life[:, None, None] > 0)
        if tr["buy"]:
            valid = valid & (k[:, None, None] < cfg.max_packs)
        return np.where(valid, q, -np.inf)

    def build(self, progress=None, *, allow_mid_buy=None):
        """Build V[R] in float64 for every R=0..H, retaining each layer."""
        started = time.perf_counter()
        if allow_mid_buy is not None:
            self.allow_mid_buy = bool(allow_mid_buy)
        self.values = [self._retirement_grid(0, at_close=True)]
        for remaining in range(1, self.horizon_steps + 1):
            self._make_transitions(remaining)
            value = self._retirement_grid(remaining)
            for action in range(1, 7):
                if not self.allow_mid_buy and ACTION_BUY_TIME[action]:
                    continue
                value = np.maximum(value, self._continuation(self.values[-1], remaining, action))
            dead = self.life_at(remaining, np.arange(self.config.max_packs + 1)) <= 0
            value[dead] = 0.
            self.values.append(value)
            if progress is not None:
                progress(remaining, time.perf_counter() - started)
        self.build_seconds = time.perf_counter() - started
        return self

    def _lookup(self, remaining, packs, child_node, wealth):
        lo, weight = self._wealth_coordinates(wealth, remaining)
        value = self.values[int(remaining)]
        low = value[packs, child_node, lo]
        high = value[packs, child_node, lo + 1]
        return low + (high - low) * weight

    def choose(self, cash, holding, life, packs_used, steps_remaining,
               market_state, alpha=None, *, depth=None, score_life=None,
               record_diagnostics=True):
        """Return all seven root Q values and the original smallest-index tie rule.

        ``depth`` may be omitted, or must cover the whole actual remaining
        episode. Limited-depth comparisons should use the original exact
        ``planner.choose(depth=2)`` instead of relabelling a terminal boundary.
        """
        if not self.values:
            self.build()
        args = np.broadcast_arrays(np.asarray(cash, float), np.asarray(holding, float),
                                   np.asarray(life, float), np.asarray(packs_used, int),
                                   np.asarray(steps_remaining, int), np.asarray(market_state, int))
        shape = args[0].shape
        cash, holding, life, packs, remaining, market = [x.ravel() for x in args]
        if (any(np.any(~np.isfinite(x)) for x in args)
                or np.any(cash < -1e-8) or np.any(holding < -1e-8)
                or np.any(life < 0) or np.any(packs < 0) or np.any(packs > self.config.max_packs)
                or np.any(remaining < 1) or np.any(remaining > self.horizon_steps)
                or np.any(market < 0) or np.any(market >= len(self.fractions_by_state))):
            raise ValueError("invalid full-horizon planner state")
        if not np.allclose(life, self.life_at(remaining, packs), atol=1e-8, rtol=0):
            raise ValueError("life disagrees with the fixed initial-life/remaining-steps/pack invariant")
        if alpha is not None and not np.allclose(alpha, self.alpha, atol=0, rtol=0):
            raise ValueError("this table was built for a different alpha")
        if score_life is not None and not np.allclose(score_life, life, atol=1e-8, rtol=0):
            raise ValueError("this table represents actual life, not a reset shadow-life ablation")
        if depth is not None and np.any(np.asarray(depth) < remaining):
            raise ValueError("full-horizon depth must cover the actual remaining episode")
        current_wealth = cash + holding
        root_caps = np.array([self.wealth_grids[r][-1] for r in remaining])
        if self.dynamic_bounds and np.any(current_wealth > root_caps + 1e-8):
            raise ValueError("actual root wealth exceeds the dynamic reachable grid; increase initial_wealth_cap")
        ker, cfg = self.kernel, self.config
        retire = original.retirement_value(cash, holding, life, packs, market,
                                            self.alpha, ker, cfg)
        q = np.empty((len(cash), 7))
        q[:, 0] = retire
        probability = ker["probabilities"][market]
        gap, body = ker["gaps"][market], ker["intraday"][market]
        maximum_child = 0.
        maximum_future = 0.
        above, below = 0, 0
        for action in range(1, 7):
            if not self.allow_mid_buy and ACTION_BUY_TIME[action]:
                q[:, action] = -np.inf
                continue
            control = original.apply_controls(cash[:, None], holding[:, None] * gap,
                                              life[:, None], packs[:, None], action, cfg)
            next_life = control["life"] - cfg.dt
            next_wealth = control["cash"] + control["holding"] * body
            child_node = self._child_nodes[(action - 1) % 3, market, :]
            branch = np.zeros_like(next_wealth)
            for r in np.unique(remaining):
                selected = remaining == r
                if r == 1:
                    branch[selected] = original.settle_retirement(
                        control["cash"][selected], control["holding"][selected] * body[selected],
                        next_life[selected], control["packs_used"][selected], self.alpha, cfg)["planned_utility"]
                else:
                    branch[selected] = self._lookup(r - 1, control["packs_used"][selected],
                                                    child_node[selected], next_wealth[selected])
            branch = np.where(next_life > 0, branch, 0.)
            expected = np.sum(probability * branch, axis=-1)
            valid = ~np.any((probability > 0) & ~control["valid"], axis=-1)
            q[:, action] = np.where(valid, expected, -np.inf)
            eligible = (probability > 0) & control["valid"] & (next_life > 0)
            if np.any(eligible):
                maximum_child = max(maximum_child, float(next_wealth[eligible].max()))
                future_bound = next_wealth * self._growth_bound ** (remaining[:, None] - 1)
                maximum_future = max(maximum_future, float(future_bound[eligible].max()))
                child_caps = np.array([self.wealth_grids[r - 1][-1] for r in remaining])[:, None]
                above += int(np.sum(eligible & (next_wealth > child_caps + 1e-8)))
                below += int(np.sum(eligible & (next_wealth < 0)))
        q[life <= 0] = 0.
        maximum = np.max(q, axis=1)
        action = np.argmax(q >= maximum[:, None] - 1e-12, axis=1)
        selected_value = q[np.arange(len(q)), action]
        if record_diagnostics:
            d = self.query_diagnostics
            d["queries"] += len(cash)
            d["root_children_above_grid"] += above
            d["root_children_below_grid"] += below
            d["future_reachability_failures"] += int(maximum_future > self.wealth_grids[0][-1] + 1e-8)
            d["maximum_root_child_wealth"] = max(d["maximum_root_child_wealth"], maximum_child)
            d["maximum_conservative_future_wealth"] = max(d["maximum_conservative_future_wealth"], maximum_future)
        action = action.reshape(shape)
        return {"action_index": action,
                "action_name": np.asarray(ACTION_NAMES, dtype=object)[action],
                "target_fraction": ACTION_FRACTIONS[action],
                "buy_time": ACTION_BUY_TIME[action], "retire": action == 0,
                "q_values": q.reshape(shape + (7,)),
                "planned_utility": selected_value.reshape(shape)}

    def metadata(self):
        return {"method": "full finite-horizon recombining Bellman DP",
                "horizon_steps": self.horizon_steps, "alpha": self.alpha,
                "initial_life": self.initial_life, "config": asdict(self.config),
                "allow_mid_buy": self.allow_mid_buy,
                "wealth_points": [len(g) for g in self.wealth_grids],
                "wealth_ceiling": [float(g[-1]) for g in self.wealth_grids],
                "initial_wealth_cap": self.initial_wealth_cap,
                "dynamic_bounds": self.dynamic_bounds,
                "portfolio_nodes_per_market": [len(x) for x in self.fractions_by_state],
                "total_portfolio_nodes": self.node_count, "value_dtype": "float64",
                "interpolation": "piecewise linear in wealth only; exact finite portfolio fractions",
                "tail": "dynamic reachable grids; out-of-domain actual queries rejected" if self.dynamic_bounds else "explicit linear extrapolation; inspect reachability diagnostics",
                "maximum_internal_wealth_extrapolation": self.maximum_internal_extrapolation,
                "build_seconds": self.build_seconds,
                "value_table_bytes": sum(v.nbytes for v in self.values),
                "query_diagnostics": dict(self.query_diagnostics)}


RecombiningPlanner = FullHorizonPlanner


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", type=Path, default=_ROOT / "time_budget_policy_v2/kernel.json")
    parser.add_argument("--alpha", type=float, default=.1)
    parser.add_argument("--initial-life", type=float, default=90.)
    parser.add_argument("--pack-cost", type=float, default=8.)
    parser.add_argument("--fine", action="store_true")
    args = parser.parse_args()
    kernel = json.loads(args.kernel.read_text())
    model = FullHorizonPlanner(kernel, args.alpha, args.initial_life,
                              replace(PlannerConfig(), pack_cost=args.pack_cost), fine=args.fine).build()
    result = model.choose(100., 0., args.initial_life, 0, 30, 0)
    output = model.metadata()
    output["initial_state_decision"] = original.action_value_table(result)
    print(json.dumps(output, indent=2))
