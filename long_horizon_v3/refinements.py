"""Independent numerical convergence probes: wealth steps 1 versus .5.

Run after or alongside experiment.py.  This does not rerun historical episodes,
select favourable outcomes, change preferences, refit the kernel, or overwrite
the production replay.  The fixed eight price-8 settings are always retained.
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
from long_horizon_v3.long_planner import FullHorizonPlanner, PlannerConfig
from time_budget_policy_v2.market import arrays

OUT = ROOT / "long_horizon_v3"
SETTINGS = list(itertools.product((.1, .02), (20., 45., 90., 120.)))


def _ledger_states(alpha, life):
    directory = OUT / "runs" / f"cost8_alpha{alpha:g}_L{life:g}"
    if not (directory / "complete.json").exists():
        return [], {"status": "replay not complete at numerical-check time", "unique_states": 0}
    seen = set()
    states = []
    with gzip.open(directory / "ledger.csv.gz", "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["controller"] != "dp30":
                continue
            state = (float(row["cash_before"]), float(row["holding_before"]),
                     float(row["life_before"]), int(row["packs_before"]),
                     30 - int(row["step"]), int(row["market_state"]))
            key = tuple(round(x, 12) if isinstance(x, float) else x for x in state)
            if key not in seen:
                seen.add(key)
                states.append(state)
    return states, {"status": "complete dp30 replay decision states included",
                    "unique_states": len(states), "file": str(directory / "ledger.csv.gz"),
                    "sha256": hashlib.sha256((directory / "ledger.csv.gz").read_bytes()).hexdigest()}


def _finite_list(values):
    return [float(x) if np.isfinite(x) else None for x in values]


def compare_queries(coarse, fine, states, *, include_rows):
    if not states:
        return {"n": 0}
    state_array = np.asarray(states, float)
    arguments = [state_array[:, i].astype(int) if i >= 3 else state_array[:, i] for i in range(6)]
    low, high = coarse.choose(*arguments), fine.choose(*arguments)
    qlow, qhigh = np.asarray(low["q_values"]), np.asarray(high["q_values"])
    finite_low, finite_high = np.isfinite(qlow), np.isfinite(qhigh)
    if np.any(finite_low != finite_high):
        raise AssertionError("wealth refinement changed a root action's financing feasibility")
    delta = np.where(finite_low, np.abs(np.where(finite_low, qhigh, 0.)
                                         - np.where(finite_low, qlow, 0.)), 0.)
    row_error = delta.max(axis=1)
    actions_low, actions_high = low["action_index"], high["action_index"]
    sorted_high = np.sort(qhigh, axis=1)
    gap = sorted_high[:, -1] - sorted_high[:, -2]
    agreement = actions_low == actions_high
    # This is an empirical stability screen, never a rigorous error bound.
    separated = gap > 2. * row_error + 1e-12
    result = {
        "n": len(states), "finite_action_q_comparisons": int(finite_low.sum()),
        "maximum_absolute_q_change": float(delta.max()),
        "mean_absolute_q_change": float(delta[finite_low].mean()),
        "maximum_per_state_q_change": float(row_error.max()),
        "action_agreement_count": int(agreement.sum()),
        "action_agreement_fraction": float(agreement.mean()),
        "changed_action_count": int((~agreement).sum()),
        "minimum_fine_best_vs_runner_up_gap": float(gap.min()),
        "maximum_fine_best_vs_runner_up_gap": float(gap.max()),
        "states_with_gap_over_twice_observed_refinement_change": int(separated.sum()),
        "numerically_close_states": int((~separated).sum()),
        "screen_interpretation": "Observed grid stability only; twice the coarse-to-fine change is not a proven error bound.",
    }
    if include_rows:
        result["rows"] = [
            {"cash": float(s[0]), "holding": float(s[1]), "life": float(s[2]),
             "packs_used": int(s[3]), "steps_remaining": int(s[4]), "market_state": int(s[5]),
             "coarse_action": str(low["action_name"][i]), "fine_action": str(high["action_name"][i]),
             "coarse_q_values": _finite_list(qlow[i]), "fine_q_values": _finite_list(qhigh[i]),
             "maximum_absolute_q_change": float(row_error[i]),
             "fine_best_vs_runner_up_gap": float(gap[i]),
             "fine_best_continue_minus_retire": float(qhigh[i, 1:].max() - qhigh[i, 0]),
             "fine_best_buy_minus_best_no_buy": float(qhigh[i, 4:].max() - qhigh[i, :4].max()),
             "action_agrees": bool(agreement[i]), "observed_gap_screen_passes": bool(separated[i])}
            for i, s in enumerate(states)]
    else:
        result["changed_action_states"] = [
            {"state": [float(x) for x in states[i]],
             "coarse_action": str(low["action_name"][i]), "fine_action": str(high["action_name"][i]),
             "maximum_absolute_q_change": float(row_error[i]),
             "fine_best_vs_runner_up_gap": float(gap[i])}
            for i in np.flatnonzero(~agreement)]
    return result


def run_setting(setting):
    alpha, life = setting
    kernel = arrays(json.loads((ROOT / "time_budget_policy_v2/kernel.json").read_text()), exact_retirement=True)
    cfg = PlannerConfig(initial_life=life, pack_cost=8.)
    started = time.perf_counter()
    coarse = FullHorizonPlanner(kernel, alpha, life, cfg, fine=False).build()
    fine = FullHorizonPlanner(kernel, alpha, life, cfg, fine=True).build()
    probes = [(cash, 100. - cash, life, 0, 30, market)
              for cash, market in itertools.product((100., 50.), range(4))]
    initial = compare_queries(coarse, fine, probes, include_rows=True)
    ledger_states, ledger_meta = _ledger_states(alpha, life)
    ledger = compare_queries(coarse, fine, ledger_states, include_rows=False)
    # Metadata must be captured after all probes, so the query diagnostics are real.
    coarse_meta, fine_meta = coarse.metadata(), fine.metadata()
    for meta in (coarse_meta, fine_meta):
        diagnostics = meta["query_diagnostics"]
        if (meta["maximum_internal_wealth_extrapolation"] > 1e-8
                or diagnostics["root_children_above_grid"]
                or diagnostics["root_children_below_grid"]
                or diagnostics["future_reachability_failures"]):
            raise AssertionError("a refinement comparison exceeded a reachable wealth grid")
    return {"pack_cost": 8., "alpha": alpha, "initial_life": life,
            "initial_probes": initial, "historical_decision_states": ledger,
            "historical_state_source": ledger_meta,
            "coarse_metadata_after_queries": coarse_meta,
            "fine_metadata_after_queries": fine_meta,
            "elapsed_seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    started = time.time()
    output = {
        "purpose": "Fixed numerical convergence check; no historical replay repeated",
        "status": "running", "settings_expected": 8,
        "fixed_settings": [{"pack_cost": 8., "alpha": a, "initial_life": l} for a, l in SETTINGS],
        "coarse_step": 1., "fine_step": .5,
        "kernel_sha256": hashlib.sha256((ROOT / "time_budget_policy_v2/kernel.json").read_bytes()).hexdigest(),
        "interpretation": "Grid refinement checks numerical stability, not out-of-sample profitability or a mathematical optimality certificate.",
        "started_at_unix": started, "results": [],
    }
    destination = OUT / "refinements.json"
    def save():
        destination.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    save()
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_setting, setting): setting for setting in SETTINGS}
        for future in as_completed(futures):
            result = future.result()
            output["results"].append(result)
            output["results"].sort(key=lambda r: (-r["alpha"], r["initial_life"]))
            save()
            initial = result["initial_probes"]
            print(json.dumps({"alpha": result["alpha"], "initial_life": result["initial_life"],
                              "completed": len(output["results"]),
                              "max_q_change": initial["maximum_absolute_q_change"],
                              "action_agreement": initial["action_agreement_fraction"],
                              "ledger_states": result["historical_decision_states"]["n"]}), flush=True)
    output["status"] = "complete"
    output["elapsed_seconds"] = time.time() - started
    output["summary"] = {
        "initial_probe_states": sum(r["initial_probes"]["n"] for r in output["results"]),
        "initial_changed_actions": sum(r["initial_probes"]["changed_action_count"] for r in output["results"]),
        "maximum_initial_absolute_q_change": max(r["initial_probes"]["maximum_absolute_q_change"] for r in output["results"]),
        "historical_unique_decision_states": sum(r["historical_decision_states"]["n"] for r in output["results"]),
        "historical_changed_actions": sum(r["historical_decision_states"].get("changed_action_count", 0) for r in output["results"]),
        "maximum_historical_absolute_q_change": max(r["historical_decision_states"].get("maximum_absolute_q_change", 0.) for r in output["results"]),
        "all_wealth_extrapolation_checks_passed": True,
    }
    save()
    print(json.dumps(output["summary"]), flush=True)


if __name__ == "__main__":
    main()
