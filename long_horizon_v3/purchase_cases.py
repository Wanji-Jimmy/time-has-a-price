"""Collect every actual dp30 mid-episode time purchase, without rerunning it.

The illustration is the first chronological event across all fixed settings,
not the most profitable episode.  One unchanged model is rebuilt only to expose
the seven pre-outcome decision values at that recorded state.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
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
from time_budget_policy_v2 import planner as old
from time_budget_policy_v2.market import arrays

OUT = ROOT / "long_horizon_v3"


def source_directory(setting):
    return OUT / "runs" / (f"cost{setting['pack_cost']:g}_alpha{setting['alpha']:g}"
                            f"_L{setting['initial_life']:g}")


def read_events(settings):
    events = []
    source_counts = []
    for setting in settings:
        directory = source_directory(setting)
        count = 0
        with gzip.open(directory / "ledger.csv.gz", "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["controller"] != "dp30" or int(float(row["mid_packs"])) <= 0:
                    continue
                count += 1
                price = float(row["pack_cost"])
                events.append({
                    "symbol": row["symbol"], "controller": "dp30", "pack_cost": price,
                    "alpha": float(row["alpha"]), "initial_life": float(row["initial_life"]),
                    "start_index": int(row["start_index"]), "date": row["date"],
                    "step": int(row["step"]), "steps_remaining": 30 - int(row["step"]),
                    "cash_before": float(row["cash_before"]),
                    "holding_before": float(row["holding_before"]),
                    "life_before": float(row["life_before"]),
                    "packs_before": int(row["packs_before"]),
                    "market_state": int(row["market_state"]),
                    "intended_action": int(row["intended_action"]),
                    "executed_action": int(row["executed_action"]),
                    "executed_action_name": old.ACTION_NAMES[int(row["executed_action"])],
                    "mid_packs": int(float(row["mid_packs"])),
                    "price_label": "free" if price == 0 else "paid",
                    "scope_label": "exploratory historical replay; overlapping episodes; not an investment return",
                })
        source_counts.append({**setting, "mid_purchase_events": count,
                              "ledger": str(directory / "ledger.csv.gz"),
                              "ledger_sha256": hashlib.sha256((directory / "ledger.csv.gz").read_bytes()).hexdigest()})
    events.sort(key=lambda r: (r["date"], r["symbol"], r["start_index"], r["step"],
                               r["pack_cost"], r["alpha"], r["initial_life"]))
    return events, source_counts


def decision_json(result):
    q = np.asarray(result["q_values"])
    return {"action_index": int(result["action_index"]), "action_name": str(result["action_name"]),
            "planned_utility": float(result["planned_utility"]),
            "q_values": [float(x) if np.isfinite(x) else None for x in q],
            "action_values": old.action_value_table(result),
            "retire_q": float(q[0]), "best_buy_minus_best_no_buy": float(q[4:].max() - q[:4].max()),
            "best_continue_minus_retire": float(q[1:].max() - q[0])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-seconds", type=float, default=180.)
    args = parser.parse_args()
    protocol = json.loads((OUT / "protocol.json").read_text())
    settings = protocol["settings"]
    started = time.monotonic()
    while True:
        missing = [s for s in settings if not (source_directory(s) / "complete.json").exists()]
        if not missing:
            break
        if time.monotonic() - started >= args.wait_seconds:
            raise TimeoutError(f"Still waiting for {len(missing)} of {len(settings)} fixed replay settings")
        time.sleep(min(10., max(.1, args.wait_seconds - (time.monotonic() - started))))
    events, sources = read_events(settings)
    output = {
        "status": "complete", "settings_complete": len(settings),
        "selection_rule": "All dp30 mid-purchase events retained. Illustration is earliest by date, symbol, start_index, step; exact ties use pack_cost, alpha, initial_life ascending.",
        "interpretation": "Exploratory historical illustration on previously inspected 2026 data. Events belong to overlapping episodes, not independent observations. A free-time case is distinct from paying for time. Q values are model expected utility, not investment returns.",
        "event_count": len(events),
        "free_event_count": sum(e["pack_cost"] == 0 for e in events),
        "paid_event_count": sum(e["pack_cost"] > 0 for e in events),
        "events": events, "sources": sources, "first_case": None,
    }
    if events:
        event = events[0]
        kernel = arrays(json.loads((ROOT / "time_budget_policy_v2/kernel.json").read_text()), exact_retirement=True)
        cfg = PlannerConfig(initial_life=event["initial_life"], pack_cost=event["pack_cost"])
        model = FullHorizonPlanner(kernel, event["alpha"], event["initial_life"], cfg).build()
        state = (event["cash_before"], event["holding_before"], event["life_before"],
                 event["packs_before"], event["steps_remaining"], event["market_state"])
        longer = model.choose(*state)
        shorter = old.choose(*state, event["alpha"], kernel, depth=2, config=cfg)
        if int(longer["action_index"]) != event["intended_action"]:
            raise AssertionError("Rebuilt first-purchase decision disagrees with the recorded intended action")
        output["first_case"] = {
            "event": event,
            "dp30": decision_json(longer), "dp2": decision_json(shorter),
            "metadata_after_query": model.metadata(),
            "cash_spent_on_this_time_purchase": event["pack_cost"] * event["mid_packs"],
            "extra_trading_time_seconds": cfg.time_pack * event["mid_packs"],
            "caution": "The choice is computed before the recorded next opening gap and intraday return are observed. No realised profit was used to select the illustration.",
        }
    destination = OUT / "purchase_cases.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"event_count": output["event_count"],
                      "free_event_count": output["free_event_count"],
                      "paid_event_count": output["paid_event_count"],
                      "first_event": events[0] if events else None,
                      "first_case_dp30": output["first_case"]["dp30"] if events else None,
                      "first_case_dp2": output["first_case"]["dp2"] if events else None}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
