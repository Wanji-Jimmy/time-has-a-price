"""Real-data, causal market states and an auditable empirical scenario model."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LOOKBACK = 21


@dataclass
class Data:
    symbol: str
    dates: list[str]
    opens: np.ndarray
    closes: np.ndarray
    momentum: np.ndarray
    volatility: np.ndarray


def past_features(closes):
    """Index t uses strictly closes before t; include the known end-of-day state."""
    closes = np.asarray(closes, float)
    momentum = np.full(len(closes)+1, np.nan)
    volatility = momentum.copy()
    for t in range(LOOKBACK, len(closes)+1):
        past = closes[t-21:t]
        momentum[t] = past[-1]/past[-6]-1
        volatility[t] = np.std(np.diff(np.log(past)), ddof=1)
    return momentum, volatility


def load_csv(path, symbol="BTCUSDT"):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    opens = np.array([float(r["open"]) for r in rows])
    closes = np.array([float(r["close"]) for r in rows])
    momentum, volatility = past_features(closes)
    return Data(symbol, [r["open_time_utc"] for r in rows], opens, closes, momentum, volatility)


def states(data, kernel):
    return ((data.momentum >= 0).astype(int)*2
            + (data.volatility >= kernel["volatility_threshold"]).astype(int))


def arrays(kernel, exact_retirement=False):
    keys = ("probabilities", "gaps", "intraday", "next_state")
    result = {key: np.array(kernel[key], dtype=int if key=="next_state" else float) for key in keys}
    if not exact_retirement:
        result["retirement_probabilities"] = np.ones((4, 1))
        result["retirement_gaps"] = np.sum(result["probabilities"]*result["gaps"], axis=1, keepdims=True)
    return result


def fit_kernel(data_list, smoothing):
    vol = np.concatenate([d.volatility[LOOKBACK:len(d.dates)] for d in data_list])
    threshold = float(np.median(vol))
    observed = []
    for data in data_list:
        market = ((data.momentum>=0).astype(int)*2 + (data.volatility>=threshold).astype(int))
        for t in range(LOOKBACK, len(data.dates)):
            observed.append((int(market[t]), int(market[t+1]),
                             float(data.opens[t]/data.closes[t-1]),
                             float(data.closes[t]/data.opens[t]), data.dates[t]))
    current = np.array([r[0] for r in observed])
    next_state = np.array([r[1] for r in observed])
    gap, body = (np.array([r[k] for r in observed]) for k in (2, 3))
    edges = np.quantile(body-1, [.25,.5,.75])
    group = next_state*4 + np.searchsorted(edges, body-1, side="right")
    global_counts = np.bincount(group, minlength=16)
    global_probability = global_counts/global_counts.sum()
    probs, gaps, bodies, successors = [], [], [], []
    sources, compression = [], []
    log_scale = np.maximum(np.std(np.column_stack((np.log(gap), np.log(body))), axis=0), 1e-8)

    def representative(indices):
        z = np.column_stack((np.log(gap[indices]), np.log(body[indices])))
        distances = np.sum(((z-z.mean(axis=0))/log_scale)**2, axis=1)
        return int(indices[np.argmin(distances)])

    for state in range(4):
        indices = np.flatnonzero(current==state)
        counts = np.bincount(group[indices], minlength=16)
        probabilities = (counts+smoothing*global_probability)/(len(indices)+smoothing)
        g, r, dates = np.ones(16), np.ones(16), []
        for event in range(16):
            own = indices[group[indices]==event]
            backoff = not len(own)
            relevant = own if len(own) else np.flatnonzero(group==event)
            if len(relevant):
                selected = representative(relevant)
                g[event], r[event] = gap[selected], body[selected]
                dates.append({"event": event, "date": observed[selected][4], "global_backoff": backoff,
                              "gap": float(g[event]), "intraday": float(r[event])})
        probs.append(probabilities.tolist()); gaps.append(g.tolist()); bodies.append(r.tolist())
        successors.append((np.arange(16)//4).tolist()); sources.append(dates)
        raw = gap[indices]*body[indices]-1
        compressed = g*r-1
        mean = float(probabilities@compressed)
        compression.append({"market_state":state, "raw_n":len(indices),
                            "raw_mean_return":float(raw.mean()), "raw_std_return":float(raw.std()),
                            "raw_return_quantiles":np.quantile(raw,[0,.05,.5,.95,1]).tolist(),
                            "compressed_mean_return":mean,
                            "compressed_std_return":float(np.sqrt(probabilities@((compressed-mean)**2))),
                            "compressed_support_min_max":[float(compressed[probabilities>0].min()),float(compressed[probabilities>0].max())]})
    return {"type":"empirical-observed-state-markov", "smoothing":float(smoothing),
            "volatility_threshold":threshold, "intraday_quartile_edges":edges.tolist(),
            "probabilities":probs, "gaps":gaps, "intraday":bodies, "next_state":successors,
            "training_rows":len(observed), "training_first_date":observed[0][4],
            "training_last_date":observed[-1][4], "training_years":sorted({int(r[4][:4]) for r in observed}),
            "state_counts":np.bincount(current,minlength=4).tolist(),
            "representative_sources":sources, "compression_diagnostics":compression,
            "state_labels":["down5_lowvol","down5_highvol","up5_lowvol","up5_highvol"]}


def log_loss(data, kernel):
    market = states(data,kernel)
    body = data.closes[LOOKBACK:]/data.opens[LOOKBACK:]-1
    group = market[LOOKBACK+1:]*4 + np.searchsorted(kernel["intraday_quartile_edges"],body,side="right")
    p = np.array(kernel["probabilities"])[market[LOOKBACK:len(data.dates)],group]
    return {"mean_event_log_loss":float(-np.log(np.maximum(p,1e-12)).mean()),"n":len(p)}
