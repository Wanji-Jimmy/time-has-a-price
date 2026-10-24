TIME HAS A PRICE — MODEL AND REPRODUCTION
V3: full 30-step planning, evaluated on previously inspected 2026 history

The question
When each market day consumes a finite time budget, should a controller continue, buy more budget, or stop? The project combines stochastic dynamic programming with optimal stopping. It uses observed prices for replay; future replay prices are never supplied to the planner.

Data and fixed rules
The frozen four-state market kernel was estimated from official BTC/USDT daily data for 2022–2025, with 16 empirical events per state. Replays use BTC and ETH from 1 January to 30 September 2026, 273 daily bars per asset. Every controller shares 213 starts, indices 30 through 242, and at most 30 market days per episode. Signals use the previous closing price or earlier information; orders are selected before the next opening price is read.

Initial cash: 100. Each market day consumes 5 budget seconds. A pack adds 15 budget seconds, with at most 6 packs. Initial budgets: 20, 45, 90 and 120. Primary pack price: 8; alternatives: 2 and 0. Primary alpha: 0.1; alternative: 0.02. Commission: 0.1% per side. Adverse slippage: 0.05% per side. No leverage or short positions.

The 2026 data were previously inspected. This is historical re-evaluation, not a new blind test. Budget seconds, exchange rates and utility preferences are assumptions, not observed human lifespan or medical effects.

Objective and actions
For positive remaining budget:
U(L, B) = alpha × ln(1 + L/90) + (1 − alpha) × ln(1 + B/100).
L is remaining budget and B is cash after trading costs and purchases. Alpha is time preference, not an upward-price probability. If L <= 0, utility is zero; exhaustion takes precedence over end-of-day settlement. A controller may exit before that happens. Q values are expected utilities, not money or return probabilities.

State: s = (market state m, cash C, holding value H, budget L, packs used k, market days remaining R).
Seven actions: exit; continue in cash, half invested or fully invested; buy one pack and continue in cash, half invested or fully invested.

The next opening gap is realized, the selected instruction executes, intraday return follows, and budget is deducted. On exit, holdings are liquidated and cash can optimally buy any affordable number of remaining packs. Every constrained controller has the same terminal conversion right. Such a terminal purchase does not extend trading.

Bellman equations
V_R(s) = max_a Q_R(s, a).
For continuation: Q_R(s, a) = sum_e p(e | m) × V_(R−1)(T(s, a, e)).
For exit: Q_R(s, exit) = sum_e p(e | m) × terminal utility at the unknown next opening.
At the actual 30-day horizon, settlement uses that day's closing price, without reading a further day.

Expectations are taken before choosing the action. Future actions can change only after observing a new state. Choosing a separate best action for each unobserved next event would incorrectly give the model foresight.

Decision criteria
Mid-episode purchase advantage = max(Q4, Q5, Q6) − max(Q0, Q1, Q2, Q3).
A purchase occurs only if affordable and better than exit and every no-purchase action.
Continuation advantage = max(Q1, ..., Q6) − Q0.
Exit when continuation is no better. Tiny differences should be assessed against numerical sensitivity. Terminal purchases are consumption of the assumed time preference, not evidence of better trading opportunities.

Why 30 steps are tractable
Equivalent reachable states are merged rather than expanded into a full decision tree. With initial budget L0:
L = L0 − 5 × (30 − R) + 15 × k.
No separate time grid is needed. Under the frozen event kernel and the three allocation controls, the next-period holding fraction has a finite set of reachable states, 61 nodes in total. Only wealth uses piecewise-linear interpolation, with primary spacing 1 and refinement spacing 0.5. Conservative bounds vary by remaining horizon; replay states cannot be silently clipped or extrapolated. This is a finite-model solution with a wealth-grid approximation, not a global optimum for real markets.

Comparators
- dp30: full 30-step planning.
- dp2: original two-step lookahead, recomputed at every decision; actual episodes may still last 30 days.
- dp30_no_mid_buy: separately solved 30-step planner that prohibits purchases during trading but retains terminal conversion.
- ema_time: public 15/30 EMA signal with the same finite budget and a safe exit before exhaustion.
- ema_mechanical: the same signal with a mechanical purchase when budget is at most 10 and the previous closing signal is bullish; safety rules remain the same.
- retire_now: immediate exit with the same terminal conversion.
- Original EMA: no time limit or conversion; reported separately as a cash-only reference, with no comparable time utility.

Cash accounting
Final cash = 100 + net trading P&L − mid-episode purchase spending − terminal conversion spending.
Trading P&L already includes transaction costs. Each episode resets to 100 cash. Overlapping episode means are not a continuous account's annual return.

Uncertainty
Within each asset, price, alpha, initial budget and start date, compare controllers in pairs. The circular moving-block bootstrap uses 2,000 samples, seed 20261024, with block lengths 30 and 60. A sensitivity check uses seven non-overlapping starts at stride 31. The 213 starts overlap heavily; neither they nor the 61,344 total episodes are independent trials. Intervals describe historical window-mean uncertainty, not all model error or future regime changes. All conditions are retained, without multiple-comparison adjustment.

Reproduction from the package root
python3 -m unittest discover -s long_horizon_v3 -p 'test_long_planner.py' -v
python3 long_horizon_v3/experiment.py --workers 2
python3 long_horizon_v3/refinements.py --workers 3
python3 long_horizon_v3/verify.py
python3 long_horizon_v3/update_report_data.py --complete

Files
protocol.json: replay design, source hashes and fixed rules.
runs/*/episodes.csv and ledger.csv.gz: per-condition episode summaries and transaction ledgers.
results.json and episodes.csv: complete summaries, paired differences, Q values and the original EMA reference.
refinements.json: coarse/fine checks at the primary price, not a rigorous error bound for every price.
verification.json: independent checks of prices, signals, transactions, fees, budget, terminal enumeration and statistics.
long_planner.py: solver; experiment.py: historical replay; verify.py: independent audit.
