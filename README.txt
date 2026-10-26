TIME HAS A PRICE
A personal side experiment: when is waiting worth its cost?

START THE DEMO (Python 3, no install needed)
  On macOS, double-click RUN_DEMO.command.
  Or open this folder in VS Code and run in its terminal:
    python3 serve.py --open
  The browser opens automatically on a free local port. Keep the terminal open.
  VS Code also provides Tasks: Run Task > Start Time Has a Price.
  The demo is a self-contained HTML export. It needs no live AI, trading API,
  login, or internet connection. The localhost URL works only on your machine.

BROWSER DEMO (no slides needed)
Start with Reset baseline. Show the question and comparison table.
1. Compare four strategies using identical fees and time rules.
2. Click Show a changed decision to compare the 30-step and 2-step policies.
3. Explain why the historical interval includes zero: improvement is unconfirmed.
Open Methods and checks only for questions about action values or uncertainty.
Data and earlier studies contains historical prices and archived experiments.
Script: submission_docs/BROWSER_DEMO_SCRIPT_EN.txt.
The mathematical experiment has already run; serving the report does not rerun
the backtest. Model source and verification commands are below.
The presentation folder contains optional backup slides; it is not required.

THE QUESTION
Market charts measure money; waiting also consumes a scarce time budget.
We ask whether to continue, buy extra decision time, or stop. This is an
optimal-stopping/data project. Market prices are real; time units, prices,
utility weights and exhaustion penalties are explicit modeling assumptions.

WHAT IS IMPLEMENTED
Seven controls, full finite 30-step Bellman planning, exact reachable resource
and portfolio nodes, and one-dimensional wealth interpolation. Frozen BTC
2022-2025 model; BTC/ETH Jan-Sep 2026 historical replay; 61,344 accounts across
24 fixed settings, 213 overlapping starts per asset, six controllers.
2026 was already inspected: this is exploratory historical re-evaluation.

RESULTS TO READ HONESTLY
At price8 and alpha.1, the 30-step planner exits immediately. Six of 192 fixed
initial decisions change with longer planning, without demonstrated stable
historical advantage. Paid mid-trading purchases: zero. Free purchases:
65 events in 51 overlapping accounts. Terminal purchases are reported
separately. Higher utility can mean preserving time, not earning more cash.

REPRODUCE THE MATHEMATICS (tested with Python3.11 and NumPy2.4.6)
  python3 -m pip install -r requirements.txt
  python3 -m unittest discover -s long_horizon_v3 -p 'test_long_planner.py' -v
  python3 long_horizon_v3/verify.py
The complete saved replay is included, so verification does not require a
long backtest. To rebuild numerical results from the included real prices:
  python3 long_horizon_v3/experiment.py --workers 2
  python3 long_horizon_v3/refinements.py --workers 3
  python3 long_horizon_v3/purchase_cases.py
This may take several minutes. It does not fetch future data or tune a new
profitable strategy. Old frozen source checksums are kept in protocol.json.

REPOSITORY MAP
  demo/index.html         portable interactive presentation
  presentation_source/   authored React and CSS used for that presentation
  presentation/          five-slide deck, PDF display backup and English script
  long_horizon_v3/       model, experiments, tests, full saved replay and audit
  time_budget_policy_v2/ frozen two-step engine/kernel and real2026 prices
  real_market_four_stage/data/ real2022-2025 training price CSVs
  moving_average_life_compare/ original EMA implementation
  submission_docs/       draft project description, pitch, Q&A and rules review

The presentation was authored/exported using the Codex Data report renderer;
the two presentation source files require that renderer to rebuild. The saved
demo runs standalone. The numerical model and its audit require only NumPy.

CREDITS / AI DISCLOSURE
Binance public market data: https://github.com/binance/binance-public-data
15/30EMA signal reference: QuantConnect LEAN MovingAverageCrossAlgorithm.
Exact reference URL is in submission_docs/PROJECT_DESCRIPTION.txt.
We used AI assistance for coding, analysis and interface production. The demo
itself does not use a live language-model service.

PROJECT STATUS
This is a personal side experiment. The source, public price data, saved
results and self-contained demo remain available to explore and reproduce.
The presentation and submission_docs folders retain earlier draft materials;
they do not establish an event submission.
