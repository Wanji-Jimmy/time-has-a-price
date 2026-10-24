import React, { useMemo, useState } from "react";
import {
  Button, ChartTooltip, DataComponent, DataTable, Dropdown, EvidenceChart,
  ReportSection, RichNarrative, Slider, useDataApp,
} from "../../data-app-public.jsx";

const ACTIONS = ["Exit and liquidate", "Continue in cash", "Continue at 50%", "Continue at 100%", "Buy time + cash", "Buy time + 50%", "Buy time + 100%"];
const VARIANTS = { original: "Original EMA", life_only: "Budget-limited EMA", life_buy: "EMA with mechanical purchases" };
const ALPHAS = [0.02, 0.05, 0.1, 0.5];
const fixed = (value, digits = 4) => Number.isFinite(value) ? (Math.abs(value) < 1e-12 ? 0 : value).toFixed(digits) : "—";
const signed = (value, digits = 4) => Number.isFinite(value) ? `${value > 1e-12 ? "+" : ""}${fixed(value, digits)}` : "—";
const numericColumn = (field, label, digits = 4) => ({ field, label, renderCell: value => fixed(value, digits) });
const pctColumn = (field, label) => ({ field, label, renderCell: value => Number.isFinite(value) ? `${fixed(value * 100, 1)}%` : "—" });
const LONG_CONTROLLERS = { dp30: "30-step planner", dp2: "2-step planner", dp30_no_mid_buy: "30-step, no mid-trade purchases", ema_time: "Budget-limited EMA", ema_mechanical: "EMA with mechanical purchases", retire_now: "Exit immediately" };
const MARKET_LABELS = { 0: "5-day down / low volatility", 1: "5-day down / high volatility", 2: "5-day up / low volatility", 3: "5-day up / high volatility" };
const longAction = index => ACTIONS[Number(index)] ?? "—";
const bestQGap = (row, prefix, first, last, reference = 0) => {
  if (!row) return NaN;
  const key = i => prefix === "dp30" ? `q${i}` : `q2_${i}`;
  const values = Array.from({ length: last - first + 1 }, (_, i) => row[key(first + i)]).filter(Number.isFinite);
  return values.length && Number.isFinite(row[key(reference)]) ? Math.max(...values) - row[key(reference)] : NaN;
};

function Narrative({ id, queryId, rows, queryIds, sourceRowsByQuery, children, className }) {
  return <ReportSection id={id} title="" queryId={queryId} sourceRows={rows} queryIds={queryIds} sourceRowsByQuery={sourceRowsByQuery} showHeading={false} className={className}>
    <RichNarrative id={`${id}:body`} value={children} label="Edit report text" />
  </ReportSection>;
}

function LongHorizonStudy({ comparisons, regions, contrasts, originals, numerics, purchases, valuations, purchaseExample, chartOverrides }) {
  const [asset, setAsset] = useState("BTCUSDT");
  const [packCost, setPackCost] = useState(8);
  const [alpha, setAlpha] = useState(0.1);
  const [initialLife, setInitialLife] = useState(90);
  const [market, setMarket] = useState(1);
  const [allocation, setAllocation] = useState("cash");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const match = row => row.symbol === asset && row.pack_cost === packCost && row.alpha === alpha && row.initial_life === initialLife;
  const selected = comparisons.filter(match);
  const selectedContrasts = contrasts.filter(match);
  const shortContrast = selectedContrasts.find(row => row.baseline === "dp2");
  const immediateExit = selected.find(row => row.controller === "retire_now");
  const selectedOriginal = originals.filter(row => row.symbol === asset);
  const originalDisplay = selectedOriginal.map(row => ({ ...row, trading_pnl: row.final_cash - 100 }));
  const regionScope = regions.filter(row => row.pack_cost === packCost && row.alpha === alpha && row.market_state === market && row.cash === (allocation === "cash" ? 100 : 50)).sort((a, b) => a.initial_life - b.initial_life);
  const region = regionScope.find(row => row.initial_life === initialLife);
  const regionCurve = regionScope.map(row => ({ initial_life: row.initial_life, fullContinue: bestQGap(row, "dp30", 1, 6), shortContinue: bestQGap(row, "dp2", 1, 6) }));
  const qRows = region ? ACTIONS.map((action, i) => ({ action, fullQ: region[`q${i}`], shortQ: region[`q2_${i}`], fullGap: Number.isFinite(region[`q${i}`]) ? region[`q${i}`] - region.q0 : null, selected: Number(region.action) === i ? "30-step choice" : "—" })) : [];
  const contrastCurve = contrasts.filter(row => row.symbol === asset && row.pack_cost === packCost && row.alpha === alpha && row.baseline === "dp2").sort((a, b) => a.initial_life - b.initial_life);
  const zeroPlanningEffect = contrastCurve.length > 0 && contrastCurve.every(row => [row.effect, row.low30, row.high30].every(value => Number.isFinite(value) && Math.abs(value) <= 1e-12));
  const primary = comparisons.find(row => row.pack_cost === 8 && row.alpha === 0.1 && row.controller === "dp30" && row.symbol === "BTCUSDT" && row.initial_life === 90);
  const primaryEMA = comparisons.find(row => row.pack_cost === 8 && row.alpha === 0.1 && row.controller === "ema_time" && row.symbol === "BTCUSDT" && row.initial_life === 90);
  const tableRows = selected.map(row => ({ ...row, mid_spend: row.pack_cost * row.mid_packs, terminal_spend: row.pack_cost * row.terminal_packs, controller_label: LONG_CONTROLLERS[row.controller] ?? row.controller }));
  const coreRows = tableRows.filter(row => ["dp30", "dp2", "ema_time", "ema_mechanical"].includes(row.controller));
  const contrastRows = selectedContrasts.map(row => ({ ...row, baseline_label: LONG_CONTROLLERS[row.baseline] ?? row.baseline, interval30: `[${signed(row.low30, 6)}, ${signed(row.high30, 6)}]`, interval60: `[${signed(row.low60, 6)}, ${signed(row.high60, 6)}]` }));
  const firstPurchases = [...(purchases ?? [])].sort((a, b) => a.date.localeCompare(b.date) || a.symbol.localeCompare(b.symbol) || a.start_index - b.start_index || a.step - b.step).slice(0, 10).map(row => ({ ...row, date_label: row.date.slice(0, 10), action_label: longAction(row.executed_action) }));
  const paidPurchases = (purchases ?? []).filter(row => row.pack_cost > 0).length;
  const freePurchases = (purchases ?? []).filter(row => row.pack_cost === 0).length;
  const freeAccounts = new Set((purchases ?? []).filter(row => row.pack_cost === 0).map(row => [row.symbol, row.pack_cost, row.alpha, row.initial_life, row.start_index].join("|"))).size;
  const firstEvent = purchaseExample?.event;
  const firstEventRows = firstEvent ? (purchases ?? []).filter(row => row.symbol === firstEvent.symbol && row.date === firstEvent.date && row.start_index === firstEvent.start_index && row.step === firstEvent.step && row.alpha === firstEvent.alpha && row.initial_life === firstEvent.initial_life && row.pack_cost === firstEvent.pack_cost) : [];
  const valuationDisplay = (valuations ?? []).map(row => ({ ...row, action_label: longAction(row.action_index), selected_label: [row.chosen30 ? "30 steps" : "", row.chosen2 ? "2 steps" : ""].filter(Boolean).join(", ") || "—" }));
  const showExample = () => { setAsset("BTCUSDT"); setPackCost(8); setAlpha(0.02); setInitialLife(45); setMarket(1); setAllocation("cash"); };
  const resetPrimary = () => { setAsset("BTCUSDT"); setPackCost(8); setAlpha(0.1); setInitialLife(90); setMarket(1); setAllocation("cash"); };

  return <>
    <Narrative id="long-summary" queryId="long_comparison" rows={comparisons}
      queryIds={["long_comparison", "long_regions", "long_contrasts", ...(purchases ? ["long_purchase_cases"] : [])]}
      sourceRowsByQuery={{ long_comparison: comparisons, long_regions: regions, long_contrasts: contrasts, ...(purchases ? { long_purchase_cases: purchases } : {}) }} className="tv-summary">
      {`## Longer planning changes decisions; a stable advantage remains unproven

- **BTC baseline:** 30-step planning retains ${fixed(primary?.final_cash, 2)} cash and ${fixed(primary?.remaining_life, 0)} budget seconds. Budget-limited EMA retains ${fixed(primaryEMA?.final_cash, 2)} cash and ${fixed(primaryEMA?.remaining_life, 0)} seconds. Baseline: α=0.1, initial L=90, pack price 8.
- **Buying time has limits:** ${paidPurchases} paid purchases during trading at prices 8 and 2; ${freeAccounts} replay accounts extend trading when time is free. Terminal exchanges are counted separately.

Prices are real Binance observations. Budgets, prices for time and preferences are model assumptions. January–September 2026 has already been examined; this is an exploratory historical replay.`}
    </Narrative>

    <section className="report-section" aria-label="Compare finite-horizon planning with EMA">
      <RichNarrative id="long-comparison:intro" value="## 1. Same cash. Limited time. Different decisions.\n\nA market model fitted to BTC in 2022–2025 is replayed on BTC and ETH in 2026. Each account starts with 100 cash and has at most 30 market days, costing 5 budget seconds per day. All four strategies share fees, budgets and settlement rules. EMA uses only 15/30-day exponential averages available before the next open." label="Edit strategy comparison" />
      <div className="tv-controls" aria-label="Historical replay settings" data-reviewed-rows>
        <Dropdown label="Market data" showLabel value={asset} choices={["BTCUSDT", "ETHUSDT"]} onChange={setAsset} />
        <Dropdown label="Initial budget L" showLabel value={initialLife} choices={[20, 45, 90, 120]} onChange={value => setInitialLife(Number(value))} />
        <Button onClick={() => setAdvancedOpen(value => !value)}>{advancedOpen ? "Hide parameters" : "Price and preferences"}</Button>
        <Button onClick={resetPrimary}>Reset baseline</Button>
      </div>
      {advancedOpen && <div className="tv-controls tv-secondary-controls" aria-label="Model parameters" data-reviewed-rows>
        <Dropdown label="Price per 15 seconds" showLabel value={packCost} choices={[8, 2, 0]} choiceLabels={{ 8: "8 cash (baseline)", 2: "2 cash (sensitivity)", 0: "Free (boundary case)" }} onChange={value => setPackCost(Number(value))} />
        <Dropdown label="Time preference α" showLabel value={alpha} choices={[0.1, 0.02]} onChange={value => setAlpha(Number(value))} />
        <Dropdown label="Market state" showLabel value={market} choices={[0, 1, 2, 3]} choiceLabels={MARKET_LABELS} onChange={value => setMarket(Number(value))} />
        <Dropdown label="Initial portfolio" showLabel value={allocation} choices={["cash", "half"]} choiceLabels={{ cash: "100 cash / 0 holdings", half: "50 cash / 50 holdings" }} onChange={setAllocation} />
      </div>}
      <DataComponent id="long-comparison-table" queryId="long_comparison" title={`${asset} · α=${alpha} · Initial L=${initialLife} · pack price ${packCost} cash`} kind="table" displayRows={coreRows} sourceRows={selected.filter(row => ["dp30", "dp2", "ema_time", "ema_mechanical"].includes(row.controller))}>
        <DataTable rows={coreRows} searchable={false} caption="Means across 213 overlapping starts; utility values both cash and time" columns={[
          { field: "controller_label", label: "Strategy" }, numericColumn("final_cash", "Final cash", 2), numericColumn("remaining_life", "Budget seconds left", 2), numericColumn("utility", "Realized utility", 6), numericColumn("mid_spend", "Mid-trade time cost", 2),
        ]} />
        <RichNarrative id="long-money-accounting:note" value="Mechanical EMA attempts a time purchase at L≤10 when its signal is bullish. Budget-limited EMA exits before exhaustion. Higher utility can reflect preserved time rather than higher cash." label="Edit comparison note" />
      </DataComponent>
      {immediateExit && immediateExit.terminal_packs > 1e-10 && <Narrative id="long-terminal-explanation" queryId="long_comparison" rows={[immediateExit]}>
      {`**Time can also be consumed at exit.** The immediate-exit benchmark exchanges ${fixed(immediateExit.terminal_packs, 2)} packs, leaving ${fixed(immediateExit.final_cash, 2)} cash and ${fixed(immediateExit.remaining_life, 2)} budget seconds. Trading has ended; this exchange represents terminal preference.`}
    </Narrative>}
      <DataComponent id="long-original-ema" queryId="long_original_ema" title="Unrestricted EMA wealth reference" kind="custom" displayRows={originalDisplay} sourceRows={selectedOriginal}>
        <RichNarrative id="long-original-ema:note" value={`Unrestricted EMA ends with **${fixed(selectedOriginal[0]?.final_cash, 2)} cash** on the same ${asset} starting dates. It can trade for all 30 days without time fees, so this is a wealth reference; its utility is not ranked against constrained strategies.`} label="Edit unrestricted EMA reference" />
      </DataComponent>
    </section>

    <section className="report-section" aria-label="When to buy time or exit">
      <RichNarrative id="long-regions:intro" value="## 2. When does more time change the decision?\n\nHold wealth and the market state fixed, then compare two-step and 30-step planning. Q is expected utility within the model. Continue when its Q exceeds exit; buying time must also beat every action without a purchase." label="Edit decision boundary introduction" />
      <div className="tv-controls" aria-label="Compare planning horizons">
        <Button onClick={showExample}>Show a changed decision</Button>
        <span className="tv-state-label" data-reviewed-rows>{MARKET_LABELS[market]} · {allocation === "cash" ? "100 cash / 0 holdings" : "50 cash / 50 holdings"} · α={alpha} · pack price {packCost}</span>
      </div>
      <DataComponent id="long-decision-now" queryId="long_regions" title={`Initial L=${initialLife}: 30 steps vs 2 steps`} kind="custom" displayRows={region ? [region] : []} sourceRows={regionScope}>
        {region && <div className="tv-decision-pair" data-reviewed-rows>
          <div><span>Full 30-step planner</span><strong>{longAction(region.action)}</strong><small>Continue − exit: {signed(bestQGap(region, "dp30", 1, 6), 6)}</small></div>
          <div><span>2-step planner</span><strong>{longAction(region.action2)}</strong><small>Continue − exit: {signed(bestQGap(region, "dp2", 1, 6), 6)}</small></div>
        </div>}
      </DataComponent>
      <EvidenceChart id="long-decision-boundary" queryId="long_regions" title="Does more time make continuing worthwhile?"
        rows={regionCurve} sourceRows={regionScope} height={300}
        spec={{ type: "line", x: "initial_life", y: "fullContinue", fields: ["fullContinue", "shortContinue"], startAtZero: false, valueDecimals: 6, stackable: false, xLabel: "Initial budget (seconds)", yLabel: "Best continue Q − exit Q", legend: { labels: { fullContinue: "Full 30 steps", shortContinue: "2 steps" } } }}
        chartOptions={{ tooltipContent: <ChartTooltip vertical xField={chartOverrides?.["long-decision-boundary"]?.x ?? "initial_life"} formatValue={value => signed(Number(value), 6)} formatLabel={item => chartOverrides?.["long-decision-boundary"]?.legend?.labels?.[item.dataKey] ?? { fullContinue: "Full 30 steps", shortContinue: "2 steps" }[item.dataKey] ?? item.name ?? item.dataKey} /> }}
        description="Four evaluated budget values; intermediate values are untested. The boundary uses the frozen BTC model for both replay assets." />
      {shortContrast && <Narrative id="long-prediction-vs-replay" queryId="long_contrasts" rows={[shortContrast]}>
      {`**A different action does not establish better performance.** At ${asset}, α=${alpha}, L=${initialLife}, price ${packCost}, the realized utility difference (30 steps minus 2) is **${signed(shortContrast.effect, 6)}**. The 30-day block 95% interval is **[${signed(shortContrast.low30, 6)}, ${signed(shortContrast.high30, 6)}]**. ${shortContrast.low30 <= 1e-12 && shortContrast.high30 >= -1e-12 ? "It includes zero; improvement is unconfirmed." : "This interval applies only to this setting; it does not establish a stable advantage."} The 213 starting dates overlap. Model Q is a forecast, not a realized return.`}
    </Narrative>}
    </section>

    <div className="tv-disclosure-control">
      <Button onClick={() => setDetailsOpen(value => !value)}>{detailsOpen ? "Hide methods and checks" : "Methods and checks"}</Button>
      <span>Action values, six strategies, purchases, uncertainty and precision</span>
    </div>
    {detailsOpen && <section className="report-section tv-research-details" aria-label="Methods and verification">
      <DataComponent id="long-action-q" queryId="long_regions" title="Q values for all seven actions" kind="table" displayRows={qRows} sourceRows={region ? [region] : []}>
        <DataTable rows={qRows} searchable={false} caption="Q is expected utility; differences below numerical error are inconclusive" columns={[{ field: "action", label: "Action" }, numericColumn("fullQ", "30-step Q", 6), numericColumn("fullGap", "30-step ΔQ vs exit", 6), numericColumn("shortQ", "2-step Q", 6), { field: "selected", label: "Choice" }]} />
      </DataComponent>
      <DataComponent id="long-money-accounting" queryId="long_comparison" title="Six strategies and cash accounting" kind="table" displayRows={tableRows} sourceRows={selected}>
        <DataTable rows={tableRows} searchable={false} caption="Final cash = 100 + net trading P&L − mid-trade purchases − terminal exchanges" columns={[
          { field: "controller_label", label: "Strategy" }, numericColumn("trading_pnl", "Net trading P&L", 3), numericColumn("mid_spend", "Mid-trade time cost", 3), numericColumn("terminal_spend", "Terminal time cost", 3), numericColumn("final_cash", "Final cash", 3), numericColumn("remaining_life", "Budget seconds left", 2), numericColumn("utility", "Utility", 6), numericColumn("exit_step", "Market days elapsed", 2), pctColumn("death_rate", "Exhaustion rate"),
        ]} />
        <RichNarrative id="long-money-accounting:detail" value="The no-mid-trade-purchase controller still permits terminal exchanges, isolating the role of longer trading. Net trading P&L includes fees and slippage. Terminal purchases do not extend trading." label="Edit cash accounting" />
      </DataComponent>
      {purchases && <DataComponent id="long-actual-purchases" queryId="long_purchase_cases" title="Time purchases during historical replay" kind="table" displayRows={firstPurchases} sourceRows={purchases}>
        <RichNarrative id="long-actual-purchases:reading" value={`${paidPurchases} paid and ${freePurchases} free purchases were recorded during trading. The first 10 below are ordered by date, asset, starting date and step. Accounts may overlap; terminal exchanges are excluded.`} label="Edit purchase events note" />
        <DataTable rows={firstPurchases} searchable={false} caption="First 10 mid-trade purchases" columns={[
          { field: "date_label", label: "Date" }, { field: "symbol", label: "Market data" }, numericColumn("pack_cost", "Price", 0), numericColumn("alpha", "α", 2), numericColumn("initial_life", "Initial L", 0), { field: "step", label: "Market days elapsed" }, numericColumn("life_before", "Budget before purchase", 0), numericColumn("cash_before", "Cash", 2), numericColumn("holding_before", "Holdings value", 2), { field: "action_label", label: "Executed action" },
        ]} />
      </DataComponent>}
      {firstEvent && valuations?.length > 0 && <>
        <Narrative id="long-actual-purchase-context" queryId="long_purchase_cases" rows={firstEventRows} queryIds={["long_purchase_cases", "long_purchase_valuation"]} sourceRowsByQuery={{ long_purchase_cases: firstEventRows, long_purchase_valuation: valuations }}>
      {`**First free purchase:** ${firstEvent.symbol}, ${firstEvent.date.slice(0, 10)}, after ${firstEvent.step} market days, with ${fixed(firstEvent.life_before, 0)} budget seconds left. One more day would exhaust the budget without a purchase. Both planners buy, so this is not unique to longer planning and does not establish willingness to pay. The current open and close were unknown when choosing.`}
    </Narrative>
        <DataComponent id="long-actual-purchase-q" queryId="long_purchase_valuation" title="Seven Q values before the first purchase" kind="table" displayRows={valuationDisplay} sourceRows={valuations}>
          <DataTable rows={valuationDisplay} searchable={false} caption="Frozen-model Q at price zero" columns={[{ field: "action_label", label: "Action" }, numericColumn("q30", "30-step Q", 6), numericColumn("gap30", "30-step ΔQ vs exit", 6), numericColumn("q2", "2-step Q", 6), { field: "selected_label", label: "Chosen by" }]} />
        </DataComponent>
      </>}
      <DataComponent id="long-contrasts-table" queryId="long_contrasts" title="Paired utility differences against five controls" kind="table" displayRows={contrastRows} sourceRows={selectedContrasts}>
        <DataTable rows={contrastRows} searchable={false} caption="30-step planner minus control, at the selected settings" columns={[{ field: "baseline_label", label: "Control" }, numericColumn("effect", "Mean utility difference", 6), { field: "interval30", label: "95% interval (30-day blocks)" }, { field: "interval60", label: "95% interval (60-day blocks)" }, numericColumn("nonoverlap_effect", "Non-overlapping mean difference", 6)]} />
        <RichNarrative id="long-contrasts-table:scope" value="Intervals use paired differences at the same starting dates. The 213 dates overlap; checks also use 60-day blocks and starts 31 days apart. No multiple-comparison adjustment was applied." label="Edit uncertainty note" />
      </DataComponent>
      {zeroPlanningEffect ? <DataComponent id="long-planning-effect" queryId="long_contrasts" title="Both horizons have equal realized utility at all four budgets" kind="custom" displayRows={contrastCurve} sourceRows={contrastCurve}>
        <DataTable rows={contrastCurve} searchable={false} caption="Utility difference at four initial budgets" columns={[{ field: "initial_life", label: "Initial budget L" }, numericColumn("effect", "Utility difference", 6), numericColumn("low30", "95% lower bound", 6), numericColumn("high30", "95% upper bound", 6)]} />
      </DataComponent> : <EvidenceChart id="long-planning-effect" queryId="long_contrasts" title={`${asset} · Price ${packCost} · α=${alpha}: 30-step minus 2-step utility`}
        rows={contrastCurve} sourceRows={contrastCurve} height={300}
        spec={{ type: "line", x: "initial_life", y: "effect", fields: ["effect", "low30", "high30"], startAtZero: false, valueDecimals: 6, stackable: false, xLabel: "Initial budget", yLabel: "Realized utility difference", legend: { labels: { effect: "Mean utility difference", low30: "95% lower bound", high30: "95% upper bound" } } }}
        chartOptions={{ tooltipContent: <ChartTooltip vertical xField={chartOverrides?.["long-planning-effect"]?.x ?? "initial_life"} formatValue={value => signed(Number(value), 6)} formatLabel={item => chartOverrides?.["long-planning-effect"]?.legend?.labels?.[item.dataKey] ?? { effect: "Mean utility difference", low30: "95% lower bound", high30: "95% upper bound" }[item.dataKey] ?? item.name ?? item.dataKey} /> }}
        description="Four evaluated budgets with 30-day block intervals." />}
      <Narrative id="long-method" queryId="long_comparison" rows={selected} queryIds={["long_comparison", "long_regions"]} sourceRowsByQuery={{ long_comparison: selected, long_regions: regionScope }} className="tv-math">
      {"### Full-horizon Bellman planning\n\nThe state contains the market regime, cash, holdings, time, packs used and remaining market steps. At each step, compare exit, cash/half/full exposure, and the same three allocations after buying time:\n\n`V_R(s) = max_a E[V_(R−1)(T(s,a,ξ))]`\n\nTerminal cash B and remaining budget L determine utility:\n\n`U(B,L) = α log(1+L/90) + (1−α) log(1+B/100)`\n\nα is a chosen preference. The identity `L = L₀ − 5(30−R) + 15k` and finitely many reachable allocations reduce the state space. Wealth still uses grid interpolation; the full horizon does not make the continuous-state solution exact.\n\nActions use information available before the next opening gap. Early exit liquidates at the next open; the 30-day boundary uses the final close. L≤0 triggers exhaustion before settlement, setting assets and utility to zero. Terminal settlement may exchange cash for time within the six-pack limit. Both planners and constrained EMA share costs and settlement.\n\nThe BTC 2022–2025 transition model stays frozen. ETH is evaluated without retraining. The 2026 data has already been examined. Budget seconds are hypothetical, and these results do not establish future profitability."}
    </Narrative>
      {numerics.length > 0 && <DataComponent id="long-numerics" queryId="long_numerics" title="Numerical checks at baseline price 8" kind="table" displayRows={numerics} sourceRows={numerics}>
        <DataTable rows={numerics} searchable={false} caption="Wealth grid refined from step 1 to 0.5" columns={[{ field: "scope", label: "Scope" }, { field: "n", label: "States" }, { field: "changed_actions", label: "Changed actions" }, numericColumn("max_absolute_q_change", "Max absolute Q change", 8)]} />
        <RichNarrative id="long-numerics:note" value="This checks the stated states, not a global error bound. Prices 0 and 2 have not had the same fine-grid check; decisions with very small Q gaps may be unstable." label="Edit numerical scope" />
      </DataComponent>}
    </section>}
  </>;
}

export function ReportContent() {
  const { snapshot, reviewedPeriodRows, appTitle, setAppTitle, canEdit, mode, chartOverrides } = useDataApp();
  const coverage = reviewedPeriodRows("source_coverage");
  const prices = reviewedPeriodRows("historical_prices");
  const curves = reviewedPeriodRows("time_curves");
  const ema = reviewedPeriodRows("ema_comparison");
  const sensitivity = reviewedPeriodRows("price_sensitivity");
  const validations = reviewedPeriodRows("historical_validation");
  const shadowPrices = reviewedPeriodRows("shadow_prices");
  const hasLong = Boolean(snapshot?.queries?.long_comparison?.rows?.length);
  const longComparisons = hasLong ? reviewedPeriodRows("long_comparison") : [];
  const longRegions = hasLong ? reviewedPeriodRows("long_regions") : [];
  const longContrasts = hasLong ? reviewedPeriodRows("long_contrasts") : [];
  const longOriginals = hasLong ? reviewedPeriodRows("long_original_ema") : [];
  const longNumerics = snapshot?.queries?.long_numerics ? reviewedPeriodRows("long_numerics") : [];
  const longPurchases = snapshot?.queries?.long_purchase_cases ? reviewedPeriodRows("long_purchase_cases") : null;
  const longValuations = snapshot?.queries?.long_purchase_valuation ? reviewedPeriodRows("long_purchase_valuation") : null;
  const [archiveOpen, setArchiveOpen] = useState(false);
  const [priceAsset, setPriceAsset] = useState("BTCUSDT");
  const [pricePeriod, setPricePeriod] = useState("2026");
  const [alpha, setAlpha] = useState(0.1);
  const [market, setMarket] = useState(1);
  const [allocation, setAllocation] = useState("cash");
  const [life, setLife] = useState(20);
  const [year, setYear] = useState("2025");
  const [validationAsset, setValidationAsset] = useState("BTCUSDT");
  const [validationAlpha, setValidationAlpha] = useState(0.02);
  const [validationLife, setValidationLife] = useState(20);
  const [shadowAlpha, setShadowAlpha] = useState(0.1);
  const [shadowMarket, setShadowMarket] = useState(1);
  const [shadowAllocation, setShadowAllocation] = useState("cash");
  const cash = allocation === "cash" ? 100 : 50;
  const selectedPrices = useMemo(() => prices.filter(r => r.symbol === priceAsset && (pricePeriod === "all" || r.date.startsWith("2026"))), [prices, priceAsset, pricePeriod]);
  const conditionRows = curves.filter(r => r.alpha === alpha && r.market_state === market && r.cash === cash);
  const current = conditionRows.find(r => r.life === life);
  const curveRows = conditionRows.map(r => ({ life: r.life, awareGap: r.aware_best_nonretire_gap, maskedGap: r.masked_best_nonretire_gap }));
  const qRows = current ? ACTIONS.map((action, index) => ({
    index, action, awareQ: current[`aware_q${index}`], maskedQ: current[`masked_q${index}`],
    awareGap: current[`aware_q${index}`] - current.aware_q0,
    maskedGap: current[`masked_q${index}`] - current.masked_q0,
    selected: [current.aware_action === index ? "Time-aware" : "", current.masked_action === index ? "Time-masked" : ""].filter(Boolean).join(", ") || "—",
  })) : [];
  const yearRows = ema.filter(r => r.year === year);
  const decomposition = yearRows.map(r => ({ variant: VARIANTS[r.variant], initial: 100, trading: r.mean_trading_pnl, timeSpend: r.mean_life_spend, final: r.mean_wealth }));
  const validationScope = validations.filter(r => r.symbol === validationAsset && r.alpha === validationAlpha);
  const validation = validationScope.find(r => r.life === validationLife);
  const allValidationRows = validations.map(r => ({ ...r, interval: `[${signed(r.low, 6)}, ${signed(r.high, 6)}]` }));
  const shadowRows = shadowPrices.filter(r => r.alpha === shadowAlpha && r.market_state === shadowMarket && r.cash === (shadowAllocation === "cash" ? 100 : 50));
  const mechanism = () => { setAlpha(0.02); setMarket(1); setAllocation("cash"); setLife(20); };

  return <article className="report-content time-value-report" aria-label="A mathematical study of wealth and finite decision time">
    <header className="report-hero">
      <div className="tv-kicker">TIME × DATA × DECISION THEORY</div>
      <h1 data-data-app-title contentEditable={canEdit && mode === "edit"} suppressContentEditableWarning
        onBlur={canEdit && mode === "edit" ? e => setAppTitle(e.currentTarget.textContent.trim() || appTitle) : undefined}
        onKeyDown={canEdit && mode === "edit" ? e => { if (e.key === "Enter") { e.preventDefault(); e.currentTarget.blur(); } } : undefined}>{appTitle}</h1>
      <Narrative id="project-question" queryId="source_coverage" rows={coverage}>
      {"**With the same wealth, why can remaining time change a rational decision?**\n\nA mathematical decision model tested on real market data. Cash, holdings and remaining time enter one state, so the model jointly chooses exposure, when to exit and whether extra decision time is worth buying."}
    </Narrative>
    </header>

    {hasLong ? <LongHorizonStudy comparisons={longComparisons} regions={longRegions} contrasts={longContrasts} originals={longOriginals} numerics={longNumerics} purchases={longPurchases} valuations={longValuations} purchaseExample={snapshot?.longStudy?.purchaseExample} chartOverrides={chartOverrides} summary={snapshot?.longStudy?.summary} /> : <>
    <Narrative id="project-summary" queryId="source_coverage" rows={coverage} queryIds={["source_coverage", "time_curves", "ema_comparison", "historical_validation"]} sourceRowsByQuery={{ source_coverage: coverage, time_curves: curves, ema_comparison: ema.filter(r => r.year === "2025"), historical_validation: validations }} className="tv-summary">
      {"## Findings\n\n- **Time can change decisions.** The original α=0.1 setting favors immediate exit. In one α=0.02 condition, a short budget favors exit and a longer budget favors continuation.\n- **Buying time need not buy profit.** Mechanical EMA in 2025 retains 83.22 cash, with +0.40 net trading P&L and 17.18 spent on time.\n- **A stable advantage is unproven.** All 32 examined historical conditions remain available with their uncertainty intervals."}
    </Narrative>
    </>}

    <div className="tv-disclosure-control"><Button onClick={() => setArchiveOpen(value => !value)}>{archiveOpen ? "Hide data and earlier studies" : "Data and earlier studies"}</Button><span>Prices, data coverage and two-step studies A–E</span></div>
    {archiveOpen && <>
    <section className="report-section" aria-label="Historical prices and coverage">
      <RichNarrative id="prices:intro" value="## The underlying data\n\nBTC and ETH use official Binance spot daily bars. The chart shows observed daily closes. Time budgets and time purchases are additional model rules. These controls filter the price chart; they do not retrain the model." label="Edit market data note" />
      <div className="tv-controls" aria-label="Price chart filters" data-reviewed-rows>
          <Dropdown label="Asset" showLabel value={priceAsset} choices={["BTCUSDT", "ETHUSDT"]} onChange={setPriceAsset} />
          <Dropdown label="Range" showLabel value={pricePeriod} choices={["2026", "all"]} choiceLabels={{ "2026": "Jan–Sep 2026", all: "All available history" }} onChange={setPricePeriod} />
      </div>
      <EvidenceChart id="price-chart" queryId="historical_prices" title={`${priceAsset} daily close · USDT`}
        spec={{ type: "line", x: "date", y: "close", startAtZero: false, valueDecimals: 2, stackable: false }}
        rows={selectedPrices} sourceRows={selectedPrices} height={330}
        description="Observed daily closes in USDT; UTC dates." />
      <DataComponent id="source-coverage" title="Available data" queryId="source_coverage" kind="table" displayRows={coverage} sourceRows={coverage}>
        <DataTable rows={coverage} searchable={false} columns={[{ field: "period", label: "Period" }, { field: "asset", label: "Asset" }, { field: "rows", label: "Daily bars" }, { field: "role", label: "Role" }]} caption="Price data coverage" />
      </DataComponent>
    </section>

    {hasLong && <Narrative id="prior-research-context" queryId="time_curves" rows={curves}>
      {"## Earlier two-step studies\n\nThe archive retains two-step diagnostics, 2022–2025 EMA experiments and one-off time valuations. These use earlier settings, starting-date counts and controls. The current 30-step replay is reported above."}
    </Narrative>}

    <section className="report-section" aria-label="Time counterfactual at a fixed market state">
      <RichNarrative id="model:intro" value="## Study A: Change only the remaining time\n\nThe state contains the market regime, cash, holdings, time L, packs used and remaining steps. Other inputs are fixed while L varies. Each point is a saved two-step Bellman calculation. The baseline is α=0.1; the α=0.02 example shows a conditional change in action." label="Edit model introduction" />
      <DataComponent id="model-control" queryId="time_curves" title="Counterfactual: total wealth fixed at 100" kind="custom"
        displayRows={current ? [current] : []} sourceRows={conditionRows}
        description="The curve uses 21 evaluated L values. Time-masked scoring resets scored L to 90 while keeping the actual time constraint.">
        <div className="tv-controls" data-reviewed-rows>
          <Dropdown label="Time preference α" showLabel value={alpha} choices={ALPHAS} onChange={v => setAlpha(Number(v))} />
          <Dropdown label="Fixed market state" showLabel value={market} choices={[0, 1, 2, 3]} choiceLabels={{ 0: "5-day down / low volatility", 1: "5-day down / high volatility", 2: "5-day up / low volatility", 3: "5-day up / high volatility" }} onChange={v => setMarket(Number(v))} />
          <Dropdown label="Fixed portfolio" showLabel value={allocation} choices={["cash", "half"]} choiceLabels={{ cash: "100 cash / 0 holdings", half: "50 cash / 50 holdings" }} onChange={setAllocation} />
          <Button onClick={mechanism}>Show the α = 0.02 example</Button>
        </div>
        <div className="tv-life-control"><Slider label="Remaining budget L" min={20} max={120} step={5} value={life} onChange={v => setLife(Number(v))} formatValue={v => `${v} budget seconds`} showBounds /></div>
        {current && <div className="tv-decision-pair" data-reviewed-rows>
          <div><span>Score actual remaining time</span><strong>{current.aware_label}</strong><small>Best continue − exit: {signed(current.aware_best_nonretire_gap, 6)}</small></div>
          <div><span>Time-masked scoring control</span><strong>{current.masked_label}</strong><small>Best continue − exit: {signed(current.masked_best_nonretire_gap, 6)}</small></div>
        </div>}
        <RichNarrative id="model-control:reading" value="Higher α places more value on terminal time. A positive Q gap favors continuation; a negative gap favors exit. Compare actions and Q gaps within each scoring rule, since the two Q levels use different time scores." label="Edit chart guidance" />
      </DataComponent>
      <EvidenceChart id="model-boundary" queryId="time_curves" title="Does extra time increase the value of continuing?"
        spec={{ type: "line", x: "life", y: "awareGap", fields: ["awareGap", "maskedGap"], startAtZero: false, valueDecimals: 6, stackable: false, xLabel: "Remaining budget L (seconds)", yLabel: "Best continue Q − exit Q", legend: { labels: { awareGap: "Time-aware", maskedGap: "Time-masked scoring" } } }}
        rows={curveRows} sourceRows={conditionRows} height={320}
        chartOptions={{ tooltipContent: <ChartTooltip vertical xField={chartOverrides?.["model-boundary"]?.x ?? "life"} formatValue={value => signed(Number(value), 6)} formatLabel={item => (chartOverrides?.["model-boundary"]?.legend?.labels?.[item.dataKey] ?? { awareGap: "Time-aware", maskedGap: "Time-masked scoring" }[item.dataKey] ?? item.name ?? item.dataKey)} /> }}
        description="α, market state, cash and holdings are fixed. The x-axis is remaining budget L." />
      <DataComponent id="model-q" queryId="time_curves" title={`All seven action Q values · L = ${life}`} kind="table" displayRows={qRows} sourceRows={current ? [current] : []}>
        <DataTable rows={qRows} searchable={false} caption="Action Q values and utility gaps versus exit" columns={[
          { field: "action", label: "Action" }, numericColumn("awareQ", "Time-aware Q", 6), numericColumn("awareGap", "ΔQ vs exit", 6),
          numericColumn("maskedQ", "Time-masked Q", 6), numericColumn("maskedGap", "ΔQ vs exit", 6), { field: "selected", label: "Chosen by" },
        ]} />
        <RichNarrative id="model-q:caption" value="Q is model-estimated expected utility. The frozen BTC 2022–2025 transition model looks ahead at most two steps and uses 16 empirical events per market state. Actions are exit, three allocations, and three allocations after buying time." label="Edit Q definition" />
      </DataComponent>
      <DataComponent id="model-grid" queryId="time_curves" title="Scope of action changes" kind="custom" displayRows={curves} sourceRows={curves}>
        <RichNarrative id="model-grid:body" value="Only **1 of 32 matched settings** (4 α values × 4 market states × 2 portfolios) changes action across the tested budgets. The full grid contains **672 rows** (32 settings × 21 L values). The α=0.02 example establishes a local mechanism, not a universal effect." label="Edit development-grid scope" />
        <details className="tv-details"><summary>All 21 budget values for this setting</summary><DataTable rows={conditionRows} caption="Full budget curve for the fixed state" columns={[{ field: "life", label: "L" }, { field: "aware_label", label: "Time-aware action" }, { field: "masked_label", label: "Time-masked action" }, numericColumn("aware_best_nonretire_gap", "Time-aware ΔQ", 6), numericColumn("masked_best_nonretire_gap", "Time-masked ΔQ", 6)]} searchable={false} /></details>
      </DataComponent>
    </section>

    <section className="report-section" aria-label="Time costs of EMA">
      <RichNarrative id="ema:intro" value="## Study B: The time cost of a public EMA strategy\n\nEach 15/30 EMA replay starts with 100 cash and runs for at most 30 market days. The limited version starts with 90 budget seconds and spends 5 per day. Mechanical purchases cost 8 cash for 15 seconds. Results are means across overlapping starting dates, not a continuous annual account return." label="Edit EMA experiment" />
      <div className="tv-controls" aria-label="EMA comparison year" data-reviewed-rows>
        <Dropdown label="Year" showLabel value={year} choices={["2022", "2023", "2024", "2025"]} onChange={setYear} />
      </div>
      <DataComponent id="ema-results" queryId="ema_comparison" title={`${year} · Three time rules for one EMA strategy`} kind="table" displayRows={yearRows} sourceRows={yearRows}
       >
        <DataTable rows={yearRows.map(r => ({ ...r, variant: VARIANTS[r.variant] }))} searchable={false} caption="EMA comparison by year" columns={[
          { field: "variant", label: "Rule" }, { field: "n", label: "Overlapping starts" }, numericColumn("mean_wealth", "Final cash", 2), numericColumn("mean_bars_elapsed", "Market days elapsed", 2), numericColumn("mean_remaining_life", "Budget seconds left", 2), pctColumn("fraction_wealth_above_100", "Cash > 100"),
        ]} />
      </DataComponent>
      <DataComponent id="ema-decomposition" queryId="ema_comparison" title="Separate trading P&L from time costs" kind="table" displayRows={decomposition} sourceRows={yearRows}>
        <DataTable rows={decomposition} searchable={false} caption="Final-cash accounting identity" columns={[
          { field: "variant", label: "Rule" }, numericColumn("initial", "Initial cash", 2), numericColumn("trading", "+ Net trading P&L", 2), numericColumn("timeSpend", "− Time cost", 2), numericColumn("final", "= Final cash", 2),
        ]} />
        <RichNarrative id="ema-decomposition:explain" value="Net trading P&L includes fees and slippage. Time purchases are a separate model cost. Whether longer participation covers that cost must be tested separately." label="Edit cost accounting note" />
      </DataComponent>
      <DataComponent id="price-sensitivity" queryId="price_sensitivity" title="Changing the price of 15 budget seconds · 2025" kind="table" displayRows={sensitivity} sourceRows={sensitivity}>
        <DataTable rows={sensitivity} searchable={false} caption="Mechanical purchase price sensitivity" columns={[
          numericColumn("pack_cost", "Price per pack", 0), numericColumn("trading_pnl", "Net trading P&L", 3), numericColumn("time_spend", "Total time cost", 3), numericColumn("wealth", "Final cash", 3), numericColumn("difference", "Difference vs limited EMA", 3), numericColumn("low", "95% lower bound", 3), numericColumn("high", "95% upper bound", 3),
        ]} />
        <RichNarrative id="price-sensitivity:explain" value="Only the four fixed prices 0, 2, 8 and 16 were tested. Intervals use moving-block resampling. Intermediate prices were not evaluated." label="Edit price sensitivity note" />
      </DataComponent>
    </section>

    <section className="report-section" aria-label="Historical replay and uncertainty">
      <RichNarrative id="validation:intro" value="## Study C: Historical two-step comparisons\n\nJanuary–September 2026 compares time-aware and time-masked controllers, evaluated with the same actual remaining time and utility function. The effect is mean realized utility (aware minus masked). These 32 conditions are 2 assets × 4 α values × 4 initial budgets, distinct from the development grid." label="Edit historical comparison" />
      <div className="tv-controls" aria-label="Historical comparison filters" data-reviewed-rows>
          <Dropdown label="Asset" showLabel value={validationAsset} choices={["BTCUSDT", "ETHUSDT"]} onChange={setValidationAsset} />
          <Dropdown label="α" showLabel value={validationAlpha} choices={ALPHAS} onChange={v => setValidationAlpha(Number(v))} />
          <Dropdown label="Initial L" showLabel value={validationLife} choices={[20, 45, 90, 120]} onChange={v => setValidationLife(Number(v))} />
      </div>
      <DataComponent id="validation-results" queryId="historical_validation" title="Historical differences with uncertainty intervals" kind="custom" displayRows={validation ? [validation] : []} sourceRows={validations}
       >
        {validation && <div className="tv-validation-result" data-reviewed-rows>
          <div><span>Mean realized utility difference</span><strong>{signed(validation.effect, 6)}</strong></div>
          <div><span>95% moving-block interval</span><strong>[{signed(validation.low, 6)}, {signed(validation.high, 6)}]</strong></div>
          <div><span>Does the interval include zero?</span><strong>{validation.low <= 1e-12 && validation.high >= -1e-12 ? (Math.abs(validation.low) < 1e-12 && Math.abs(validation.high) < 1e-12 ? "≈ 0 within floating-point precision" : "Yes; direction remains uncertain") : "No; this setting only"}</strong></div>
        </div>}
        {validation && <DataTable rows={[validation]} searchable={false} caption="Replay results for this setting" columns={[
          numericColumn("wealth_aware", "Time-aware: cash", 4), numericColumn("wealth_masked", "Time-masked: cash", 4), pctColumn("retire_rate_aware", "Time-aware: immediate exit rate"), numericColumn("time_spend_aware", "Time-aware: time cost", 4), { field: "n_overlapping", label: "Overlapping starts" },
        ]} />}
        <details className="tv-details"><summary>All 32 historical conditions</summary><DataTable rows={allValidationRows} caption="All 32 historical comparisons" columns={[
          { field: "symbol", label: "Asset" }, numericColumn("alpha", "α", 2), { field: "life", label: "Initial L" }, numericColumn("effect", "Utility difference", 6), { field: "interval", label: "95% interval" }, numericColumn("wealth_aware", "Time-aware cash", 4), numericColumn("wealth_masked", "Time-masked cash", 4),
        ]} /></details>
        <RichNarrative id="validation-results:limitations" value="Each condition has 222 highly overlapping starts. Intervals use paired moving-block bootstrap with 30-day blocks. The 2026 data was a frozen test for V2 but has since been examined. Further iterations require new unseen data for a fresh holdout." label="Edit historical limitations" />
      </DataComponent>
      <EvidenceChart id="validation-chart" queryId="historical_validation" title={`${validationAsset} · α = ${validationAlpha} · Intervals across initial budgets`}
        rows={validationScope} sourceRows={validationScope} height={300}
        chartOptions={{ tooltipContent: <ChartTooltip vertical xField={chartOverrides?.["validation-chart"]?.x ?? "life"} formatValue={value => signed(Number(value), 6)} formatLabel={item => (chartOverrides?.["validation-chart"]?.legend?.labels?.[item.dataKey] ?? { effect: "Mean effect", low: "95% lower bound", high: "95% upper bound" }[item.dataKey] ?? item.name ?? item.dataKey)} /> }}
        spec={{ type: "line", x: "life", y: "effect", fields: ["effect", "low", "high"], startAtZero: false, valueDecimals: 6, stackable: false, xLabel: "Initial remaining budget L", yLabel: "Realized utility difference", legend: { labels: { effect: "Mean effect", low: "95% lower bound", high: "95% upper bound" } } }}
        description="Lines show the estimate and interval bounds at four evaluated initial budgets." />
    </section>

    <Narrative id="model-math" queryId="time_curves" rows={conditionRows} className="tv-math">
      {"## Study D: Two-step model definition\n\n**Terminal utility**\n\n`U(B,L) = α log(1 + L/90) + (1−α) log(1 + B/100)`\n\nB is cash after liquidation; L is remaining budget seconds. The scales 90 and 100 are fixed. Logarithms represent diminishing marginal value. α is a chosen preference weight.\n\n**Two-step Bellman planning**\n\n`Q_d(s,a) = E[V_(d−1)(T(s,a,ξ))]`\n`V_d(s) = max_a Q_d(s,a)`\n\nξ is an empirical price event. T executes the preselected action, charges fees and time, then moves to the next state. Future states reselect actions within the two-step horizon. Leaf settlement uses the next unknown open and permits terminal exchange at price 8 within the six-pack limit. The actual 30-day boundary uses the current close. This is a finite-lookahead approximation.\n\n**A local terminal exchange rate**\n\n`MRS = (∂U/∂L)/(∂U/∂B) = [α/(1−α)] × [(100+B)/(90+L)]`\n\nAt B=100, L=90 and α=0.1, one budget second has a local value of **0.12346 cash**. Five seconds per market day correspond to **0.6173 cash**. This is a local utility trade-off; dynamic willingness to pay also depends on how extra time changes future choices."}
    </Narrative>

    <section className="report-section" aria-label="Valuing extra time">
      <Narrative id="shadow-price-explanation" queryId="shadow_prices" rows={shadowPrices}>
      {"## Study E: Three valuations of extra time\n\nAcross 64 fixed states, willingness to pay for a one-off offer of 15 extra seconds is **0.2813–2.8207 cash**, below 8. Payment occurs before the current decision, uses one pack and consumes no market step; the original price-8 shop remains available. This extra-offer counterfactual differs from the original seven-action purchase threshold.\n\nTwo separate benchmarks are shown: terminal exchange value at immediate exit, and physical-extension value with scoring time fixed and later purchases forbidden. Their assumptions differ, so they are not an additive decomposition.\n\nPhysical-extension value is zero in all 64 states: the two-step horizon covers only 10 seconds, while tested budgets start at 20. Extra time cannot relax a constraint within that visible horizon. This limitation motivated the longer planner."}
    </Narrative>
      <div className="tv-controls" aria-label="Time valuation filters" data-reviewed-rows>
          <Dropdown label="Preference α" showLabel value={shadowAlpha} choices={[0.02, 0.1]} onChange={v => setShadowAlpha(Number(v))} />
          <Dropdown label="Market state" showLabel value={shadowMarket} choices={[0, 1, 2, 3]} choiceLabels={{ 0: "5-day down / low volatility", 1: "5-day down / high volatility", 2: "5-day up / low volatility", 3: "5-day up / high volatility" }} onChange={v => setShadowMarket(Number(v))} />
          <Dropdown label="Portfolio" showLabel value={shadowAllocation} choices={["cash", "half"]} choiceLabels={{ cash: "100 cash / 0 holdings", half: "50 cash / 50 holdings" }} onChange={setShadowAllocation} />
      </div>
      <DataComponent id="shadow-price-table" queryId="shadow_prices" title="Three counterfactual values for 15 extra seconds" kind="table" displayRows={shadowRows} sourceRows={shadowRows}
       >
        <DataTable rows={shadowRows} searchable={false} caption="One-off offer, terminal exchange and physical extension" columns={[
          { field: "life", label: "Actual L" }, numericColumn("extra_offer_price", "One-off willingness to pay", 4), numericColumn("terminal_preference_price", "Terminal exchange value", 4), numericColumn("physical_extension_trade_price", "Physical-extension value", 4), numericColumn("original_best_buy_q_gap", "Buy at 8 − best no-buy ΔQ", 6),
        ]} />
      </DataComponent>
    </section>

    </>}

    <Narrative id="next-model" queryId="source_coverage" rows={coverage}>
      {"## When is extra time worth its price?\n\nContinue, exit or buy more time: finite time makes each a calculable choice. Real-data replays show that a longer horizon can change actions, while a stable performance advantage remains unproven. A next test should freeze the rules and evaluate genuinely unseen data and alternative transition models."}
    </Narrative>
  </article>;
}
