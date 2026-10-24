"""Fixed, exploratory real-data replay: old 2-step vs full 30-step planning.

Run `python3 long_horizon_v3/experiment.py --workers 2` from the workspace.
The protocol is written before any new replay. No fitted kernel is changed.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse, csv, gzip, hashlib, itertools, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from time_budget_policy_v2 import planner as old, market
from moving_average_life_compare import engine as ema_engine
from long_horizon_v3.long_planner import FullHorizonPlanner

OUT = ROOT / 'long_horizon_v3'
CONTROLLERS = ('dp30','dp2','dp30_no_mid_buy','ema_time','ema_mechanical','retire_now')
SETTINGS = list(itertools.product((8.,2.,0.),(.1,.02),(20.,45.,90.,120.)))
TOTAL_KEYS = ('commission','slippage_cost','fees','turnover','trades','life_spend','packs_bought')
LEDGER_FIELDS = ('symbol','controller','pack_cost','alpha','initial_life','start_index','step','date',
 'cash_before','holding_before','life_before','packs_before','market_state','intended_action',
 'executed_action','gap','intraday','cash_after','holding_after','life_after','packs_after','done',
 'terminal_reason','actual_utility',*TOTAL_KEYS,'mid_packs','terminal_packs')

def json_write(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def source_hashes():
    files=['time_budget_policy_v2/kernel.json','time_budget_policy_v2/planner.py',
           'time_budget_policy_v2/market.py','moving_average_life_compare/engine.py']
    files += [f'time_budget_policy_v2/data/{s}-1d-2026.csv' for s in ('BTCUSDT','ETHUSDT')]
    files += [f'real_market_four_stage/data/BTCUSDT-1d-{y}.csv' for y in range(2022,2026)]
    return {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}

def freeze_protocol():
    path=OUT/'protocol.json'
    if path.exists():
        p=json.loads(path.read_text())
        assert p['source_hashes']==source_hashes(),'Frozen sources changed'
        return p
    p={'status':'exploratory historical replication; 2026 previously inspected',
       'question':'Does extending contingent planning from 2 to 30 steps change buying time and stopping?',
       'training':'Frozen BTC2022–2025 four-state empirical kernel; no refitting',
       'test_period':'2026-01-01 to 2026-09-30; BTCUSDT and ETHUSDT, 273 daily rows each',
       'starts':'range(30, len(data)-30) = 213 overlapping starts per asset, same for all controllers',
       'settings':[{'pack_cost':p,'alpha':a,'initial_life':l} for p,a,l in SETTINGS],
       'controllers':list(CONTROLLERS),'primary_pack_cost':8,'primary_alpha':.1,
       'config':asdict(old.Config()),'ema':{'fast':15,'slow':30,'buy_tolerance':.00015},
       'fairness':'Same costs, actual horizon, safety, and optimal terminal exchange for every constrained controller. Original unlimited-time EMA is a separate wealth-only benchmark.',
       'numerics':'Full finite 30-step Bellman DP, exact reachable life/fraction states; linear wealth interpolation step1, dynamic reachable bounds. Fine step.5 checks primary price8 conditions.',
       'uncertainty':'Paired circular moving-block bootstrap, blocks30 and60, 2000 draws seed20261024; nonoverlapping stride31 sensitivity; episodes are not independent.',
       'terminal_exchange':'Terminal purchases separated from mid-episode purchases; terminal exchange cannot extend trading.',
       'selection':'All fixed settings retained. Illustrations use first chronological mid-buy, if any, not best return.',
       'source_hashes':source_hashes(),'written_at_unix':time.time()}
    json_write(path,p)
    return p

def evaluate(data, record, kernel, controller, cfg, alpha, starts, model, writer):
    n=len(starts); cash=np.full(n,100.); holding=np.zeros(n); life=np.full(n,cfg.initial_life)
    packs=np.zeros(n,int); active=np.ones(n,bool); scores=np.zeros(n); reasons=np.full(n,'',object)
    exit_step=np.full(n,-1,int); exit_day=exit_step.copy(); totals={k:np.zeros(n) for k in TOTAL_KEYS}
    mid_total=np.zeros(n,int); terminal_total=np.zeros(n,int)
    m_all=market.states(data,record)
    fast=np.asarray(ema_engine.ema_values(data.closes,15),float)
    slow=np.asarray(ema_engine.ema_values(data.closes,30),float)
    for step in range(cfg.horizon_steps):
        ids=np.flatnonzero(active)
        if not len(ids): break
        days=starts[ids]+step; m=m_all[days]; R=cfg.horizon_steps-step
        before=[v[ids].copy() for v in (cash,holding,life,packs)]
        if controller.startswith('dp30'):
            actions=model.choose(*before,R,m)['action_index'].copy()
        elif controller=='dp2':
            actions=old.choose(*before,R,m,alpha,kernel,depth=2,config=cfg)['action_index'].copy()
        elif controller=='retire_now': actions=np.zeros(len(ids),int)
        else:
            bullish=fast[days-1]>slow[days-1]*(1+.00015)
            full=np.where(holding[ids]>1e-10,fast[days-1]>=slow[days-1],bullish)
            actions=np.where(full,3,1)
            request=(controller=='ema_mechanical') & (life[ids]<=2*cfg.dt) & (packs[ids]<cfg.max_packs) & bullish
            actions[request]+=3
            actions[(life[ids]<=cfg.dt)&~request]=0
        intended=actions.copy()
        # First access to today's price occurs AFTER every action was selected.
        gaps=data.opens[days]/data.closes[days-1]; bodies=data.closes[days]/data.opens[days]
        event={k:np.zeros(len(ids)) for k in TOTAL_KEYS}
        done=np.zeros(len(ids),bool); reason=np.full(len(ids),'continue',object); score=np.zeros(len(ids))
        j=np.flatnonzero(actions!=0)
        if len(j):
            o=old.transition(cash[ids[j]],holding[ids[j]],life[ids[j]],packs[ids[j]],R,
                actions[j],gaps[j],bodies[j],m_all[days[j]+1],alpha,cfg)
            bad=j[~o['valid']]
            if len(bad):
                safe=life[ids[bad]]<=cfg.dt
                actions[bad[safe]]=0; actions[bad[~safe]]-=3
                j=np.flatnonzero(actions!=0)
                o=old.transition(cash[ids[j]],holding[ids[j]],life[ids[j]],packs[ids[j]],R,
                    actions[j],gaps[j],bodies[j],m_all[days[j]+1],alpha,cfg)
                if not np.all(o['valid']): raise ArithmeticError('Invalid fallback')
            for key,target in (('cash',cash),('holding',holding),('life',life),('packs_used',packs)):
                target[ids[j]]=o[key]
            for k in TOTAL_KEYS:event[k][j]=o[k]
            done[j]=o['done'];reason[j]=o['terminal_reason'];score[j]=o['utility']
        j=np.flatnonzero(actions==0)
        if len(j):
            o=old.settle_retirement(cash[ids[j]],holding[ids[j]]*gaps[j],life[ids[j]],packs[ids[j]],alpha,cfg)
            for key,target in (('cash',cash),('holding',holding),('life',life),('packs_used',packs)):
                target[ids[j]]=o[key]
            for k in TOTAL_KEYS:event[k][j]=o[k]
            done[j]=True;reason[j]='retire';score[j]=o['utility']
        mid=(actions>=4).astype(int); terminal=event['packs_bought'].astype(int)-mid
        assert np.all(terminal>=0)
        for k in TOTAL_KEYS:totals[k][ids]+=event[k]
        mid_total[ids]+=mid;terminal_total[ids]+=terminal
        ended=ids[done];active[ended]=False;scores[ended]=score[done];reasons[ended]=reason[done]
        exit_step[ended]=step+(reason[done]!='retire');exit_day[ended]=days[done]
        if writer:
            for j,i in enumerate(ids):
                writer.writerow(dict(symbol=data.symbol,controller=controller,pack_cost=cfg.pack_cost,alpha=alpha,
                    initial_life=cfg.initial_life,start_index=int(starts[i]),step=step,date=data.dates[days[j]],
                    cash_before=before[0][j],holding_before=before[1][j],life_before=before[2][j],packs_before=before[3][j],
                    market_state=int(m[j]),intended_action=int(intended[j]),executed_action=int(actions[j]),
                    gap=gaps[j],intraday=bodies[j],cash_after=cash[i],holding_after=holding[i],life_after=life[i],
                    packs_after=int(packs[i]),done=bool(done[j]),terminal_reason=reason[j],actual_utility=score[j],
                    **{k:v[j] for k,v in event.items()},mid_packs=int(mid[j]),terminal_packs=int(terminal[j])))
    assert not active.any()
    return [dict(symbol=data.symbol,controller=controller,pack_cost=cfg.pack_cost,alpha=alpha,
        initial_life=cfg.initial_life,start_index=int(starts[i]),start_date=data.dates[starts[i]],
        exit_date=data.dates[exit_day[i]],final_cash=float(cash[i]),remaining_life=float(life[i]),utility=float(scores[i]),
        total_life_spend=float(totals['life_spend'][i]),mid_packs=int(mid_total[i]),terminal_packs=int(terminal_total[i]),
        fees=float(totals['fees'][i]),trades=int(totals['trades'][i]),exit_step=int(exit_step[i]),reason=str(reasons[i]),
        trading_pnl=float(cash[i]+totals['life_spend'][i]-100)) for i in range(n)]

def run_setting(setting):
    price,alpha,L=setting; dest=OUT/'runs'/f'cost{price:g}_alpha{alpha:g}_L{L:g}'
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return {'setting':setting,'cached':True}
    record=json.loads((ROOT/'time_budget_policy_v2/kernel.json').read_text())
    kernel=market.arrays(record,exact_retirement=True)
    cfg=old.Config(initial_life=L,pack_cost=price)
    all_eps=[];regions=[];diagnostics={}
    with gzip.open(dest/'ledger.csv.gz','wt',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=LEDGER_FIELDS);writer.writeheader()
        for allow,controls in ((True,('dp30','dp2','ema_time','ema_mechanical','retire_now')),
                               (False,('dp30_no_mid_buy',))):
            model=FullHorizonPlanner(kernel,alpha,initial_life=L,config=cfg,allow_mid_buy=allow).build()
            if allow:
                for C,H,m in itertools.product((100.,50.),(0,),range(4)):
                    H=100-C
                    d=model.choose(C,H,L,0,30,m);d2=old.choose(C,H,L,0,30,m,alpha,kernel,depth=2,config=cfg)
                    q=np.asarray(d['q_values']).tolist();q2=np.asarray(d2['q_values']).tolist()
                    regions.append(dict(pack_cost=price,alpha=alpha,initial_life=L,market_state=m,cash=C,holding=H,
                        action=int(d['action_index']),action2=int(d2['action_index']),q_retire=q[0],
                        best_continue_gap=max(q[1:])-q[0],best_buy_gap=max(q[4:])-max(q[:4]),
                        **{f'q{i}':v if np.isfinite(v) else None for i,v in enumerate(q)},
                        **{f'q2_{i}':v if np.isfinite(v) else None for i,v in enumerate(q2)}))
            for symbol in ('BTCUSDT','ETHUSDT'):
                data=market.load_csv(ROOT/f'time_budget_policy_v2/data/{symbol}-1d-2026.csv',symbol)
                starts=np.arange(30,len(data.dates)-30)
                for controller in controls:
                    all_eps.extend(evaluate(data,record,kernel,controller,cfg,alpha,starts,model,writer))
            diagnostics['full' if allow else 'no_mid_buy']=model.metadata()
            del model
    with (dest/'episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=all_eps[0]);w.writeheader();w.writerows(all_eps)
    json_write(dest/'regions.json',regions);json_write(dest/'diagnostics.json',diagnostics)
    json_write(dest/'complete.json',{'episodes':len(all_eps),'source_hashes':source_hashes()})
    return {'setting':setting,'episodes':len(all_eps)}

def bootstrap(v,block):
    v=np.asarray(v,float);rng=np.random.default_rng(20261024);n=len(v)
    picks=rng.integers(0,n,size=(2000,int(np.ceil(n/block))))
    indices=(picks[:,:,None]+np.arange(block)[None,None,:])%n
    means=v[indices.reshape(2000,-1)[:,:n]].mean(axis=1)
    return np.quantile(means,[.025,.975]).tolist()

def aggregate():
    episodes=[];regions=[];diags=[]
    for d in sorted((OUT/'runs').glob('cost*')):
        if not (d/'complete.json').exists():continue
        for r in csv.DictReader((d/'episodes.csv').open()):
            for k in r:
                if k not in ('symbol','controller','start_date','exit_date','reason'):r[k]=float(r[k])
            episodes.append(r)
        regions.extend(json.loads((d/'regions.json').read_text()))
        diags.append({'setting':d.name,**json.loads((d/'diagnostics.json').read_text())})
    rows=[];contrasts=[]
    for symbol,p,a,L in itertools.product(('BTCUSDT','ETHUSDT'),(8.,2.,0.),(.1,.02),(20.,45.,90.,120.)):
        group=[r for r in episodes if (r['symbol'],r['pack_cost'],r['alpha'],r['initial_life'])==(symbol,p,a,L)]
        if not group:continue
        ordered={c:sorted([r for r in group if r['controller']==c],key=lambda r:r['start_index']) for c in CONTROLLERS}
        assert all(len(v)==213 for v in ordered.values())
        identity=dict(symbol=symbol,pack_cost=p,alpha=a,initial_life=L,n=213)
        for c,rs in ordered.items():
            row={**identity,'controller':c}
            for k in ('final_cash','remaining_life','utility','total_life_spend','mid_packs','terminal_packs','fees','trades','exit_step','trading_pnl'):
                row[k]=float(np.mean([r[k] for r in rs]))
            row.update(immediate_exit_rate=float(np.mean([r['exit_step']==0 for r in rs])),
                death_rate=float(np.mean([r['reason']=='death' for r in rs])),
                mid_buy_rate=float(np.mean([r['mid_packs']>0 for r in rs])),terminal_buy_rate=float(np.mean([r['terminal_packs']>0 for r in rs])))
            rows.append(row)
        for baseline in CONTROLLERS[1:]:
            differences=np.array([x['utility']-y['utility'] for x,y in zip(ordered['dp30'],ordered[baseline])])
            lo30,hi30=bootstrap(differences,30);lo60,hi60=bootstrap(differences,60)
            contrasts.append({**identity,'baseline':baseline,'effect':float(differences.mean()),'low30':lo30,'high30':hi30,
                'low60':lo60,'high60':hi60,'nonoverlap_n':len(differences[::31]),
                'nonoverlap_effect':float(differences[::31].mean()),
                'cash_effect':float(np.mean([x['final_cash']-y['final_cash'] for x,y in zip(ordered['dp30'],ordered[baseline])]))})
    original=[]
    for symbol in ('BTCUSDT','ETHUSDT'):
        raw=list(csv.DictReader((ROOT/f'time_budget_policy_v2/data/{symbol}-1d-2026.csv').open()))
        vals=[ema_engine.run_episode(raw,i,'original') for i in range(30,len(raw)-30)]
        # The original public EMA has no time constraint or time purchases.
        original.append({'symbol':symbol,'n':len(vals),'final_cash':float(np.mean([v['wealth'] for v in vals])),
                         'interpretation':'unlimited-time original EMA, wealth only, no comparable utility'})
    with (OUT/'episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=episodes[0]);w.writeheader();w.writerows(episodes)
    result={'comparison':rows,'contrasts':contrasts,'regions':regions,'original_ema':original,
            'diagnostics':diags,'source_hashes_after':source_hashes(),'episodes':len(episodes)}
    assert source_hashes()==freeze_protocol()['source_hashes']
    json_write(OUT/'results.json',result)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--pilot',action='store_true');parser.add_argument('--aggregate-only',action='store_true')
    args=parser.parse_args();freeze_protocol()
    if not args.aggregate_only:
        settings=SETTINGS[:1] if args.pilot else SETTINGS
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run_setting,s) for s in settings]
            for f in as_completed(futures): print(json.dumps(f.result()),flush=True)
    if not args.pilot:
        r=aggregate();print(json.dumps({'episodes':r['episodes'],'groups':len(r['comparison'])}),flush=True)

if __name__=='__main__':main()
