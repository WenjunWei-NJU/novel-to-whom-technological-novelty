from pathlib import Path
import argparse,sys,json,gc,time
import duckdb,pandas as pd,numpy as np
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "scripts" / "_shared"))
from study.analysis import Analysis,OUTCOMES,period_ranges
from study.fe import holm
from study.io import save_json
parser = argparse.ArgumentParser(description="Estimate THRESHOLDS_PERIODS_WINDOWS_AI_FIELDS.")
parser.add_argument("--database", type=Path, required=True, help="Read-only analysis database")
parser.add_argument("--output", type=Path, required=True, help="New results directory")
parser.add_argument("--config", type=Path, default=R / "config" / "analysis.json")
args = parser.parse_args()
for path in (args.database, args.config):
    if not path.is_file():
        parser.error(f"Required input file does not exist: {path}")
O = args.output.resolve()
O.mkdir(parents=True, exist_ok=False)
(O / "tables").mkdir()
cfg = json.loads(args.config.read_text())
cfg.update(execution_scope="THRESHOLDS_PERIODS_WINDOWS_AI_FIELDS", source_database=str(args.database.resolve()))
c=duckdb.connect(cfg['source_database'],read_only=True);c.execute('set threads=2');c.execute("set memory_limit='2GB'")
fields=['ai','early_share','senior_share','age_mean','logq_mean','width_mean','first_share'];cols=['patent_id','primary_cpc','application_year','n','n_early','n_senior','ai_score']+fields+OUTCOMES
for w in [5,10]:cols += [f'relative_y{w}',f'relative_full_on_w{w}',f'team_y{w}',f'team_full_on_w{w}']
base='select '+','.join(cols)+',1.0 weight from teams where n>=2 and n_unknown=0 and eligible_n>0 and ai is not null and application_year between 2000 and 2024'
team=c.execute(base).df();job=Analysis(c,cfg,O);start=time.time()
def checkpoint():
 rows=list(job.estimands)
 for family in sorted({r['family']for r in rows}):
  group=[r for r in rows if r['family']==family];size={'THRESHOLDS':27,'PERIODS':15,'HISTORY_WINDOWS':12,'AI_MEASUREMENT':9,'FIELDS':27}[family]
  raw=[r.get('p_raw') for r in group];adjusted=holm(raw+[None]*(size-len(group)))[:len(group)]
  for r,p in zip(group,adjusted):r['p_Holm']=p;r['family_size']=size
 pd.DataFrame(rows).to_csv(O/'tables/estimands_running.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(job.status).to_csv(O/'tables/model_status.csv',index=False,encoding='utf-8-sig')
def run(name,data,family):
 fit,common=job.fit(name,data,fields,OUTCOMES,'surface')
 if fit is not None:job.summarize_team(fit,family)
 print('Completed',name,'N',fit['N']if fit else'not_estimable',flush=True);checkpoint();del fit,common;gc.collect()
for early in [3,5,7]:
 for senior in [15,16,20]:
  counts=c.execute(f'select patent_id,count(*) filter(where age<={early})::integer n_early,count(*) filter(where age>={senior})::integer n_senior,count(*) filter(where age<={early})::double/count(*) early_share,count(*) filter(where age>={senior})::double/count(*) senior_share from personal group by 1').df()
  data=team.drop(columns=['early_share','senior_share','n_early','n_senior']).merge(counts,on='patent_id',validate='one_to_one');run(f'threshold_E{early}_S{senior}',data,'THRESHOLDS');del counts,data;gc.collect()
for low,high in period_ranges(cfg):run(f'period_{low}_{high}',team.loc[team.application_year.between(low,high)].copy(),'PERIODS')
for w in [5,10]:
 paired=[f'relative_full_on_w{w}',f'team_full_on_w{w}',f'relative_y{w}',f'team_y{w}','pair_y'];sample=team.dropna(subset=paired)
 for ref in ['full',str(w)]:
  data=sample.copy();data['relative_y']=data[f'relative_full_on_w{w}']if ref=='full'else data[f'relative_y{w}'];data['team_y']=data[f'team_full_on_w{w}']if ref=='full'else data[f'team_y{w}'];run(f'history_{w}_{ref}',data,'HISTORY_WINDOWS');del data
 del sample;gc.collect()
for label,threshold in [('p30',.3),('p70',.7),('continuous',None)]:
 data=team.copy();data['ai']=(data.ai_score>=threshold).astype(float)if threshold is not None else data.ai_score;run('ai_'+label,data,'AI_MEASUREMENT');del data;gc.collect()
for section in list('ABCDEFGHY'):
 data=team.loc[team.primary_cpc.str.startswith(section,na=False)].copy()
 if len(data)<500:
  job.status.append(dict(model='field_'+section,status='insufficient_support',N=len(data),reason='Below 500 eligible source rows'))
 else:run('field_'+section,data,'FIELDS')
 del data;gc.collect()
# Reconcile every planned identifier including failures, preserving family size.
from study.analysis import planned_tests
families={'THRESHOLDS','PERIODS','HISTORY_WINDOWS','AI_MEASUREMENT','FIELDS'};keys=['family','model','outcome','contrast','covariance'];actual={tuple(r[k]for k in keys):r for r in job.estimands};rows=[];status={r['model']:r for r in job.status}
for plan in planned_tests(cfg):
 if plan['family']not in families:continue
 key=tuple(plan[k]for k in keys);row=actual.pop(key,None)
 if row is None:row=dict(plan,p_raw=None,estimate=None,inference_ok=False,test_status=status.get(plan['model'],{}).get('status','not_computed'))
 else:row['test_status']='estimated'if row['inference_ok']else'inference_unavailable'
 rows.append(row)
assert not actual;df=pd.DataFrame(rows)
for family,idx in df.groupby('family').groups.items():
 raw=[float(v)if pd.notna(v)else None for v in df.loc[idx,'p_raw']];df.loc[idx,'p_Holm']=[v if raw[i]is not None else np.nan for i,v in enumerate(holm(raw))];df.loc[idx,'family_size']=len(idx)
 arr=np.array([v if v is not None else 1 for v in raw]);order=np.argsort(arr);expected=np.maximum.accumulate((len(idx)-np.arange(len(idx)))*arr[order]).clip(0,1);valid=np.array([v is not None for v in raw])[order];np.testing.assert_allclose(df.loc[idx,'p_Holm'].to_numpy()[order][valid],expected[valid],rtol=1e-12,atol=1e-15)
df['data_kind']='OBSERVED';df.to_csv(O/'tables/estimands.csv',index=False,encoding='utf-8-sig')
for name,vals in [('sample_sizes',job.samples),('coefficients',job.coefficients),('model_status',job.status),('profile_support',job.profile_rows)]:pd.DataFrame(vals).to_csv(O/f'tables/{name}.csv',index=False,encoding='utf-8-sig')
save_json(O/'analysis_settings.json',cfg);save_json(O/'execution_summary.json',dict(seconds=time.time()-start,planned_tests=len(df),independent_Holm_verified=True));print('FINISHED',len(df),'planned tests',flush=True)
