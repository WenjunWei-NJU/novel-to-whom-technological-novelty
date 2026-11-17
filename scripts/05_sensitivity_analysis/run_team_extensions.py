from pathlib import Path
import argparse,sys,json,time,gc
import duckdb,pandas as pd,numpy as np
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "scripts" / "_shared"))
from study.analysis import Analysis,OUTCOMES
from study.fe import contrast,holm
from study.io import save_json
parser = argparse.ArgumentParser(description="Estimate TEAM_EXTENSIONS.")
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
cfg.update(execution_scope="TEAM_EXTENSIONS", source_database=str(args.database.resolve()))
c=duckdb.connect(cfg['source_database'],read_only=True);c.execute('set threads=2');c.execute("set memory_limit='2GB'")
fields=['ai','early_share','senior_share','age_mean','logq_mean','width_mean','first_share'];cols=['patent_id','primary_cpc','application_year','n','n_early','n_senior','ai_score','pair_n','mean_age_distance']+fields+OUTCOMES
base='select '+','.join(cols)+',1.0 weight from teams where n>=2 and n_unknown=0 and eligible_n>0 and ai is not null and application_year between 2000 and 2024'
job=Analysis(c,cfg,O);start=time.time()
for name,kind,family,outcomes in [('element_scales','surface','ELEMENT_SCALES',OUTCOMES[:2]),('career_distance','distance','CONTINUOUS_DISTANCE',OUTCOMES),('pair_denominator','denominator','DENOMINATOR',OUTCOMES),('linear_composition','linear','FUNCTIONAL_FORM',OUTCOMES)]:
 query=base
 if name!='element_scales':query+=' and relative_y is not null and team_y is not null and pair_y is not null'
 data=c.execute(query).df();extra=['mean_age_distance']if kind=='distance'else['pair_n']if kind=='denominator'else[]
 fit,common=job.fit(name,data,fields+extra,outcomes,kind)
 if fit is None:raise RuntimeError(job.status[-1])
 if kind=='distance':
  for outcome,r in zip(outcomes,fit['outcomes']):
   info=r['covariance']['CPC'];vec=np.eye(len(fit['names']))[fit['names'].index('career_distance_10y_AI')]
   job.add(family,name,outcome,'10y_AI_difference',contrast(r['beta'],info['V'],vec,info['df'],info['inference_ok']),N=fit['N'])
 else:job.summarize_team(fit,family)
 del fit,common,data;gc.collect();print('Completed',name,flush=True)
rows=pd.DataFrame(job.estimands)
for fam,idx in rows.groupby('family').groups.items():
 vals=[float(v)if pd.notna(v)else None for v in rows.loc[idx,'p_raw']];rows.loc[idx,'p_Holm']=holm(vals);rows.loc[idx,'family_size']=len(idx)
 raw=rows.loc[idx,'p_raw'].fillna(1).to_numpy();order=np.argsort(raw);expected=np.maximum.accumulate((len(idx)-np.arange(len(idx)))*raw[order]).clip(0,1);valid=rows.loc[idx,'p_raw'].notna().to_numpy()[order];np.testing.assert_allclose(rows.loc[idx,'p_Holm'].to_numpy()[order][valid],expected[valid],rtol=1e-12,atol=1e-15)
rows['data_kind']='OBSERVED';rows.to_csv(O/'tables/estimands.csv',index=False,encoding='utf-8-sig')
for label,v in [('sample_sizes',job.samples),('coefficients',job.coefficients),('model_status',job.status),('profile_support',job.profile_rows)]:pd.DataFrame(v).to_csv(O/f'tables/{label}.csv',index=False,encoding='utf-8-sig')
save_json(O/'analysis_settings.json',cfg);save_json(O/'execution_summary.json',dict(seconds=time.time()-start,tests=len(rows),independent_Holm_verified=True))
print(rows[['family','outcome','estimate','p_Holm','N']].to_string(index=False),flush=True)
