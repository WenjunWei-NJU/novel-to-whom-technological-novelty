from pathlib import Path
import argparse,sys,json,time,gc
import duckdb,pandas as pd,numpy as np
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "scripts" / "_shared"))
from study.analysis import Analysis,OUTCOMES
from study.fe import holm
from study.io import save_json
parser = argparse.ArgumentParser(description="Estimate HISTORY_COMPLETENESS.")
parser.add_argument("--database", type=Path, required=True, help="Read-only analysis database")
parser.add_argument("--output", type=Path, required=True, help="New results directory")
parser.add_argument("--config", type=Path, default=R / "config" / "analysis.json")
parser.add_argument("--patent-dates", type=Path, required=True)
parser.add_argument("--patent-inventors", type=Path, required=True)
args = parser.parse_args()
for path in (args.database, args.config, args.patent_dates, args.patent_inventors):
    if not path.is_file():
        parser.error(f"Required input file does not exist: {path}")
O = args.output.resolve()
O.mkdir(parents=True, exist_ok=False)
(O / "tables").mkdir()
cfg = json.loads(args.config.read_text())
cfg.update(execution_scope="HISTORY_COMPLETENESS", source_database=str(args.database.resolve()))
c=duckdb.connect(cfg['source_database'],read_only=True);c.execute('set threads=2');c.execute("set memory_limit='2GB'")
def lit(p):return "'"+str(p).replace("'","''")+"'"
c.execute("create temp view grant_dates as select patent_id,patent_date::date grant_date from read_parquet("+lit(args.patent_dates.resolve())+") where patent_date::date<=date '2025-12-30'")
boundary=c.execute("select min(grant_date),min(grant_date)+interval '5 years' from grant_dates").fetchone()
c.execute('create temp table first_grants as select m.inventor_id,min(p.grant_date) first_grant from read_parquet('+lit(args.patent_inventors.resolve())+') m join grant_dates p using(patent_id) group by 1')
c.execute('create temp table team_origins as select p.patent_id,min(g.first_grant) first_grant from personal p join first_grants g using(inventor_id) group by 1')
fields=['ai','early_share','senior_share','age_mean','logq_mean','width_mean','first_share'];cols=['t.patent_id','t.primary_cpc','t.application_year','t.n','t.n_early','t.n_senior','t.ai_score']+['t.'+f for f in fields+OUTCOMES]
query='select '+','.join(cols)+',1.0 weight from teams t'
conditions="t.n>=2 and t.n_unknown=0 and t.eligible_n>0 and t.ai is not null and t.application_year between 2000 and 2024"
job=Analysis(c,cfg,O);start=time.time()
for name,q in [('strict_history',query+' where '+conditions+' and not t.any_history_gap and t.uncertain_new_pairs=0'),('left_boundary_excluded',query+' join team_origins o using(patent_id) where '+conditions+" and o.first_grant>=timestamp '"+str(boundary[1])+"'")]:
 data=c.execute(q).df();fit,common=job.fit(name,data,fields,OUTCOMES,'surface');job.summarize_team(fit,'HISTORY_COMPLETENESS');print(name,fit['N']if fit else job.status[-1],flush=True);del data,fit,common;gc.collect()
rows=pd.DataFrame(job.estimands);assert len(rows)==6;raw=rows.p_raw.to_numpy();rows['p_Holm']=holm([float(v)if pd.notna(v)else None for v in raw]);rows['family_size']=6;rows['data_kind']='OBSERVED';rows.to_csv(O/'tables/estimands.csv',index=False,encoding='utf-8-sig')
order=np.argsort(np.nan_to_num(raw,nan=1));expected=np.maximum.accumulate((6-np.arange(6))*np.nan_to_num(raw,nan=1)[order]).clip(0,1);valid=np.isfinite(raw)[order];np.testing.assert_allclose(rows.p_Holm.to_numpy()[order][valid],expected[valid],rtol=1e-12,atol=1e-15)
for name,v in [('sample_sizes',job.samples),('coefficients',job.coefficients),('model_status',job.status),('profile_support',job.profile_rows)]:pd.DataFrame(v).to_csv(O/f'tables/{name}.csv',index=False,encoding='utf-8-sig')
save_json(O/'analysis_settings.json',cfg);save_json(O/'execution_summary.json',dict(seconds=time.time()-start,history_start=str(boundary[0]),boundary_cutoff=str(boundary[1]),independent_Holm_verified=True));print(rows[['model','outcome','estimate','p_Holm','N']].to_string(index=False))
