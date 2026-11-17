"""Run only the seven MAIN tests, retaining actual-data pilot/full separation."""
from pathlib import Path
import sys,json,argparse,time
import duckdb,pandas as pd,numpy as np
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "scripts" / "_shared"))
from study.analysis import Analysis,OUTCOMES
from study.fe import holm
from study.io import save_json
parser = argparse.ArgumentParser(description="Estimate the seven MAIN tests.")
parser.add_argument("--database", type=Path, required=True, help="Read-only analysis database")
parser.add_argument("--output", type=Path, required=True, help="New results directory")
parser.add_argument("--config", type=Path, default=R / "config" / "analysis.json")
parser.add_argument("--pilot", action="store_true")
args = parser.parse_args()
for path in (args.database, args.config):
    if not path.is_file():
        parser.error(f"Required input file does not exist: {path}")
O = args.output.resolve()
O.mkdir(parents=True, exist_ok=False)
(O / "tables").mkdir()
config = json.loads(args.config.read_text())
config.update(synthetic=False, execution_scope="MAIN_ONLY_PILOT" if args.pilot else "MAIN_ONLY", source_database=str(args.database.resolve()), aipat_source="Wu et al. (2026), AIPat")
config.pop('inputs',None);config.pop('data_root',None);config['input_mode']='READ_EXISTING_CORE_DATABASE';config['input_tables']=['teams']
c=duckdb.connect(config['source_database'],read_only=True);c.execute('set threads=2');c.execute("set memory_limit='2GB'")
fields=['ai','early_share','senior_share','age_mean','logq_mean','width_mean','first_share']
needed=['patent_id','primary_cpc','application_year','n','n_early','n_senior','ai_score']+list(dict.fromkeys(fields+OUTCOMES))
query='select '+','.join(needed)+',1.0 weight from teams where n>=2 and n_unknown=0 and eligible_n>0 and ai is not null'
if args.pilot:query+=' and hash(patent_id)%100=0'
start=time.time();d=c.execute(query).df();print('Loaded',len(d),'rows; frame MB',round(d.memory_usage(deep=True).sum()/1e6,1),flush=True)
job=Analysis(c,config,O);fit,common=job.fit('main_surface',d,fields,OUTCOMES,'surface')
if fit is None:raise RuntimeError(job.status)
job.summarize_team(fit,'MAIN');rows=pd.DataFrame(job.estimands);assert len(rows)==7
rows['p_Holm']=holm([float(x)if pd.notna(x)else None for x in rows.p_raw]);rows['family_size']=7;rows['data_kind']='OBSERVED';rows['execution_scope']=config['execution_scope'];rows.to_csv(O/'tables/estimands.csv',index=False,encoding='utf-8-sig')
for name,data in [('coefficients',job.coefficients),('sample_sizes',job.samples),('model_status',job.status),('profile_support',job.profile_rows)]:pd.DataFrame(data).to_csv(O/f'tables/{name}.csv',index=False,encoding='utf-8-sig')
common.groupby(['ai','n','n_early','n_senior','first_share'],observed=True).size().rename('N').reset_index().to_csv(O/'tables/composition_support.csv',index=False)
save_json(O/'analysis_settings.json',config);save_json(O/'execution_summary.json',dict(seconds=time.time()-start,initial_rows=len(d),final_rows=len(common),scope=config['execution_scope']))
print(rows[['outcome','contrast','p_raw','p_Holm']].to_string(index=False),flush=True);print('Final N',len(common),flush=True)
