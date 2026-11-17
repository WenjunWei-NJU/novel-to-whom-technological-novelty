from pathlib import Path
import argparse,sys,json,time
import duckdb,pandas as pd,numpy as np
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "scripts" / "_shared"))
from study.analysis import Analysis
from study.fe import contrast,joint_wald,holm
from study.io import save_json
parser = argparse.ArgumentParser(description="Fit one formal inventor-level model.")
parser.add_argument("--database", type=Path, required=True, help="Read-only analysis database")
parser.add_argument("--output", type=Path, required=True, help="New results directory")
parser.add_argument("--config", type=Path, default=R / "config" / "analysis.json")
parser.add_argument("kind", choices=["full", "stage", "source", "opportunity"])
parser.add_argument("--pilot", action="store_true")
a = parser.parse_args()
for path in (a.database, a.config):
    if not path.is_file():
        parser.error(f"Required input file does not exist: {path}")
O = a.output.resolve()
O.mkdir(parents=True, exist_ok=False)
(O / "tables").mkdir()
cfg = json.loads(a.config.read_text())
cfg.update(execution_scope="PERSONAL_PILOT" if a.pilot else "PERSONAL_MODELS", source_database=str(a.database.resolve()), input_tables=["personal"])
c=duckdb.connect(cfg['source_database'],read_only=True);c.execute('set threads=2');c.execute("set memory_limit='2GB'")
fields=['ai','early_peers','senior_peers','first_peers','age','prior_patents','history_width']
cols=['patent_id','inventor_id','primary_cpc','application_year','n','ai_score']+fields
if a.kind=='stage':cols+=['stage']
outcomes=['y','A','B','C','D']if a.kind=='source'else['utilization']if a.kind=='opportunity'else['y']
if a.kind=='opportunity':cols+=['opportunity_n'];fields+=['opportunity_n']
cols+=outcomes;weight={'source':'source_weight','opportunity':'utilization_weight'}.get(a.kind,'weight')
query='select '+','.join(cols)+f',{weight} weight from personal where {weight} is not null and application_year<=2024'
if a.pilot:query+=' and hash(patent_id)%100=0'
start=time.time();d=c.execute(query).df();print(a.kind,'loaded',len(d),'MB',round(d.memory_usage(deep=True).sum()/1e6),flush=True)
name={'full':'personal_full','stage':'personal_stage','source':'source_composition','opportunity':'opportunity_utilization'}[a.kind]
initial_n=len(d);holder=[d];del d
job=Analysis(c,cfg,O);fit,data=job.fit(name,holder.pop(),fields,outcomes,'full'if a.kind=='source'else a.kind,True)
if fit is None:raise RuntimeError(job.status)
names=fit['names'];unit=lambda key:np.eye(len(names))[names.index(key)]
if a.kind=='source':
 b=np.array([r['beta']for r in fit['outcomes']]);assert np.max(abs(b[0]-b[1:].sum(axis=0)))<1e-8
for outcome,r in zip(outcomes,fit['outcomes']):
 for cov,info in r['covariance'].items():
  def add(family,target,label,vec,joint=False):
   value=joint_wald(r['beta'],info['V'],vec,info['df'],info['inference_ok'])if joint else contrast(r['beta'],info['V'],vec,info['df'],info['inference_ok'])
   job.add(family,name,target,label,value,covariance=cov,N=fit['N'])
  if a.kind=='full':
   add('PERSONAL',outcome,'E_AI',unit('E_AI'));add('PERSONAL',outcome,'peer_AI_restrictions',np.vstack([unit('FIRSTpeer_AI'),unit('SENIORpeer_AI')]),True)
  elif a.kind=='stage':
   for stage in ['E','M','S']:add('CAREER_STAGE',stage,'E_AI',unit('E_AI')+(unit('E_AI_stage_'+stage)if stage!='M'else 0))
   add('CAREER_STAGE','all','stage_equality',np.vstack([unit('E_AI_stage_E'),unit('E_AI_stage_S')]),True)
  else:add('SOURCE_COMPOSITION'if a.kind=='source'else'OPPORTUNITY',outcome,'E_AI',unit('E_AI'))
rows=pd.DataFrame(job.estimands)
for fam,idx in rows.groupby('family').groups.items():
 rows.loc[idx,'p_Holm']=holm([float(v)if pd.notna(v)else None for v in rows.loc[idx,'p_raw']]);rows.loc[idx,'family_size']=len(idx)
rows.to_csv(O/'tables/estimands.csv',index=False,encoding='utf-8-sig')
for label,vals in [('sample_sizes',job.samples),('coefficients',job.coefficients),('model_status',job.status)]:pd.DataFrame(vals).to_csv(O/f'tables/{label}.csv',index=False,encoding='utf-8-sig')
save_json(O/'analysis_settings.json',cfg);save_json(O/'execution_summary.json',dict(seconds=time.time()-start,initial_N=initial_n,final_N=fit['N'],scope=cfg['execution_scope'],component_identity_checked=a.kind=='source'))
print(rows[['family','outcome','covariance','estimate','p_raw','p_Holm']].to_string(index=False),flush=True)
