"""Audit threshold-specific endpoint support on the actual common FE sample."""
from pathlib import Path
import argparse,sys,json
import duckdb,pandas as pd,numpy as np
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "_shared"))
from study.analysis import profile_support

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--database", type=Path, required=True)
parser.add_argument("--results", type=Path, required=True, help="Results produced by the sensitivity batch")
args = parser.parse_args()
for path in (
    args.database,
    args.results / "analysis_settings.json",
    args.results / "tables/estimands.csv",
    args.results / "tables/profile_support.csv",
):
    if not path.is_file():
        parser.error(f"Required input file does not exist: {path}")
O = args.results.resolve()
cfg = json.loads((O / "analysis_settings.json").read_text())
c = duckdb.connect(str(args.database.resolve()), read_only=True)
c.execute("set threads=2")
c.execute("set memory_limit='2GB'")
c.execute('''create temp table common_ids as select patent_id,primary_cpc,application_year,n,ai,ai_score,first_share from teams where n>=2 and n_unknown=0 and eligible_n>0 and ai is not null and primary_cpc is not null and relative_y is not null and team_y is not null and pair_y is not null and age_mean is not null and logq_mean is not null and width_mean is not null''')
while True:
 n=c.execute('select count(*) from common_ids').fetchone()[0]
 c.execute('''delete from common_ids where (primary_cpc,application_year) in (select primary_cpc,application_year from common_ids group by 1,2 having count(*)=1) or n in (select n from common_ids group by n having count(*)=1)''')
 if c.execute('select count(*) from common_ids').fetchone()[0]==n:break
assert n==2670332
rows=pd.read_csv(O/'tables/estimands.csv');supportrows=[];audit=[]
for early in [3,5,7]:
 for senior in [15,16,20]:
  name=f'threshold_E{early}_S{senior}';f=O/f'models/{name}.json';model=json.loads(f.read_text())
  if model.get('status')=='not_estimable':continue
  assert model['N']==n
  # Endpoint support only needs five-member FIRST=0 teams, all from the final FE sample.
  d=c.execute(f'''select t.n,t.first_share,t.ai,t.ai_score,count(*) filter(where p.age<={early})::integer n_early,count(*) filter(where p.age>={senior})::integer n_senior from common_ids t join personal p using(patent_id) where t.n=5 and t.first_share=0 group by t.patent_id,t.n,t.first_share,t.ai,t.ai_score''').df()
  info=profile_support(d,cfg);model['profile_support']=info;f.write_text(json.dumps(model,ensure_ascii=False,indent=2))
  mask=rows.model==name;rows.loc[mask,'profile_support_ok']=info['adequate_exact_support'];rows.loc[mask,'profile_minimum_N']=info['minimum_endpoint_N'];audit.append(dict(model=name,N=n,minimum_endpoint_N=info['minimum_endpoint_N'],adequate_exact_support=info['adequate_exact_support']))
  for cell in info['cells']:supportrows.append(dict(model=name,n=5,first_count=0,minimum_required_N=info['minimum_required_N'],adequate_exact_support=info['adequate_exact_support'],moderator_type=info['moderator_type'],support_kind='exact_n_FIRST_count',**cell))
old=pd.read_csv(O/'tables/profile_support.csv');old=old[~old.model.str.startswith('threshold_')];pd.concat([old,pd.DataFrame(supportrows)],ignore_index=True).to_csv(O/'tables/profile_support.csv',index=False,encoding='utf-8-sig');rows.to_csv(O/'tables/estimands.csv',index=False,encoding='utf-8-sig');(O/'threshold_support_audit.json').write_text(json.dumps(audit,indent=2));print(audit)
