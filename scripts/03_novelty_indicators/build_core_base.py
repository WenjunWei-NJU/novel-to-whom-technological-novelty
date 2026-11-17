from pathlib import Path
import duckdb,json
import argparse
import glob

parser = argparse.ArgumentParser(description="Assemble the formal team and inventor analytical tables from saved inputs.")
for argument in ("team-boundaries", "team-history", "primary-cpc", "pair-history", "personal-candidates", "output"):
    parser.add_argument("--" + argument, required=True)
args = parser.parse_args()
for pattern in (args.team_boundaries, args.team_history, args.primary_cpc,
                args.pair_history, args.personal_candidates):
    if not glob.glob(pattern):
        parser.error(f"Required input file or pattern has no matches: {pattern}")
O = Path(args.output).resolve()
O.mkdir(parents=True, exist_ok=False)
c = duckdb.connect(str(O / "core.duckdb"))
c.execute("set threads=2")
c.execute("set memory_limit='3GB'")
def view(name,path):
    literal="'"+str(path).replace("'","''")+"'"
    c.execute(f'create or replace temp view {name} as select * from read_parquet({literal})')
c.execute("set temp_directory='" + str(O / "tmp").replace("'", "''") + "'")
view("boundary", args.team_boundaries)
view("oldteam", args.team_history)
view("field", args.primary_cpc)
view("pairs", args.pair_history)
c.execute('''create or replace table teams as select b.*,t.filing_date,t.grant_date,t.application_year,t.ai_aipat50::integer ai,t.aipat_ai_score ai_score,f.cpc_subclass primary_cpc,t.n_newcomer n_early,t.n_veteran n_senior,t.n_unknown,t.first_observed_count::double/b.n first_share,t.n_newcomer::double/b.n early_share,t.n_veteran::double/b.n senior_share,t.ibi_n5v16 ibi,p.K pair_n,p.new_count new_pairs,p.new_pairs_exposed_to_undated uncertain_new_pairs,case when p.K>0 then p.new_count::double/p.K end pair_y from boundary b join oldteam t using(patent_id) left join field f using(patent_id) left join pairs p using(patent_id)''')
checks={}
for label,sql in {'duplicate_team':'select count(*)-count(distinct patent_id) from teams','team_identity':'select count(*) from teams where abs(relative_y-team_y-peer_covered_y)>1e-12','ordering':'select count(*) from teams where team_y>relative_y+1e-12','old_relative_difference':'select count(*) from teams n join oldteam o using(patent_id) where abs(n.relative_y-o.team_existing_Y)>1e-12','n_difference':'select count(*) from teams n join oldteam o using(patent_id) where n.n<>o.n_full'}.items():
 checks[label]=c.execute(sql).fetchone()[0]
assert not any(checks.values()),checks
checks['teams']=c.execute('select count(*) from teams').fetchone()[0]
checks['common_pre_FE']=c.execute('select count(*) from teams where n>=2 and n_unknown=0 and ai is not null and primary_cpc is not null and relative_y is not null and team_y is not null and pair_y is not null and age_mean is not null and logq_mean is not null and width_mean is not null').fetchone()[0]
c.execute("copy teams to '"+str(O/'teams.parquet')+"' (format parquet,compression zstd)")
(O/'core_base_checks.json').write_text(json.dumps(checks,indent=2));print(checks,flush=True)
view("candidate", args.personal_candidates)
c.execute('''create or replace table personal as with base as (select v.patent_id,v.inventor_id,t.primary_cpc,v.application_year,v.n_full n,t.ai,t.ai_score,v.E_early_peer_share early_peers,v.veteran_peer_share senior_peers,v.first_peer_share first_peers,v.career_age_years age,v.prior_patent_count prior_patents,v.prior_cpc_count history_width,v.personal_Y y,case v.career_stage when 'N' then 'E' when 'V' then 'S' when 'M' then 'M' else 'U' end stage,v.A_share A,v.B_share B,v.C_share C,v.D_share D,v.early_only_opportunity_n opportunity_n,v.early_only_utilization utilization,v.P1_basic_variable_candidate p_ok,v.source_basic_variable_candidate s_ok,(v.utilization_eligible and t.ai is not null and v.P1_basic_variable_candidate) u_ok from candidate v join teams t using(patent_id)) select * exclude(p_ok,s_ok,u_ok),case when p_ok then 1.0/count(*) filter(where p_ok) over(partition by patent_id) end weight,case when s_ok then 1.0/count(*) filter(where s_ok) over(partition by patent_id) end source_weight,case when u_ok then 1.0/count(*) filter(where u_ok) over(partition by patent_id) end utilization_weight from base''')
checks['personal_rows']=c.execute('select count(*) from personal').fetchone()[0]
checks['personal_duplicates']=c.execute('select count(*)-count(distinct (patent_id,inventor_id)) from personal').fetchone()[0]
for weight in ['weight','source_weight','utilization_weight']:
 checks[weight+'_sum_violations']=c.execute(f'select count(*) from (select patent_id,sum({weight}) s from personal where {weight} is not null group by patent_id) where abs(s-1)>1e-10').fetchone()[0]
assert checks['personal_duplicates']==0
assert all(checks[w+'_sum_violations']==0 for w in ['weight','source_weight','utilization_weight'])
c.execute("copy personal to '"+str(O/'inventor_patents.parquet')+"' (format parquet,compression zstd)")
(O/'core_base_checks.json').write_text(json.dumps(checks,indent=2));print(checks,flush=True)

c.close()
