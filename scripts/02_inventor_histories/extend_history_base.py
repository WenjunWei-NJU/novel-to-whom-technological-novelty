from pathlib import Path
import duckdb,json,time
import argparse
import shutil

parser = argparse.ArgumentParser(description="Build rolling histories and team age distance in a new copy of the analytical database.")
for argument in ("database", "history-database", "cpc-dictionary", "current-cpc-sets", "output"):
    parser.add_argument("--" + argument, type=Path, required=True)
args = parser.parse_args()
for path in (args.database, args.history_database, args.cpc_dictionary, args.current_cpc_sets):
    if not path.is_file():
        parser.error(f"Input does not exist: {path}")
if Path(str(args.database) + ".wal").exists():
    parser.error("Checkpoint the analytical database before copying it.")

O = args.output.resolve()
O.mkdir(parents=True, exist_ok=False)
shutil.copyfile(args.database, O / "core.duckdb")
c = duckdb.connect(str(O / "core.duckdb"))
c.execute("set threads=2")
c.execute("set preserve_insertion_order=false")
c.execute("set memory_limit='3GB'")
def lit(p):return "'"+str(p).replace("'","''")+"'"
c.execute("set temp_directory="+lit(O/"tmp"))
c.execute('attach '+lit(args.history_database.resolve())+' as hist (read_only)')
c.execute('create or replace table window_dictionary as select cpc_subclass,code_index::integer code from read_csv_auto('+lit(args.cpc_dictionary.resolve())+')')
codes=dict(c.execute('select cpc_subclass,code from window_dictionary').fetchall())
def encode_set(values):
 mask=0
 for code in values:mask|=1<<(679-codes[code])
 return format(mask,'0680b')
c.create_function('encode_cpc_set',encode_set,[duckdb.list_type('VARCHAR')],'VARCHAR')
c.execute('create or replace table current_bits as select patent_id,encode_cpc_set(current_cpc_set)::BIT bits from read_parquet('+lit(args.current_cpc_sets.resolve())+')')

# Reuse complete lifetime ages; pairwise absolute distance via sorted rank identity.
c.execute('''create or replace table age_distance as select patent_id,2*sum((2*rn-n-1)*age)/(max(n)*(max(n)-1)) mean_age_distance from (select patent_id,age,n,row_number() over(partition by patent_id order by age,inventor_id) rn from personal) where n>=2 and age is not null group by patent_id''')
c.execute('alter table teams add column if not exists mean_age_distance double')
c.execute('update teams set mean_age_distance=a.mean_age_distance from age_distance a where teams.patent_id=a.patent_id and teams.n_unknown=0')
log=[]
for shard in range(16):
 start=time.time()
 c.execute(f'''create or replace table window_days as select e.inventor_id,e.filing_date,bit_or(coalesce(b.bits,bitstring('0',680))) bits from hist.events e left join current_bits b using(patent_id) where e.shard={shard} group by 1,2''')
 c.execute('''create or replace table window_history as select inventor_id,filing_date,bit_or(bits) over(partition by inventor_id order by filing_date range between interval '5 years' preceding and interval '1 day' preceding) h5,bit_or(bits) over(partition by inventor_id order by filing_date range between interval '10 years' preceding and interval '1 day' preceding) h10 from window_days''')
 c.execute(f'''create or replace table window_members as select p.patent_id,p.inventor_id,p.y,p.y is not null existing,p.stage<>'U' known,p.prior_patents,b.bits current_bits,coalesce(h.h5,bitstring('0',680)) h5,coalesce(h.h10,bitstring('0',680)) h10 from personal p join hist.events e using(patent_id,inventor_id) join window_history h on p.inventor_id=h.inventor_id and e.filing_date=h.filing_date left join current_bits b using(patent_id) where e.shard={shard}''')
 for w in [5,10]:
  c.execute(f'alter table window_members add column y{w} double')
  c.execute(f'update window_members set y{w}=bit_count(current_bits & ~h{w})::double/bit_count(current_bits) where existing and bit_count(h{w})>0')
 dest=O/f'window_members_{shard:02d}.parquet';c.execute('copy window_members to '+lit(dest)+' (format parquet,compression zstd)');log.append({'shard':shard,'seconds':round(time.time()-start,2)});print(log[-1],flush=True)
c.execute('create or replace temp view all_window_members as select * exclude(current_bits,h5,h10),current_bits::BIT current_bits,h5::BIT h5,h10::BIT h10 from read_parquet('+lit(O/'window_members_*.parquet')+')')
parts=[]
for w in [5,10]:
 parts.extend([f'avg(y{w}) relative_y{w}',f'avg(y) filter(where y{w} is not null) relative_full_on_w{w}',f'count(y{w}) eligible_n{w}',f"case when bool_and(known and (prior_patents=0 or bit_count(h{w})>0)) and max(bit_count(current_bits))>0 then bit_count(any_value(current_bits) & ~bit_or(h{w}))::double/max(bit_count(current_bits)) end team_y{w}"])
c.execute('create or replace table window_teams as select patent_id,'+','.join(parts)+' from all_window_members group by 1')
for w in [5,10]:
 for name,kind in [(f'relative_y{w}','double'),(f'relative_full_on_w{w}','double'),(f'eligible_n{w}','bigint'),(f'team_y{w}','double'),(f'team_full_on_w{w}','double')]:
  c.execute(f'alter table teams add column if not exists {name} {kind}')
 c.execute(f'update teams set relative_y{w}=x.relative_y{w},relative_full_on_w{w}=x.relative_full_on_w{w},eligible_n{w}=x.eligible_n{w},team_y{w}=x.team_y{w},team_full_on_w{w}=case when x.team_y{w} is not null then teams.team_y end from window_teams x where teams.patent_id=x.patent_id')
c.execute('copy teams to '+lit(O/'teams.parquet')+' (format parquet,compression zstd)')
(O/'window_build.json').write_text(json.dumps(log,indent=2))

c.close()
