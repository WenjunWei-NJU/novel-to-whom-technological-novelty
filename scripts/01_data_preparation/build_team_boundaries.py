"""Reuse full D2B roster; derive team-history novelty without rebuilding histories."""
from pathlib import Path
import json,time
import pyarrow as pa
import pyarrow.parquet as pq
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--roster-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
SRC = args.roster_dir.resolve()
shards = sorted(SRC.glob("part-*.parquet"))
if not shards:
    parser.error("roster directory must contain part-*.parquet")
OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=False)

columns=['patent_id','history_blob','current_blob','history_state','personal_Y','career_age_years','prior_patent_count','prior_cpc_count','prior_cpc_gap']
report=[]
for f in shards:
    teams={};start=time.time()
    for batch in pq.ParquetFile(f).iter_batches(batch_size=65536,columns=columns):
        cols=[batch.column(i).to_pylist() for i in range(len(columns))]
        for pat,h,c,state,y,age,q,width,gap in zip(*cols):
            rec=teams.setdefault(pat,[0,0,0,True,0.,0,0.,0.,0.,0,False])
            hb=int.from_bytes(h,'big') if h is not None else 0
            cb=int.from_bytes(c,'big') if c is not None else 0
            if rec[0] and rec[2]!=cb:raise ValueError('Inconsistent current CPC set')
            rec[0]+=1;rec[1]|=hb;rec[2]=cb
            if state not in ('FIRST','FIRST_OBSERVED','EXISTING_VALID_HISTORY','PRIOR_WITHOUT_VALID_CPC','HISTORY_INCOMPLETE'):raise ValueError('Unknown history state: '+str(state))
            rec[3]&=state in ('FIRST','FIRST_OBSERVED','EXISTING_VALID_HISTORY')
            if y is not None:rec[4]+=y;rec[5]+=1
            if age is not None:
                import math
                rec[6]+=age;rec[7]+=math.log1p(q);rec[8]+=width;rec[9]+=1
            rec[10]|=bool(gap)
    rows=[]
    for pat,(n,h,c,valid,ys,m,ages,qs,widths,nage,gap) in teams.items():
        cn=c.bit_count();yt=(c&~h).bit_count()/cn if valid and cn else None
        yr=ys/m if m else None
        if yt is not None and yr is not None and yt>yr+1e-12:raise ValueError('Team novelty exceeds individual mean')
        rows.append(dict(patent_id=pat,n=n,history_usable=valid,relative_y=yr,team_y=yt,peer_covered_y=yr-yt if yr is not None and yt is not None else None,eligible_n=m,cpc_n=cn,age_mean=ages/n if nage==n else None,logq_mean=qs/n if nage==n else None,width_mean=widths/n if nage==n else None,any_history_gap=gap))
    dest=OUT/f'team_boundaries_{f.stem}.parquet'
    schema=pa.schema([('patent_id',pa.int64()),('n',pa.int64()),('history_usable',pa.bool_()),('relative_y',pa.float64()),('team_y',pa.float64()),('peer_covered_y',pa.float64()),('eligible_n',pa.int64()),('cpc_n',pa.int64()),('age_mean',pa.float64()),('logq_mean',pa.float64()),('width_mean',pa.float64()),('any_history_gap',pa.bool_())])
    pq.write_table(pa.Table.from_pylist(rows,schema=schema),dest,compression='zstd')
    report.append(dict(source=str(f),output=str(dest),teams=len(rows),members=sum(x[0]for x in teams.values()),seconds=round(time.time()-start,2)))
    print(f.name,len(rows),flush=True)
(OUT/'team_boundary_build.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
