from pathlib import Path
import duckdb,json,csv,math
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "03_novelty_indicators"))
from peer_history_sql import exposure_sql,source_sql,history_sql,BUILD,POLICY
O=Path(__file__).resolve().parent;c=duckdb.connect();c.execute('SET threads=1');c.execute("SET memory_limit='1GB'")
# Fixtures use eight codes for transparent independent set arithmetic.
def actor(stage='N',H=(),state='EXISTING_VALID_HISTORY',gap=False):return dict(stage=stage,H=set(H),state=state,gap=gap)
N=lambda H=(),gap=False:actor('N',H,gap=gap)
M=lambda H=(),gap=False:actor('M',H,gap=gap)
V=lambda H=(),gap=False:actor('V',H,gap=gap)
F=lambda:actor('N',(),state='FIRST_OBSERVED')
I=lambda:actor('UNKNOWN',(),state='HISTORY_INCOMPLETE')
P=lambda:actor('M',(),state='PRIOR_WITHOUT_VALID_CPC')
fixtures=[('ALL_FIRST',[F(),F()],{0}),('FIRST_AND_EXISTING',[F(),M({0})],{0,1}),('NEW_AND_VETERAN',[N({0}),V({1})],{1,2}),('N_M_V',[N({0}),M({1}),V({2})],{0,1,3}),('PRIOR_CPC_EMPTY',[N({0}),P()],{1}),('UNKNOWN_PEER',[N({0}),I()],{1}),('DUPLICATE_SHARED_CODES',[N({0,1}),N({1,2}),M({3})],{2,3,4}),('NO_EARLY',[M({0}),V({1})],{1,2}),('BOTH_PEER_SOURCES',[M({0}),N({1}),V({1})],{1,2}),('EARLY_ONLY',[M({0}),N({1}),V({2})],{1,3}),('OA_ZERO',[M({0}),N({0})],{1}),('NEW_ZERO_OA_POSITIVE',[M({0}),N({1})],{0}),('SINGLETON',[M({0})],{1}),('PARTIAL_CPC_GAP',[M({0},True),N({1})],{1}),('CPC_EMPTY_CURRENT',[M({0}),N({1})],set()),('UTILIZATION_ONE',[M({0}),N({1})],{1}),('UNKNOWN_FOCAL_DESCRIPTIVE_E',[I(),N({1})],{1}),('AI_MISSING',[M({0}),N({1})],{1}),('OVER70_VALID_V',[V({0}),N({1})],{1}),('LARGE_TEAM_1000',[N({0}) if j%2==0 else V({1}) for j in range(1000)],{0,1,2})]
c.execute('CREATE TABLE fm(patent_id VARCHAR,inventor_id VARCHAR,career_stage VARCHAR,history_state VARCHAR,eligible_existing_Y BOOLEAN,personal_Y DOUBLE,current_cpc_count INTEGER,prior_cpc_count INTEGER,new_cpc_count INTEGER,prior_cpc_gap BOOLEAN)')
c.execute('CREATE TABLE ft(patent_id VARCHAR,n_full INTEGER,n_newcomer INTEGER,n_middle INTEGER,n_veteran INTEGER,n_unknown INTEGER,first_observed_count INTEGER,m_search INTEGER,team_existing_Y DOUBLE,ibi_n5v16 DOUBLE,ai_aipat50 INTEGER,history_incomplete_count INTEGER,prior_without_valid_cpc_count INTEGER)')
c.execute('CREATE TABLE fh(patent_id VARCHAR,inventor_id VARCHAR,history_blob BLOB,encoded_prior_cpc_count INTEGER)');c.execute('CREATE TABLE fc(patent_id VARCHAR,current_blob BLOB)')
def blob(S):return bytes([sum(1<<(7-k) for k in S)])
for pid,actors,cp in fixtures:
 rows=[];hs=[];ys=[]
 for k,x in enumerate(actors):
  good=x['state']=='EXISTING_VALID_HISTORY' and bool(x['H']) and bool(cp);y=len(cp-x['H'])/len(cp) if good else None
  rows.append((pid,str(k),x['stage'],x['state'],good,y,len(cp),len(x['H']),len(cp-x['H']),x['gap']));hs.append((pid,str(k),blob(x['H']),len(x['H'])))
  if y is not None:ys.append(y)
 cnt=lambda s:sum(x['stage']==s for x in actors);n=len(actors);unknown=cnt('UNKNOWN');ibi=cnt('N')*cnt('V')/(n*(n-1)/2) if n>1 and unknown==0 else None
 c.executemany('INSERT INTO fm VALUES (?,?,?,?,?,?,?,?,?,?)',rows);c.executemany('INSERT INTO fh VALUES (?,?,?,?)',hs)
 c.execute('INSERT INTO ft VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',[pid,n,cnt('N'),cnt('M'),cnt('V'),unknown,sum(x['state']=='FIRST_OBSERVED' for x in actors),len(ys),sum(ys)/len(ys) if ys else None,ibi,None if pid=='AI_MISSING' else 1,sum(x['state']=='HISTORY_INCOMPLETE' for x in actors),sum(x['state']=='PRIOR_WITHOUT_VALID_CPC' for x in actors)])
 c.execute('INSERT INTO fc VALUES (?,?)',[pid,blob(cp)])
c.execute('CREATE TABLE fe AS '+exposure_sql('fm','ft'));cur=c.execute(source_sql('fe','fh','fc',8));names=[x[0] for x in cur.description];actual={(r[0],r[1]):dict(zip(names,r)) for r in cur.fetchall()}
checks=[]
def test(case,member,field,expected,got):
 ok=(expected is None and got is None) or (expected is not None and got is not None and (abs(float(expected)-float(got))<1e-12 if isinstance(expected,(float,int,bool)) else expected==got))
 checks.append(dict(case=case,inventor_id=member,field=field,expected=expected,actual=got,status='PASS' if ok else 'FAIL'))
for pid,actors,cp in fixtures:
 for k,x in enumerate(actors):
  a=actual[(pid,str(k))];peers=actors[:k]+actors[k+1:];n=len(actors);eok=n>1 and all(p['stage']!='UNKNOWN' for p in peers);yok=x['state']=='EXISTING_VALID_HISTORY' and bool(x['H']) and bool(cp)
  sok=eok and yok and all(p['state'] in ('FIRST_OBSERVED','EXISTING_VALID_HISTORY') for p in peers)
  HE=set().union(*(p['H'] for p in peers if p['stage']=='N'));HO=set().union(*(p['H'] for p in peers if p['stage'] in ('M','V')))
  J=cp-x['H'];A=J&HE-HO;B=J&HO-HE;C=J&HE&HO;D=J-(HE|HO);OA=(HE-HO)-x['H'];adopt=len(cp&OA)
  vals={'E_descriptive_eligible':eok,'E_early_peer_share':sum(p['stage']=='N' for p in peers)/(n-1) if eok else None,'first_peer_share':sum(p['state']=='FIRST_OBSERVED' for p in peers)/(n-1) if eok else None,'source_main_eligible':sok,'source_strict_peer_eligible':sok and all(p['state']=='EXISTING_VALID_HISTORY' for p in peers),'source_strict_CPC_eligible':sok and not any(p['gap'] for p in actors),'P1_basic_variable_candidate':eok and yok and pid!='AI_MISSING','n_full':n,'early_peer_count':sum(p['stage']=='N' for p in peers),'unknown_peer_count':sum(p['stage']=='UNKNOWN' for p in peers),'early_only_opportunity_n':len(OA) if sok else None,'early_only_adopted_n':adopt if sok else None,'early_only_utilization':adopt/len(OA) if sok and OA else None}
  for cat,S in [('A',A),('B',B),('C',C),('D',D)]:
   vals[cat+'_count']=len(S) if sok else None;vals[cat+'_share']=len(S)/len(cp) if sok else None
  for field,value in vals.items():test(pid,str(k),field,value,a[field])
  if sok:
   test(pid,str(k),'partition_sum',len(J)/len(cp),sum(a[z+'_share'] for z in 'ABCD'));test(pid,str(k),'adopt_equals_A',a['A_count'],a['early_only_adopted_n'])
   test(pid,str(k),'peer_history_excludes_focal_retains_shared',blob(HE),a['early_peer_history_blob'])
# Same production history function, first occurrence dates around a focal batch and pre-2000.
c.execute('CREATE TABLE code_dictionary(cpc_subclass VARCHAR,code_index INTEGER)');c.execute("INSERT INTO code_dictionary VALUES ('A01B',0),('A01C',1),('A01D',2),('A01F',3)")
c.execute('CREATE TABLE hf(inventor_id VARCHAR,first_filing_date DATE,cpc_subclass VARCHAR)');c.execute("INSERT INTO hf VALUES ('i','1990-01-01','A01B'),('i','2000-01-01','A01C'),('i','2000-01-01','A01D'),('i','2000-01-02','A01F')")
c.execute('CREATE TABLE hm(patent_id VARCHAR,inventor_id VARCHAR,filing_date DATE)');c.execute("INSERT INTO hm VALUES ('same1','i','2000-01-01'),('same2','i','2000-01-01'),('next','i','2000-01-02'),('later','i','2000-01-03')")
for pid,inv,b,n in c.execute(history_sql('hm','hf',8)).fetchall():
 S={'same1':{0},'same2':{0},'next':{0,1,2},'later':{0,1,2,3}}[pid];test('STRICT_PRIOR_SAME_DAY_PRE2000',pid,'history_blob',blob(S),b);test('STRICT_PRIOR_SAME_DAY_PRE2000',pid,'history_count',len(S),n)
result={'policy':POLICY,'build':BUILD,'fixture_patents':len(fixtures),'fixture_members':sum(len(a) for _,a,_ in fixtures),'history_batch_cases':4,'comparisons':len(checks),'failure_count':sum(r['status']=='FAIL' for r in checks),'case_summary':{case:{'comparisons':sum(r['case']==case for r in checks),'failures':sum(r['case']==case and r['status']=='FAIL' for r in checks)} for case in sorted(set(r['case'] for r in checks))}}
print(json.dumps(result)); assert result['failure_count']==0
