"""Reproduce the 2.2 sample, certify support, then fit the two organization specifications."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '2'
import argparse
import hashlib
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import linalg

SHARED = Path(__file__).resolve().parents[1] / "_shared"
sys.path.insert(0, str(SHARED))
from study.analysis import OUTCOMES, subset, team_design, profile_vector, profile_support
from study.fe import TwoWayFE, ClusterPlan, cluster_covariance, contrast, covariance_ok, holm, nested_in
from three_fe import ThreeFE, rank_three

parser = argparse.ArgumentParser(description=__doc__)
for argument in ("database", "main-results", "anchor", "qualification", "output"):
    parser.add_argument("--" + argument, type=Path, required=True)
args = parser.parse_args()
for path in (
    args.database,
    args.anchor,
    args.qualification,
    args.main_results / "analysis_settings.json",
    args.main_results / "tables/estimands.csv",
    args.main_results / "tables/composition_support.csv",
):
    if not path.is_file():
        parser.error(f"Required input file does not exist: {path}")
MAIN = args.main_results.resolve()
CFG = json.loads((MAIN / "analysis_settings.json").read_text())
CFG["source_database"] = str(args.database.resolve())
OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=False)
FIELDS = ['ai','early_share','senior_share','age_mean','logq_mean','width_mean','first_share']
OUTCOMES_USED = OUTCOMES[:2]
SOURCES = {'ANCHOR': str(args.anchor.resolve()), 'Q_PATENT': str(args.qualification.resolve())}
GATE = dict(min_patents_per_endpoint=20, min_organizations_per_endpoint=20,
            max_single_organization_share=.5, min_jointly_varying_organizations=20,
            min_normalized_gram_eigenvalue=1e-9, max_projection_group_mean=1e-8,
            max_independent_projection_difference=1e-7, max_single_org_information_share=.5)

def save(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def labels(values):
    return pd.factorize(values, sort=True)[0].astype(np.int32)

def group_codes(d, org=False):
    groups = [labels(pd.MultiIndex.from_frame(d[['primary_cpc','application_year']])), labels(d.n)]
    if org:
        groups.append(labels(d.org))
    return groups

def support(d, by_org=False):
    base = d.n.eq(5) & d.first_share.eq(0)
    rows = []
    org_sets = {}
    for name, count in [('start',1),('end',2)]:
        for ai in (0,1):
            hit = base & d.n_early.eq(count) & d.n_senior.eq(count) & d.ai.eq(ai)
            row = dict(endpoint=name, AI=ai, patents=int(hit.sum()))
            if by_org:
                counts = d.loc[hit,'org'].value_counts()
                org_sets[name,ai] = set(counts.index)
                shares = counts / counts.sum()
                row.update(organizations=len(counts), top1=float(shares.iloc[0]) if len(shares) else 0.,
                           top5=float(shares.iloc[:5].sum()), HHI=float(np.square(shares).sum()))
            rows.append(row)
    overlap = {}
    if by_org:
        overlap = dict(all_four=len(set.intersection(*org_sets.values())),
                       both_endpoints_nonAI=len(org_sets['start',0]&org_sets['end',0]),
                       both_endpoints_AI=len(org_sets['start',1]&org_sets['end',1]),
                       both_AI_at_start=len(org_sets['start',0]&org_sets['start',1]),
                       both_AI_at_end=len(org_sets['end',0]&org_sets['end',1]),
                       both_endpoints_any_AI=len((org_sets['start',0]|org_sets['start',1])&
                                                 (org_sets['end',0]|org_sets['end',1])))
    return rows, overlap

def org_sizes(d):
    s = d.org.value_counts()
    shares = s/len(d)
    return dict(N=len(d), organizations=len(s), singleton_orgs=int(s.eq(1).sum()),
                quantiles={str(q):float(s.quantile(q)) for q in (0,.25,.5,.75,.9,.95,.99,1)},
                top1=float(shares.iloc[0]), top5=float(shares.iloc[:5].sum()),
                top10=float(shares.iloc[:10].sum()), HHI=float(np.square(shares).sum()))

def load_and_link():
    conn = duckdb.connect(CFG['source_database'],read_only=True)
    conn.execute("set threads=2; set memory_limit='1400MB'")
    columns = ['patent_id','primary_cpc','application_year','n','n_early','n_senior','ai_score']+FIELDS+OUTCOMES
    # SQL only narrows the input by conditions already enforced by formal subset().
    finite = ' and '.join(f'isfinite({x})' for x in FIELDS+OUTCOMES)
    source = conn.execute('select '+','.join(columns)+',1.0 weight from teams where n>=2 '
                          'and n_unknown=0 and eligible_n>0 and ai is not null and primary_cpc is not null and '+finite).df()
    main,_ = subset(source,FIELDS+OUTCOMES)
    n_common_before_fe = len(source)
    del source
    main.sort_values('patent_id',inplace=True,ignore_index=True)
    formal = json.loads((MAIN/'models/main_surface.json').read_text())
    assert len(main)==formal['N']==2670332
    assert main.patent_id.is_unique and main.ai.isin([0,1]).all()
    assert (main.ai.to_numpy()==(main.ai_score.to_numpy()>=CFG['ai_threshold'])).all()
    actual = main.groupby(['ai','n','n_early','n_senior','first_share'],observed=True).size().rename('N').reset_index()
    recorded = pd.read_csv(MAIN/'tables/composition_support.csv')
    pd.testing.assert_frame_equal(actual,recorded,check_dtype=False,rtol=1e-12,atol=1e-14)
    assert profile_support(main,CFG)==formal['profile_support']
    print('MAIN sample reproduced',len(main),flush=True)
    # Keep original patent IDs in the link audit. Only project relevant source columns.
    conn.register('main_ids',main[['patent_id']])
    qpath,apath = SOURCES['Q_PATENT'],SOURCES['ANCHOR']
    link = conn.execute("""
      select m.patent_id,q.patent_id is not null has_qualification,
             a.patent_id is not null has_authority,q.reliable_organization,
             a.anchor_assignee_id org,a.anchor_is_organization,
             a.anchor_missing,a.mapping_failure_n,a.anchor_selection_rule,
             a.multiple_organization_assignees,
             q.anchor_assignee_id is distinct from a.anchor_assignee_id anchor_disagrees
      from main_ids m
      left join read_parquet(?) q on m.patent_id=q.patent_id
      left join read_parquet(?) a on m.patent_id=a.patent_id
      order by m.patent_id
      """,[qpath,apath]).df()
    assert len(link)==len(main) and link.patent_id.is_unique
    assert np.array_equal(link.patent_id,main.patent_id)
    mismatch = link.has_qualification & link.has_authority & link.anchor_disagrees
    reliable = link.reliable_organization.fillna(False).astype(bool)
    inconsistent = reliable & link.has_authority & (~link.anchor_is_organization.fillna(False).astype(bool)|link.org.isna())
    conditions = [~link.has_authority,link.org.isna(),~link.anchor_is_organization.fillna(False).astype(bool),
                  ~link.has_qualification,~reliable]
    reasons = ['no_authoritative_record','authoritative_anchor_missing','anchor_is_individual_or_nonorganization',
               'no_existing_reliability_record','fails_existing_reliability_flag']
    link['status'] = np.select(conditions,reasons,default='linked')
    link['ai'] = main.ai.to_numpy()
    breakdown = link.groupby(['status','ai'],observed=True).size().rename('N').reset_index()
    breakdown.to_csv(OUT/'link_status_by_ai.csv',index=False)
    identity = dict(main_N=len(main),pre_base_FE=n_common_before_fe,authority_disagreements=int(mismatch.sum()),
                    inconsistent_reliable_anchors=int(inconsistent.sum()), duplicate_link_rows=len(link)-link.patent_id.nunique())
    save('link_identity.json',identity)
    assert not mismatch.any() and not inconsistent.any(),identity
    sel = link.status.eq('linked')
    d = main.loc[sel].copy()
    d['org'] = link.loc[sel,'org'].to_numpy()
    linked = org_sizes(d)
    rates = link.groupby('ai').status.agg(N='size',linked=lambda x:x.eq('linked').sum()).reset_index()
    rates['rate']=rates.linked/rates.N
    linked.update(rate=len(d)/len(main), rates=rates.to_dict('records'),
                  multi_org_patents=int(link.loc[sel,'multiple_organization_assignees'].sum()))
    link_initial, link_overlap = support(d,True)
    masks = group_codes(d,True)
    keep = np.ones(len(d),bool)
    rounds = []
    while True:
        singles = [(np.bincount(g[keep],minlength=int(g.max())+1)[g]==1)&keep for g in masks]
        remove = np.logical_or.reduce(singles)
        rounds.append(dict(round=len(rounds)+1,before=int(keep.sum()),deleted=int(remove.sum()),
                           after=int((keep&~remove).sum()), singleton_by_FE=[int(x.sum()) for x in singles]))
        if not remove.any():
            break
        keep &= ~remove
    d = d.loc[keep].reset_index(drop=True)
    assert len(d)>0
    g = group_codes(d,True)
    link['final'] = link.patent_id.isin(d.patent_id)
    conn.register('link_audit',link[['patent_id','ai','org','status','final']])
    conn.execute("copy link_audit to '"+str(OUT/'patent_sample_audit.parquet').replace("'","''")+"' (format parquet, compression zstd)")
    conn.close()
    eps, overlap = support(d,True)
    gp = d.groupby('org',sort=True)
    ai_vary = gp.ai.nunique().gt(1)
    design = team_design(d,'surface')
    variation = {}
    for j, name in enumerate(design.names):
        if j<=10:
            v = pd.Series(design.functions[j]()).groupby(d.org).agg(['min','max'])
            variation[name] = int((v['max']-v['min']>1e-12).sum())
    e_vary = gp.early_share.nunique().gt(1)
    s_vary = gp.senior_share.nunique().gt(1)
    joint = ai_vary & (e_vary|s_vary)
    audit = dict(identity=identity, linked=linked, link_status=breakdown.to_dict('records'),
                 rounds=rounds, final=org_sizes(d), FE_levels=[int(x.max())+1 for x in g],
                 CPC_clusters=d.primary_cpc.nunique(), full_endpoints=support(main)[0],
                 linked_endpoints=link_initial, linked_endpoint_overlap=link_overlap,
                 final_endpoints=eps, final_endpoint_overlap=overlap, raw_organization_variation=variation,
                 AI_and_either_share=int(joint.sum()), AI_and_both_shares=int((ai_vary&e_vary&s_vary).sum()),
                 patents_in_jointly_varying_organizations=int(d.org.isin(joint[joint].index).sum()),
                 final_ai_counts={str(k):int(v) for k,v in d.ai.value_counts().items()})
    save('sample_support.json',audit)
    assert all(x['patents']>=GATE['min_patents_per_endpoint'] and
               x['organizations']>=GATE['min_organizations_per_endpoint'] and
               x['top1']<GATE['max_single_organization_share'] for x in eps), 'Endpoint gate failed'
    assert joint.sum()>=GATE['min_jointly_varying_organizations'], 'Organization variation gate failed'
    print('Link and endpoint gates passed',len(d),audit['final']['organizations'],flush=True)
    return main,d,g

def project_checked(fe, z, groups, independent, scale=1.):
    z=np.asarray(z,dtype=float)
    r = fe.project(z)[:,0]
    ri = fe.project(z,independent)[:,0]
    diff = float(np.max(np.abs(r-ri))*scale)
    err = orthogonal(r,groups)*scale
    assert diff<=GATE['max_independent_projection_difference'] and err<=GATE['max_projection_group_mean'],(diff,err)
    return r,dict(independent_difference=diff, group_mean_error=err)

def orthogonal(r,groups):
    return max(float(np.max(np.abs(np.bincount(g,weights=r)/np.bincount(g)))) for g in groups)

def prepare(d,g,model,fe,independent,rank):
    design = team_design(d,'surface')
    scales=np.array(design.scales)
    target=profile_vector(design.names,CFG)
    n,p=len(d),len(scales)
    rx=np.lib.format.open_memmap(OUT/f'tmp_{model}_RX.npy',mode='w+',dtype='float64',shape=(n,p),fortran_order=True)
    checks={}
    for j,fn in enumerate(design.functions):
        rx[:,j],checks[design.names[j]]=project_checked(fe,fn()/scales[j],g,independent,scales[j])
        print(model,'X',j+1,design.names[j],flush=True)
    gram=rx.T@rx
    sd=np.sqrt(np.diag(gram))
    assert np.all(sd>1e-12),'Absorbed regressors'
    corr=gram/np.outer(sd,sd)
    eigenvalues=linalg.eigvalsh(corr)
    assert eigenvalues[0]>GATE['min_normalized_gram_eigenvalue'],('Rank/condition gate',eigenvalues[0])
    bread=linalg.cho_solve(linalg.cho_factor(corr),np.eye(p))/np.outer(sd,sd)
    cp=labels(d.primary_cpc)
    nested=[nested_in(x,cp) for x in g]
    assert nested==[True]+[False]*(len(g)-1),('Unexpected nesting pattern',nested)
    # Extend the 2.2 union-span-minus-nested-span convention to the third block.
    counted=rank-(int(g[0].max())+1)
    k=p+counted
    G=int(cp.max())+1
    factor=G/(G-1)*(n-1)/(n-k)
    assert G>1 and n>k
    plan=ClusterPlan([dict(codes=cp,groups=G,sign=1,factor=factor,margins=[0])],G-1,k,n,counted)
    info=dict(N=n,p=p,FE_rank=rank,design_rank=p,full_rank=rank+p,
              eigen_min=float(eigenvalues[0]), condition=float(eigenvalues[-1]/eigenvalues[0]),
              K=k,FE_rank_counted=counted,clusters=G,df=G-1,CRV1_factor=factor,nested=[bool(x) for x in nested],
              projection_checks=checks,residual_column_SS=dict(zip(design.names,(np.diag(gram)*scales**2).tolist())))
    direction = rx@(bread@(target/scales))
    sq=np.square(direction)
    cluster_mass=np.bincount(cp,weights=sq)/sq.sum()
    info['contrast_design_top_CPC_share']=float(cluster_mass.max())
    if model=='M1':
        mass=np.bincount(g[2],weights=sq)/sq.sum()
        info.update(contrast_design_top_org_share=float(mass.max()),
                    contrast_design_top5_org_share=float(np.sort(mass)[-5:].sum()),
                    contrast_design_effective_orgs=float(1/(mass@mass)))
        assert mass.max()<GATE['max_single_org_information_share'],'Contrast information concentration gate failed'
    save(f'{model}_design.json',info)
    return rx,scales,bread,target,plan,info

def fit(d,g,model,fe,independent,prepared):
    rx,scales,bread,target,plan,info=prepared
    w=np.ones(len(d))
    output={}
    for outcome in OUTCOMES_USED:
        ry,check=project_checked(fe,d[outcome].to_numpy(),g,independent)
        bs=bread@(rx.T@ry)
        residual=ry-rx@bs
        normal=float(np.max(np.abs(rx.T@residual)/
                           (np.sqrt(np.diag(rx.T@rx))*max(1.,np.linalg.norm(residual)))))
        err=orthogonal(residual,g)
        assert normal<1e-8 and err<1e-8,(normal,err)
        v=cluster_covariance(rx,residual,w,bread,plan)/np.outer(scales,scales)
        assert covariance_ok(v)
        beta=bs/scales
        c=contrast(beta,v,target,plan.df)
        assert c['inference_ok']
        output[outcome]=dict(beta=beta.tolist(),V=v.tolist(),contrast=c,projection=check,
                             normal_equation=normal,residual_FE_error=err)
    print(model,{y:v['contrast'] for y,v in output.items()},flush=True)
    return output

def fixture():
    rng=np.random.default_rng(502)
    gg=[labels(rng.integers(0,k,400)) for k in (9,6,23)]
    zz=rng.normal(size=(400,4))
    dense=np.column_stack([np.eye(g.max()+1)[g] for g in gg])
    folder=OUT/'fixture'
    folder.mkdir(exist_ok=True)
    r=rank_three(gg,folder)
    assert r['union_rank']==np.linalg.matrix_rank(dense)
    fe=ThreeFE(gg,np.ones(len(zz)),r)
    ref=zz-dense@np.linalg.lstsq(dense,zz,rcond=None)[0]
    err=float(np.max(np.abs(fe.project(zz)-ref)))
    independent=float(np.max(np.abs(fe.project(zz,'independent')-ref)))
    assert max(err,independent)<1e-10
    save('three_fe_fixture.json',dict(rank=r['union_rank'],dense_rank=int(np.linalg.matrix_rank(dense)),
                                     LSQR_max_error=err,LSMR_max_error=independent))

def run():
    save('config.json',dict(main=CFG,sources={k:SOURCES[k] for k in ('ANCHOR','Q_PATENT')},
                            gate=GATE,targets=OUTCOMES_USED,holm_family='ORG_M1_TWO_OUTCOMES',
                            ssc='Union FE rank minus nested FE span, plus slope rank',
                            source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in (Path(__file__),SHARED/'three_fe.py',
                                                     SHARED/'study/analysis.py',SHARED/'study/fe.py',
                                                     MAIN/'models/main_surface.json',MAIN/'tables/estimands.csv')}))
    main,d,g=load_and_link()
    fixture()
    rank=rank_three(g,OUT)
    fe1=ThreeFE(g,np.ones(len(d)),rank)
    prep1=prepare(d,g,'M1',fe1,'independent',rank['union_rank'])
    save('stage1_gate.json',dict(status='PASS',gates=GATE,stage='before outcome fitting'))
    print('Stage 1 PASS before outcome fitting',flush=True)
    gm=group_codes(main)
    fm=TwoWayFE(gm,np.ones(len(main)))
    pm=prepare(main,gm,'FULL_REPRO',fm,'svd',fm.rank)
    repro=fit(main,gm,'FULL_REPRO',fm,'svd',pm)
    saved=json.loads((MAIN/'models/main_surface.json').read_text())
    comparisons={}
    for i,y in enumerate(OUTCOMES_USED):
        got=repro[y];want=saved['outcomes'][i]
        np.testing.assert_allclose(got['beta'],want['beta'],atol=1e-9,rtol=1e-7)
        np.testing.assert_allclose(got['V'],want['covariance']['CPC']['V'],atol=1e-10,rtol=1e-6)
        comparisons[y]=dict(max_abs_beta=float(np.max(np.abs(np.array(got['beta'])-want['beta']))),
                            max_abs_V=float(np.max(np.abs(np.array(got['V'])-want['covariance']['CPC']['V']))))
    assert pm[-1]['K']==saved['outcomes'][0]['covariance']['CPC']['K'] and pm[-1]['FE_rank']==saved['FE_rank']
    save('full_reproduction.json',comparisons)
    del pm,main,fm
    (OUT/'tmp_FULL_REPRO_RX.npy').unlink()
    f0=TwoWayFE(g[:2],np.ones(len(d)))
    p0=prepare(d,g[:2],'M0',f0,'svd',f0.rank)
    m0=fit(d,g[:2],'M0',f0,'svd',p0)
    m1=fit(d,g,'M1',fe1,'independent',prep1)
    ids_hash=hashlib.sha256(pd.util.hash_pandas_object(d.patent_id,index=False).to_numpy().tobytes()).hexdigest()
    w_hash=hashlib.sha256(np.ones(len(d)).tobytes()).hexdigest()
    assert d.patent_id.is_unique and len(d)==p0[-1]['N']==prep1[-1]['N']
    rows=[]
    formal=pd.read_csv(MAIN/'tables/estimands.csv')
    adjusted=holm([m1[y]['contrast']['p_raw'] for y in OUTCOMES_USED])
    for i,y in enumerate(OUTCOMES_USED):
        original=formal[(formal.outcome==y)&(formal.contrast=='profile_AI_difference')].iloc[0]
        rows.append(dict(outcome=y,model='FULL_M0',N=int(original.N),**{k:original[k] for k in
                         ('estimate','SE','CI_low','CI_high','p_raw','p_Holm','df')},family='MAIN_7'))
        for model,res,pa in [('M0',m0,None),('M1',m1,adjusted[i])]:
            rows.append(dict(outcome=y,model=model,N=len(d),**res[y]['contrast'],p_Holm=pa,
                             family='ORG_M1_TWO_OUTCOMES' if model=='M1' else None))
    pd.DataFrame(rows).to_csv(OUT/'organization_contrasts.csv',index=False)
    save('organization_models.json',dict(M0=m0,M1=m1,final_patent_id_hash=ids_hash,
                                         weight_hash=w_hash,N=len(d),weight=1,
                                         terms=team_design(d).names,target=prep1[3].tolist()))
    save('execution_status.json',dict(status='PASS',stage='complete'))
    for path in OUT.glob('tmp_*_RX.npy'):
        path.unlink()

if __name__=='__main__':
    try:
        run()
    except Exception as error:
        save('execution_status.json',dict(status='STOP',error=repr(error),stage1_completed=(OUT/'stage1_gate.json').exists()))
        raise
