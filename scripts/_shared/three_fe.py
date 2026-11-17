import numpy as np
from study.io import save_json as save

def log(message):
    print(message, flush=True)

from scipy.sparse import csr_matrix,bmat
from scipy.sparse.linalg import cg,lsmr,lsqr,LinearOperator
from scipy.sparse.csgraph import connected_components
from collections import deque
class ThreeFE:
 def __init__(self,g,w,rank_certificate):
  self.g=g;self.w=w;self.ng=[int(x.max())+1 for x in g];self.offset=np.cumsum([0]+self.ng[:-1]);self.c=[np.bincount(x,weights=w)for x in g]
  packed=(g[0].astype('int64')*self.ng[1]+g[1])*self.ng[2]+g[2]
  unique,self.cell=np.unique(packed,return_inverse=True);del packed
  self.cw=np.bincount(self.cell,weights=w);self.nc=len(unique);gg=[unique//(self.ng[1]*self.ng[2]),(unique//self.ng[2])%self.ng[1],unique%self.ng[2]]
  rr=np.repeat(np.arange(self.nc),3);cc=np.column_stack([gg[j]+self.offset[j]for j in range(3)]).ravel();vv=np.column_stack([np.sqrt(self.cw/self.c[j][gg[j]])for j in range(3)]).ravel()
  full=csr_matrix((vv,(rr,cc)),shape=(self.nc,sum(self.ng)))
  pair=g[0].astype('int64')*self.ng[2]+g[2];up=np.unique(pair);C=csr_matrix((np.ones(len(up)),(up//self.ng[2],up%self.ng[2])),shape=(self.ng[0],self.ng[2]));_,labels=connected_components(bmat([[None,C],[C.T,None]],format='csr'),directed=False)
  representatives={}
  for i,component in enumerate(labels[self.ng[0]:]):
   if component not in representatives or self.c[2][i]>self.c[2][representatives[component]]:representatives[component]=i
  keep=np.ones(sum(self.ng),bool)
  for i in representatives.values():keep[self.offset[2]+i]=False
  n_pivots=rank_certificate['additional_pivot_columns']
  if len(n_pivots)==self.ng[1]-1:
   # The exact constraint nullspace is the common constant. Any omitted size
   # category is a valid gauge; choosing the largest avoids a rare-level gauge.
   n_pivots=[i for i in range(self.ng[1])if i!=int(np.argmax(self.c[1]))]
  for i in range(self.ng[1]):keep[self.offset[1]+i]=i in n_pivots
  self.keep=np.flatnonzero(keep);assert len(self.keep)==rank_certificate['union_rank']
  self.F=full[:,self.keep];self.FT=self.F.T.tocsr();self.L=LinearOperator((len(self.keep),len(self.keep)),matvec=lambda x:self.FT@(self.F@x),dtype='float64')
  log(f'ThreeFE compressed {len(w)} rows to {self.nc} FE cells; exact full-rank dummy basis {len(self.keep)}/{sum(self.ng)} columns; FE span unchanged')
 def project(self,X,method='lsqr'):
  X=np.asarray(X);X=X[:,None]if X.ndim==1 else X;R=np.empty_like(X,dtype=float)
  for k in range(X.shape[1]):
   cellrhs=np.bincount(self.cell,weights=self.w*X[:,k])/np.sqrt(self.cw)
   if method=='independent':
    sol=lsmr(self.F,cellrhs,atol=1e-15,btol=1e-15,conlim=1e14,maxiter=5000);theta=sol[0]
    assert sol[1] in [1,2,4,5],('LSMR',sol[1:4])
   else:
    # Solve the weighted incidence problem directly. Normal-equation CG was rejected
    # by the full-data orthogonality gate; the singular gauges can cause CG drift.
    sol=lsqr(self.F,cellrhs,atol=1e-15,btol=1e-15,conlim=1e14,iter_lim=5000);theta=sol[0]
    assert sol[1] in [1,2,4,5],('LSQR',sol[1:4])
   expanded=np.zeros(sum(self.ng));expanded[self.keep]=theta;theta=expanded
   fit=np.zeros(len(self.w))
   for j,gg in enumerate(self.g):fit+=(theta[self.offset[j]:self.offset[j]+self.ng[j]]/np.sqrt(self.c[j]))[gg]
   residual=X[:,k]-fit
   for refinement in range(4):
    error=max(float(np.max(abs(np.bincount(gg,weights=self.w*residual)/self.c[j])))for j,gg in enumerate(self.g))
    if error<=5e-13:break
    rhs=self.FT@(np.bincount(self.cell,weights=self.w*residual)/np.sqrt(self.cw))
    correction,info=cg(self.L,rhs,rtol=1e-13,atol=1e-24,maxiter=5000);assert info==0,('Full-rank residual refinement CG',info)
    expanded=np.zeros(sum(self.ng));expanded[self.keep]=correction
    for j,gg in enumerate(self.g):residual-=(expanded[self.offset[j]:self.offset[j]+self.ng[j]]/np.sqrt(self.c[j]))[gg]
   assert error<=5e-13,('Full-rank projection residual',error)
   R[:,k]=residual
  return R

def rank_three(g,folder):
 # Exact lower/upper-bound certificate. Eliminate two incidence blocks on a spanning forest;
 # independent integer constraints on the third block establish its additional rank.
 cy,nn,org=g;na=int(cy.max())+1;nb=int(org.max())+1;ns=int(nn.max())+1
 packed=cy.astype('int64')*nb+org;u,ix=np.unique(packed,return_index=True)
 rr=u//nb;cc=u%nb;edge_n=nn[ix]
 C=csr_matrix((edge_n.astype('int32')+1,(rr,cc)),shape=(na,nb));G=bmat([[None,C],[C.T,None]],format='csr')
 components=int(connected_components(G,directed=False,return_labels=False));seen=np.zeros(na+nb,bool);pot=np.zeros((na+nb,ns),dtype='int32');tree_edges=[]
 for root in range(na+nb):
  if seen[root]:continue
  seen[root]=True;q=deque([root])
  while q:
   v=q.popleft()
   for pos in range(G.indptr[v],G.indptr[v+1]):
    child=G.indices[pos]
    if not seen[child]:
     s=int(G.data[pos])-1;pot[child]=-pot[v];pot[child,s]-=1;seen[child]=True;q.append(child)
 # A nonzero modular minor proves real independence. Additional rank <= ns-1:
 # the constant third-block column is in the first-block span.
 prime=1000003;basis={};witness=[]
 def add(row,index):
  nonlocal basis
  v=row.astype('int64')%prime
  for pivot,b in basis.items():v=(v-v[pivot]*b)%prime
  nz=np.flatnonzero(v)
  if len(nz):
   pivot=int(nz[0]);v=(v*pow(int(v[pivot]),-1,prime))%prime;basis[pivot]=v;basis=dict(sorted(basis.items()));witness.append(dict(observation_index=int(index),constraint=row.tolist()));return True
  return False
 en=min(10000,len(cy));z=pot[cy[:en]]+pot[na+org[:en]];z[np.arange(en),nn[:en]]+=1
 for j,row in enumerate(z):
  add(row,j)
  if len(basis)==ns-1:break
 # A rational nullspace verified on EVERY actual observation supplies the matching upper bound.
 from fractions import Fraction
 from math import lcm,gcd
 upper_verified=False;null_int=None
 for attempt in range(ns+1):
  rr=[[Fraction(v)for v in r['constraint']]for r in witness];piv=[];pr=0
  for col in range(ns):
   found=next((j for j in range(pr,len(rr))if rr[j][col]),None)
   if found is None:continue
   rr[pr],rr[found]=rr[found],rr[pr];div=rr[pr][col];rr[pr]=[v/div for v in rr[pr]]
   for j in range(len(rr)):
    if j!=pr and rr[j][col]:
     mult=rr[j][col];rr[j]=[x-mult*y for x,y in zip(rr[j],rr[pr])]
   piv.append(col);pr+=1
   if pr==len(rr):break
  assert pr==len(basis)
  vec=[]
  for free in [j for j in range(ns)if j not in piv]:
   v=[Fraction(0)for _ in range(ns)];v[free]=Fraction(1)
   for j,pc in enumerate(piv):v[pc]=-rr[j][free]
   den=lcm(*(x.denominator for x in v));vec.append([int(x*den)for x in v])
  null_int=np.array(vec,dtype='int64').T
  assert np.max(abs(pot.astype('int64')))*ns*max(1,int(np.max(abs(null_int))))<2**60
  groupnull=pot.astype('int64')@null_int;bad=None
  for st in range(0,len(cy),200000):
   en=min(st+200000,len(cy));errors=groupnull[cy[st:en]]+groupnull[na+org[st:en]]+null_int[nn[st:en]]
   hit=np.flatnonzero(np.any(errors!=0,axis=1))
   if len(hit):bad=st+int(hit[0]);break
  if bad is None:upper_verified=True;break
  constraint=pot[cy[bad]]+pot[na+org[bad]];constraint[nn[bad]]+=1
  assert add(constraint,bad),'Unlucky modular prime or rank inconsistency; no uncertified rank used'
 assert upper_verified
 result=dict(groups=[na,ns,nb],two_block_components=components,two_block_rank=na+nb-components,additional_exact_n_rank=len(basis),additional_pivot_columns=piv,additional_rank_upper_bound=ns-null_int.shape[1],union_rank=na+nb-components+len(basis),method='Spanning-forest integer elimination; modular nonzero minor lower bound and exact rational nullspace upper bound verified on every observation',modulus=prime,witness_rows=len(witness),verified_nullity=null_int.shape[1],all_observations_exact_nullspace_verified=True)
 save(folder/'FE_RANK_WITNESS.json',dict(certificate=result,witness=witness,nullspace_integer=null_int));return result
