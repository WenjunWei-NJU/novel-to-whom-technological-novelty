from pathlib import Path
import shutil

import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,PathPatch
from matplotlib.path import Path as MP
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm
R=Path(__file__).resolve().parent
for folder in ("figures", "validation"):
    (R / folder).mkdir(exist_ok=True)
rd=lambda n:pd.read_csv(R/'source_data'/f'{n}.csv')
e=rd('estimands');B='#1677A6';O='#CC7437';G='#218C72';INK='#283D49';GRAY='#929DA5';PALE='#EDF1F3';PURPLE='#9272B1';SC={'A':G,'B':'#527EAA','C':'#C6A341','D':PURPLE,'y':INK};OUT=['relative_y','team_y','pair_y'];COL=dict(zip(OUT,[B,O,G]));COL['relative_minus_team']=INK
LAB={'relative_y':'Personal','team_y':'Team','pair_y':'Corpus pairs','relative_minus_team':'Personal − team'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':7,'ytick.labelsize':8,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'axes.edgecolor':'#A8B3BA','axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'legend.fontsize':7,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
cmap=LinearSegmentedColormap.from_list('blue_white_terracotta',['#1A5A85','#83B4CA','#F8F8F4','#E4A486','#AC443F'])
checks=[];plot_records=[]
def check(name,error,tol=1e-9):checks.append({'check':name,'maximum_absolute_error':float(error),'tolerance':tol,'pass':bool(error<=tol)})
def panel(ax,letter,title,sub=None):
 ax.text(0,1.07,letter,transform=ax.transAxes,weight='bold',fontsize=11,va='bottom');ax.annotate(title,xy=(0,1.07),xycoords='axes fraction',xytext=(18 if letter else 0,0),textcoords='offset points',fontsize=9,weight='medium',va='bottom')
 if sub:ax.text(0,1.015,sub,transform=ax.transAxes,fontsize=6.5,color='#647580',va='bottom')
def tidy(ax,grid=True):
 ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0);ax.tick_params(axis='x',length=3)
 if grid:ax.grid(axis='x',color='#E8ECEE',lw=.5);ax.set_axisbelow(True)
def interval(ax,r,y,col,marker='o',open=False,lw=1.25,size=4):
 if pd.notna(r.CI_low):ax.plot([r.CI_low*100,r.CI_high*100],[y,y],color=col,lw=lw,solid_capstyle='round',zorder=3)
 ax.plot(r.estimate*100,y,marker,ms=size,mfc='white' if open else col,mec=col,mew=.95,zorder=4)
 plot_records.append(r.to_dict())
def dual_legend(ax,anchor=(0,1.005),fontsize=6.7):
 ax.legend(handles=[Line2D([],[],color=INK,marker='o',ms=4,lw=1,label='Inventor + patent'),Line2D([],[],color=INK,marker='D',mfc='white',ms=4,lw=1,label='Inventor + CPC')],loc='lower left',bbox_to_anchor=anchor,ncol=2,fontsize=fontsize,handlelength=1.6,columnspacing=1.3)
over=PdfPages(R/'figures/Figures34_review.pdf')
def save(fig,i,src):
 d=R/'figures'/f'Figure{i}';d.mkdir(exist_ok=True)
 for ext in ['pdf','svg','png']:fig.savefig(d/f'Figure{i}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.10)
 over.savefig(fig,bbox_inches='tight',pad_inches=.10);plt.close(fig)
 for n in src:shutil.copy2(R/'source_data'/f'{n}.csv',d/f'{n}.csv')
# F3: full weighted distributions with explicit atoms; aligned adjusted stage contrasts.
f=plt.figure(figsize=(10,7.5));gs=f.add_gridspec(2,2,left=.13,right=.97,bottom=.14,top=.86,width_ratios=[1.4,1],height_ratios=[1.4,1],wspace=.38,hspace=.65);a=f.add_subplot(gs[0,0]);b=f.add_subplot(gs[0,1]);panel(a,'A','Observed novelty distributions','Weighted empirical data; display dots omitted');panel(b,'C','Adjusted stage differences','E × AI slope difference; pointwise 95% CIs');dist=rd('member_distribution');grid=np.linspace(0,1,201);summary=[];densities=[]
for j,stage in enumerate(['E','M','S']):
 for ai in [0,1]:
  q=dist[(dist.stage==stage)&(dist.ai==ai)];x=q.y.values;w=q.mass.values;total=w.sum();y=5-j*2-ai;col=B if ai else GRAY;inside=(x>0)&(x<1);xx=x[inside][None,:];ww=w[inside];density=(norm.pdf((grid[:,None]-xx)/.06)+norm.pdf((-grid[:,None]-xx)/.06)+norm.pdf((2-grid[:,None]-xx)/.06))@ww/(.06*total)
  if density.max()>0:a.fill_between(grid*100,y,y+.32*density/density.max(),facecolor=col,alpha=.36,edgecolor=col,lw=.7)
  order=np.argsort(x);cum=np.cumsum(w[order])/total;quant=x[order][np.searchsorted(cum,[.25,.5,.75])];a.plot(quant[[0,2]]*100,[y-.09]*2,color=col,lw=3,solid_capstyle='butt');a.plot(quant[1]*100,y-.09,'|',color='white',ms=5,mew=1)
  atoms=[w[x==v].sum()/total for v in [0,1]];a.vlines([0,100],[y,y],y+.32*np.array(atoms),colors=col,lw=2)
  a.text(104,y,f'{atoms[0]*100:.1f}',fontsize=6.5,va='center',color=col)
  summary.append(dict(stage=stage,ai=ai,weight_mass=total,observations=q.observations.sum(),mean=np.average(x,weights=w),q25=quant[0],median=quant[1],q75=quant[2],zero_share=atoms[0],one_share=atoms[1]))
  densities.extend(dict(stage=stage,ai=ai,x=t,density=v) for t,v in zip(grid,density))
 for ax in [a,b]:ax.axhspan(3.5-j*2,5.65-j*2,color='#F1F4F5',alpha=.65,zorder=0)
a.set(yticks=[5,4,3,2,1,0],yticklabels=['Early / non-AI','Early / AI','Mid / non-AI','Mid / AI','Senior / non-AI','Senior / AI'],xlim=(-3,113),ylim=(-.55,5.7),xticks=[0,25,50,75,100],xlabel='Personal novelty (% of current CPC subclasses)');tidy(a,False);a.text(104,5.8,'Zero %',fontsize=6.5,ha='left')
for j,stage in enumerate(['E','M','S']):
 for k,cov in enumerate(['INV_PAT','INV_CPC']):
  r=e[(e.family=='CAREER_STAGE')&(e.outcome==stage)&(e.covariance==cov)&(e.contrast=='E_AI')].iloc[0];interval(b,r,4.5-j*2+(.12 if k==0 else -.12),B,'o' if k==0 else 'D',k==1,lw=1.5,size=4.7)
b.set(yticks=[4.5,2.5,.5],yticklabels=['Early','Mid','Senior'],ylim=(-.55,5.7),xlim=(-1.85,3.9),xticks=[-1,0,1,2,3],xlabel='E × AI slope difference (pp)');b.axvline(0,color='#84959F',lw=.9);tidy(b);dual_legend(b,anchor=(-.02,-.26),fontsize=6.4)
f.text(.13,.94,'Career stage: distributions and conditional associations',fontsize=12,weight='bold');f.text(.13,.025,'Rainclouds: common 6 pp bandwidth, reflected boundaries; 0/100% atoms shown separately.\nCloud heights are shape-normalized. Bars: weighted IQR and median. Individual display dots omitted in aggregate-only candidate.',fontsize=6.6,color='#647580')
pd.DataFrame(summary).to_csv(R/'source_data/Figure3_weighted_summary.csv',index=False);pd.DataFrame(densities).to_csv(R/'source_data/Figure3_density.csv',index=False)
age=rd('age_profiles');age['weighted_y']=age.y*age.weight_mass
ag=age.groupby(['age','ai'],as_index=False)[['weighted_y','weight_mass','N']].sum();ag['novelty']=ag.weighted_y/ag.weight_mass
assert int(ag.N.sum())==int(dist.observations.sum())
ag.to_csv(R/'source_data/age_profiles_aggregated.csv',index=False)
c=f.add_subplot(gs[1,:]);panel(c,'B','Novelty across observed career ages','Descriptive cross-sectional relationship')
for ai in [0,1]:
 q=ag[ag.ai==ai].sort_values('age');c.plot(q.age,q.novelty*100,color=B if ai else GRAY,marker='o' if ai else 's',ms=3,lw=1.4,label='AI' if ai else 'Non-AI')
c.set(xlabel='Observed career age (years)',ylabel='Personal novelty (%)',xticks=[0,15,30,45],xticklabels=['0','15','30','45+'],ylim=(20,31));tidy(c);c.legend(loc='upper center',ncol=2,fontsize=7)
for j,stage in enumerate(['E','M','S']):
 rr=e[(e.family=='CAREER_STAGE')&(e.outcome==stage)&(e.covariance=='INV_PAT')].iloc[0]
 b.text(.98,(4.5-j*2+1.0)/6.25,f'{rr.estimate*100:+.3f} pp',transform=b.transAxes,ha='right',va='bottom',fontsize=7,color='#647580')
save(f,3,['age_profiles','age_profiles_aggregated','member_distribution','estimands','Figure3_weighted_summary','Figure3_density'])

over.close()
pd.DataFrame(plot_records).to_csv(R/'validation/Figure3_plotted.csv',index=False)
