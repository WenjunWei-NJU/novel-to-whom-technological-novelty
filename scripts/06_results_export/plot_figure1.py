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
B='#1677A6';O='#CC7437';G='#218C72';INK='#283D49';GRAY='#929DA5';PALE='#EDF1F3';PURPLE='#9272B1';SC={'A':G,'B':'#527EAA','C':'#C6A341','D':PURPLE,'y':INK};OUT=['relative_y','team_y','pair_y'];COL=dict(zip(OUT,[B,O,G]));COL['relative_minus_team']=INK
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
over=PdfPages(R/'figures/Figure1/Figure1_review.pdf')
def save(fig,i,src):
 d=R/'figures'/f'Figure{i}';d.mkdir(exist_ok=True)
 for ext in ['pdf','svg','png']:fig.savefig(d/f'Figure{i}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.10)
 over.savefig(fig,bbox_inches='tight',pad_inches=.10);plt.close(fig)
 for n in src:shutil.copy2(R/'source_data'/f'{n}.csv',d/f'{n}.csv')
# F1: top reference strip; lower composition + dot strips + correlation tiles.
f=plt.figure(figsize=(9.6,5.8));gs=f.add_gridspec(2,3,left=.09,right=.975,bottom=.13,top=.90,height_ratios=[1.1,1],width_ratios=[1,1,1.15],hspace=.50,wspace=.75)
a=f.add_subplot(gs[0,:]);a.set(xlim=(0,3),ylim=(0,1));a.axis('off');panel(a,'A','Three historical reference scales of technological novelty','Conceptual illustration • coloured content lies beyond the stated reference')
for i,(label,unit,c) in enumerate(zip(['Personal history','Team-history union','Earlier patent corpus'],['CPC subclasses','CPC subclasses','Unordered subclass pairs'],[B,O,G])):
 a.add_patch(Rectangle((i+.02,.13),.93,.75,facecolor='#F5F7F8',edgecolor='#D9E1E5',lw=.7));a.text(i+.485,.77,label,ha='center',weight='bold',fontsize=8.5)
 coords=[(i+.2,.55),(i+.48,.55),(i+.76,.55),(i+.2,.31),(i+.48,.31),(i+.76,.31)]
 if i==2:
  for p,q in [(0,1),(0,3),(1,4),(3,4),(4,5)]:a.plot([coords[p][0],coords[q][0]],[coords[p][1],coords[q][1]],color=c if (p,q)==(4,5) else '#C5CED3',lw=2.5 if (p,q)==(4,5) else 1,zorder=1)
 a.scatter(*np.array(coords).T,s=100,c=[c if i<2 and j>=[3,5][i] else '#CBD4D9' for j in range(6)],edgecolors='white',linewidths=.8,zorder=2)
 a.text(i+.485,.02,unit,ha='center',fontsize=7,color='#657781')
b=f.add_subplot(gs[1,0]);panel(b,'B','Career composition');comp=rd('composition');left=np.zeros(2)
for k,(label,c) in enumerate(zip(['Early','Mid','Senior'],['#58A6CA','#CBD4DA',O])):
 vals=[]
 for ai in [0,1]:
  q=comp[comp.ai==ai];v=[q.n_early/q.n,1-(q.n_early+q.n_senior)/q.n,q.n_senior/q.n][k];vals.append(np.average(v,weights=q.team_count)*100)
 b.barh([1,0],vals,left=left,height=.42,color=c,label=label)
 for y,v,l in zip([1,0],vals,left):b.text(l+v/2,y,f'{v:.1f}',ha='center',va='center',fontsize=7,color='white' if k!=1 else INK)
 left+=vals
b.set(yticks=[1,0],yticklabels=['Non-AI','AI'],xlim=(0,100),ylim=(-.7,1.7),xlabel='Mean team share (%)',xticks=[0,50,100]);tidy(b,False);b.legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=3,handlelength=.8,columnspacing=.8)
c=f.add_subplot(gs[1,1]);panel(c,'C','Zero-valued novelty');z=rd('zeros').dropna(subset=['ai']).set_index('ai')
for j,key in enumerate(['relative_zero','team_zero','pair_zero']):
 vals=[z.loc[ai,key]*100 for ai in [0,1]];c.plot(vals,[2-j]*2,color='#C4CED4',lw=2)
 for ai,v in enumerate(vals):c.plot(v,2-j+(.065 if ai else -.065),'o' if ai else 's',ms=4,color=B if ai else GRAY)
 c.text(108,2-j,f'{vals[0]:.1f} / {vals[1]:.1f}',va='center',fontsize=6.5)
c.set(xlim=(0,158),xticks=[0,50,100],yticks=[2,1,0],yticklabels=['Personal','Team','Corpus pairs'],ylim=(-.5,2.5),xlabel='Zero mass (%)');tidy(c);c.legend(handles=[Line2D([],[],marker='s',color=GRAY,ls='',label='Non-AI'),Line2D([],[],marker='o',color=B,ls='',label='AI')],loc='upper center',bbox_to_anchor=(.5,-.19),ncol=2)
d=f.add_subplot(gs[1,2]);corr=rd('novelty_correlations').pivot(index='metric_x',columns='metric_y',values='rho').loc[OUT,OUT].values
d.imshow(corr,vmin=0,vmax=1,cmap='Blues',aspect='auto')
for y in range(3):
 for x in range(3):d.text(x,y,f'{corr[y,x]:.3f}',ha='center',va='center',color='white' if corr[y,x]>.6 else INK,fontsize=8.5)
d.set(xticks=[0,1,2],yticks=[0,1,2],xticklabels=['Personal','Team','Corpus pairs'],yticklabels=['Personal','Team','Corpus pairs'])
d.tick_params(length=0,pad=6);[sp.set_visible(False) for sp in d.spines.values()];d.set_xlabel('Spearman correlation',labelpad=10)
panel(d,'D','Scale relations')
f.text(.09,.005,'FIRST is included in Early. Historical references are parallel measurement scales, not a transmission sequence.',fontsize=7,color='#647580')
save(f,1,['composition','zeros','novelty_correlations'])

over.close()
