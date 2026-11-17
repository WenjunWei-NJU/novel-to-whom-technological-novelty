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
# F4: dominant parallel sets with compact regression strip, lower opportunity pair.
f=plt.figure(figsize=(9,7.1));gs=f.add_gridspec(2,2,left=.10,right=.97,bottom=.14,top=.87,width_ratios=[1.65,1],height_ratios=[1.2,1],wspace=.32,hspace=.50);a=f.add_subplot(gs[0,0]);panel(a,'A','Historical composition of new content');cube=rd('alluvial');arr=np.zeros((3,3,4));stages=['E','M','S'];peers=['Low','Mid','High'];sources=list('ABCD')
for r in cube.itertuples():
 for k,key in enumerate(sources):arr[stages.index(r.stage),peers.index(r.peer_bin),k]=getattr(r,key)
assert (arr>=0).all();mass=arr.sum();arr/=mass;sums=[arr.sum((1,2)),arr.sum((0,2)),arr.sum((0,1))];starts=[];h=.76;gap=.035
for totals in sums:
 yy=.06;pos=[]
 for val in totals:pos.append(yy);yy+=h*val+gap
 starts.append(pos)
off=[np.zeros(len(t)) for t in sums];xpos=[.08,.48,.89];bar=.025
for i in range(3):
 for j in range(3):
  for k in range(4):
   ht=h*arr[i,j,k];idx=[i,j,k];ypos=[starts[c][idx[c]]+off[c][idx[c]] for c in range(3)]
   for c in [0,1]:
    x0=xpos[c]+bar/2;x1=xpos[c+1]-bar/2;mid=(x0+x1)/2;y0,y1=ypos[c:c+2];verts=[(x0,y0),(mid,y0),(mid,y1),(x1,y1),(x1,y1+ht),(mid,y1+ht),(mid,y0+ht),(x0,y0+ht),(x0,y0)];a.add_patch(PathPatch(MP(verts,[1,4,4,4,2,4,4,4,79]),fc=SC[sources[k]],ec='white',lw=.12,alpha=.56))
   for c in range(3):off[c][idx[c]]+=ht
for c,totals in enumerate(sums):
 for j,val in enumerate(totals):
  a.add_patch(Rectangle((xpos[c]-bar/2,starts[c][j]),bar,h*val,fc=SC[sources[j]] if c==2 else '#CBD5DB',ec='white',lw=.5));label=[['Early','Mid','Senior'],peers,sources][c][j];a.text(xpos[c]+(.025 if c==2 else -.025),starts[c][j]+h*val/2,label,ha='left' if c==2 else 'right',va='center',fontsize=7)
for x,title in zip(xpos,['Focal stage','Early peers','Coverage']):a.text(x,.97,title,ha='center',fontsize=7)
a.set(xlim=(-.04,1),ylim=(0,1));a.axis('off')
b=f.add_subplot(gs[0,1]);panel(b,'B','Component associations');order=['y','A','B','C','D']
for j,key in enumerate(order):
 for k,cov in enumerate(['INV_PAT','INV_CPC']):r=e[(e.family=='SOURCE_COMPOSITION')&(e.outcome==key)&(e.covariance==cov)].iloc[0];interval(b,r,4-j+(.09 if k==0 else -.09),SC[key],'o' if k==0 else 'D',k==1)
b.axhline(3.5,color='#DCE3E7',lw=.7);b.text(.98,.96,'Adjusted estimates',transform=b.transAxes,ha='right',fontsize=6.5,color=GRAY)
b.set(yticks=range(5),yticklabels=['D','C','B','A','Total Y'],ylim=(-.5,4.6),xlim=(-1.05,1.6),xticks=[-1,0,1],xlabel='E × AI slope difference (pp)');b.axvline(0,color='#8798A1',lw=.8);tidy(b)
c=f.add_subplot(gs[1,0]);panel(c,'C','Coverage conditional on opportunity size','Descriptive; nonempty early-exclusive history sets');op=rd('opportunities')
for ai in [0,1]:
 for category,ls in [('1-3','-'),('4-10','--'),('11+',':')]:
  q=op[(op.ai==ai)&(op.opportunity_bin==category)].set_index('exposure_bin').reindex(range(11));y=q.utilization.where(q.N>=20)*100;c.plot(np.minimum(1,(q.index.to_numpy()+.5)/10)*100,y,color=B if ai else GRAY,ls=ls,lw=1.5,marker='o',ms=2.3)
c.set(xlim=(0,102),ylim=(0,17),xlabel='Early-peer share (%)',ylabel='Opportunity coverage (%)',xticks=[0,25,50,75,100]);tidy(c);c.legend(handles=[Line2D([],[],color=B,label='AI'),Line2D([],[],color=GRAY,label='Non-AI')]+[Line2D([],[],color=INK,ls=l,label=t+' CPCs') for t,l in [('1–3','-'),('4–10','--'),('11+',':')]],loc='upper left',ncol=2,fontsize=6.5,columnspacing=1,handlelength=1.7)
d=f.add_subplot(gs[1,1]);panel(d,'D','Adjusted opportunity coverage')
for k,cov in enumerate(['INV_PAT','INV_CPC']):r=e[(e.family=='OPPORTUNITY')&(e.covariance==cov)].iloc[0];interval(d,r,.50-k*.20,B,'o' if k==0 else 'D',k==1,lw=1.6,size=5)
d.text(.5,.83,f'{r.estimate*100:+.3f} pp',transform=d.transAxes,ha='center',fontsize=14,color=B,weight='bold');d.text(.5,.72,'Point estimate · both clusterings',transform=d.transAxes,ha='center',fontsize=7,color=GRAY)
d.axvline(0,color='#8798A1',lw=.9);d.set(ylim=(-.05,1.05),yticks=[],xlim=(-.52,1.36),xticks=[-.5,0,.5,1],xlabel='E × AI slope difference (pp)');tidy(d);dual_legend(d,anchor=(-.02,.00),fontsize=6.2)
f.text(.10,.96,'Peer histories and opportunity coverage',fontsize=12,weight='bold');f.text(.10,.925,'Cross-sectional weighted composition, not longitudinal flows or knowledge transfer.',fontsize=7,color='#647580');f.text(.10,.055,'A  Early only     B  Other only     C  Both     D  Neither peer history\nRibbons: weighted new-CPC volume. Circles / diamonds: inventor + patent / inventor + CPC clustering.',fontsize=6.6,color='#647580')
save(f,4,['alluvial','source_means','opportunities','estimands'])

over.close()
