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
over=PdfPages(R/'figures/Figure2/review.pdf')
def save(fig,i,src):
 d=R/'figures'/f'Figure{i}';d.mkdir(exist_ok=True)
 for ext in ['pdf','svg','png']:fig.savefig(d/f'Figure{i}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.10)
 over.savefig(fig,bbox_inches='tight',pad_inches=.10);plt.close(fig)
 for n in src:shutil.copy2(R/'source_data'/f'{n}.csv',d/f'{n}.csv')
# F2: triptych, support embedded, compact contrast band.
f=plt.figure(figsize=(7.87,6.9));gs=f.add_gridspec(2,3,left=.10,right=.96,bottom=.12,top=.88,height_ratios=[1.2,1.15],hspace=.60,wspace=.27);s=rd('composition_surface');xs=np.sort(s.early_share.unique());ys=np.sort(s.senior_share.unique());vmax=np.nanmax(np.abs(s[['nonAI_component_pp','AI_component_pp']].values));diffmax=np.nanmax(abs(s.AI_modulation_pp));axes=[]
for j,(key,title) in enumerate(zip(['nonAI_component_pp','AI_component_pp','AI_modulation_pp'],['Non-AI','AI','AI − non-AI'])):
 ax=f.add_subplot(gs[0,j]);axes.append(ax);mat=s.pivot(index='senior_share',columns='early_share',values=key).loc[ys,xs].values;mask=s.pivot(index='senior_share',columns='early_share',values='supported').loc[ys,xs].values
 ax.set_facecolor(PALE);im=ax.pcolormesh(xs*100,ys*100,np.ma.masked_where(~mask,mat),cmap=cmap,vmin=-(diffmax if j==2 else vmax),vmax=(diffmax if j==2 else vmax),shading='nearest');ax.plot([0,100],[100,0],color='#7E909A',lw=.6)
 ax.plot([20,40],[20,40],color=INK,lw=.75);ax.plot(20,20,'o',ms=4,mfc='white',mec=INK);ax.plot(40,40,'D',ms=4,mfc='white',mec=INK);ax.set(xlim=(0,100),ylim=(0,100),aspect='equal',xticks=[0,50,100],yticks=[0,50,100],xlabel='Early-career share (%)');panel(ax,chr(65+j),title)
 if j==0:ax.set_ylabel('Senior-career share (%)')
 else:ax.set_yticklabels([])
 cb=f.colorbar(im,ax=ax,orientation='horizontal',fraction=.045,pad=.20,aspect=24);cb.ax.tick_params(labelsize=6);cb.set_label('Anchored composition component (pp)' if j<2 else 'Anchored difference (pp)',fontsize=6.3,labelpad=2)
f.text(.10,.96,'Nonlinear career composition',fontsize=12,weight='bold');f.text(.10,.925,r'Personal novelty · Reference: $s_E=s_S=0.20$ · Common support at FIRST = 0',fontsize=7.5,color='#647580')
outer=f.add_subplot(gs[1,:]);outer.axis('off');outer.text(0,1.18,'D',transform=outer.transAxes,fontsize=11,weight='bold');outer.annotate('Four five-member composition contrasts',xy=(0,1.18),xycoords='axes fraction',xytext=(18,0),textcoords='offset points',fontsize=9)
sub=gs[1,:].subgridspec(2,2,wspace=.24,hspace=1.10)
m=e[(e.family=='MAIN')&e.estimate.notna()]
for j,(o,title) in enumerate(zip(['relative_y','team_y','relative_minus_team','pair_y'],['Personal novelty','Team novelty','Personal−team','Corpus-first pairs'])):
 ax=f.add_subplot(sub[j//2,j%2]);r=m[m.outcome==o].iloc[0];col=COL[o]
 ax.text(0,1.05,title,transform=ax.transAxes,fontsize=8.5,weight='medium',va='bottom')
 val=f'{r.estimate*100:+.4f}' if o!='pair_y' else f'{r.estimate*100:.7f}'
 ax.text(1,1.05,val+' pp',transform=ax.transAxes,fontsize=10,weight='bold',color=col,ha='right',va='bottom')
 interval(ax,r,.50,col,lw=2,size=5)
 ax.axvline(0,color='#8B9BA4',lw=.8)
 ax.set(yticks=[],ylim=(0,1),xlim=(-.1,1.4) if o!='pair_y' else (-.014,.014),xticks=[0,.4,.8,1.2] if o!='pair_y' else [-.01,0,.01])
 tidy(ax,False)
 ax.text(.5,-.58,'AI − non-AI difference (pp)'+(' · Fine scale' if o=='pair_y' else ''),transform=ax.transAxes,ha='center',fontsize=6.5)
f.text(.10,.025,'Five members, FIRST = 0: Early/Mid/Senior (1,3,1) → (2,1,2). Pointwise 95% CIs.\nCorpus-first pairs use a different knowledge unit and a fine scale. Grey heatmap cells indicate unsupported or infeasible compositions.',fontsize=6.2,color='#647580')
save(f,2,['composition_surface','profile_support','estimands'])

over.close()
pd.DataFrame(plot_records).to_csv(R/'figures/Figure2/plotted_estimates.csv',index=False)
