import json, warnings; warnings.filterwarnings('ignore')
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'font.family':'DejaVu Sans','font.size':8,'axes.linewidth':0.7,
                     'savefig.bbox':'tight','axes.titlesize':9,'axes.labelsize':8,
                     'legend.fontsize':7,'figure.dpi':300})
CH,CS='#2c6fb5','#d1495b'
d=json.load(open('classify_results.json')); null=np.load('perm_null_exact.npy')
evr=np.array(d['evr'])*100; cum=np.array(d['cum'])*100
S=np.array(d['scores']); y=np.array(d['y']); fish=d['fish']

fig,axes=plt.subplots(1,3,figsize=(7.1,2.5),gridspec_kw={'width_ratios':[1,1,1]})

# (a) scree
ax=axes[0]; npc=min(10,len(evr)); x=np.arange(1,npc+1)
ax.bar(x,evr[:npc],color='#4a4a4a',ec='k',lw=.5,width=.68)
ax.set_xlabel('principal component'); ax.set_ylabel('variance explained (%)')
ax.set_xticks(x); ax.set_ylim(0,max(evr)*1.18)
ax2=ax.twinx()
ax2.plot(x,cum[:npc],'-o',color='#c0392b',ms=3,lw=1.1)
ax2.set_ylabel('cumulative (%)',color='#c0392b'); ax2.set_ylim(0,105)
ax2.tick_params(axis='y',colors='#c0392b')
ax2.axhline(80,color='#c0392b',ls=':',lw=.7)
ax.set_title('(a) scree plot',loc='left')

# (b) PCA
ax=axes[1]
for i,f in enumerate(fish):
    ax.scatter(S[i,0],S[i,1],s=52,marker='^' if y[i] else 'o',
               c=CS if y[i] else CH,ec='k',lw=.5,zorder=3)
    off={'sf3':(0,8),'f9':(10,-2),'sfs4':(-10,-2),'f6':(-10,-1),
         'sf2':(0,-12),'f5':(0,8)}.get(f,(0,7))
    ha={'f9':'left','sfs4':'right','f6':'right'}.get(f,'center')
    ax.annotate(f,(S[i,0],S[i,1]),textcoords='offset points',xytext=off,
                ha=ha,va='center' if abs(off[0])>5 else 'baseline',
                fontsize=6.0,color='0.25')
ax.axhline(0,color='0.85',lw=.6); ax.axvline(0,color='0.85',lw=.6)
ax.set_xlabel(f'PC1 ({evr[0]:.0f}%)'); ax.set_ylabel(f'PC2 ({evr[1]:.0f}%)')
ax.set_title('(b) animals in PC space',loc='left')
ax.margins(0.18)
lo,hi=ax.get_ylim(); ax.set_ylim(lo-0.30*(hi-lo),hi)
ax.legend(handles=[Line2D([],[],marker='o',ls='',mfc=CH,mec='k',ms=5,label='healthy'),
                   Line2D([],[],marker='^',ls='',mfc=CS,mec='k',ms=5,label='scoliotic')],
          loc='lower center',ncol=2,frameon=False,handletextpad=.3,
          columnspacing=1.0,borderaxespad=0.2)

# (c) permutation null
ax=axes[2]
obs=d['perm_exact']['observed']
ax.hist(null,bins=np.arange(0,1.06,0.0625),color='#b8c6d9',ec='k',lw=.4)
top=ax.get_ylim()[1]*1.34; ax.set_ylim(0,top)
ax.axvline(obs,color=CS,lw=1.8)
ax.annotate(f"observed {obs:.2f}",(obs,top*0.93),xytext=(-5,0),
            textcoords='offset points',ha='right',va='center',fontsize=6.4,
            color=CS)
q95=np.quantile(null,0.95)
ax.axvline(q95,color='0.35',ls='--',lw=.9)
ax.annotate('95th pct',(q95,top*0.78),xytext=(-5,0),
            textcoords='offset points',ha='right',va='center',fontsize=6.4,
            color='0.35')
ax.set_xlabel('balanced accuracy'); ax.set_ylabel('label assignments')
ax.set_title(f"(c) exact null, $p={d['perm_exact']['p']:.3f}$",loc='left')
fig.tight_layout(); fig.savefig('fig_ml.pdf'); plt.close(fig)
print('ok fig_ml.pdf | PC1-2 =',round(cum[1],1),'% | p =',d['perm_exact']['p'])
