"""Why the published separation appeared: subset vs full cohort, truth vs k-means."""
import json, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'font.family':'DejaVu Sans','font.size':8,'axes.linewidth':0.7,
                     'savefig.bbox':'tight','axes.titlesize':8.5,'axes.labelsize':8,
                     'legend.fontsize':6.6,'figure.dpi':300})
CH,CS='#2c6fb5','#d1495b'; K1,K2='#4878a8','#b03a2e'
d=pd.read_csv('features_final.csv'); d['y']=d.fish.str.startswith('sf').astype(int)
SUB=['f2','f3','sf2','sf3']

def plane(sub):
    x=(sub.distance_cm-sub.distance_cm.mean())/sub.distance_cm.std()
    y=(sub.SampEn-sub.SampEn.mean())/sub.SampEn.std()
    return np.c_[x,y]

def km_labels(X,y):
    lab=KMeans(2,n_init=50,random_state=0).fit(X).labels_
    if (lab==y).mean()<(1-lab==y).mean(): lab=1-lab
    return lab,max((lab==y).mean(),(1-lab==y).mean())

fig=plt.figure(figsize=(7.1,5.1))
gs=fig.add_gridspec(2,3,width_ratios=[1,1,0.82],hspace=0.50,wspace=0.40)

panels=[('subset',d[d.fish.isin(SUB)],'2 vs 2 subset (40 trials)'),
        ('full',d,'full cohort (110 trials)')]
for r,(tag,sub,title) in enumerate(panels):
    X=plane(sub); y=sub.y.values
    lab,ag=km_labels(X,y)
    # ground truth
    ax=fig.add_subplot(gs[r,0])
    for c,col,mk,nm in ((0,CH,'o','healthy'),(1,CS,'^','scoliotic')):
        m=y==c
        ax.scatter(X[m,0],X[m,1],s=22,c=col,marker=mk,ec='k',lw=.35,alpha=.9,label=nm)
    ax.set_ylabel('normalised sample entropy')
    ax.set_title(f"({'ac'[r]}) ground truth\n{title}",loc='left',fontsize=8)
    if r==0:
        ax.margins(0.10)
        lo,hi=ax.get_ylim(); ax.set_ylim(lo,hi+0.30*(hi-lo))
        ax.legend(loc='upper left',frameon=False,ncol=2,handletextpad=.3,
                  columnspacing=.9,borderaxespad=0.2)
    if r==1: ax.set_xlabel('normalised distance travelled')
    # k-means
    ax=fig.add_subplot(gs[r,1])
    for c,col,nm in ((0,K1,'cluster 1'),(1,K2,'cluster 2')):
        m=lab==c
        ax.scatter(X[m,0],X[m,1],s=22,c=col,marker='o',ec='k',lw=.35,alpha=.9,label=nm)
    ax.set_title(f"({'bd'[r]}) $k$-means\nagrees with truth: {100*ag:.0f}%",loc='left',fontsize=8)
    if r==0:
        ax.margins(0.10)
        lo,hi=ax.get_ylim(); ax.set_ylim(lo,hi+0.30*(hi-lo))
        ax.legend(loc='upper left',frameon=False,ncol=2,handletextpad=.3,
                  columnspacing=.9,borderaxespad=0.2)
    if r==1: ax.set_xlabel('normalised distance travelled')

# (e) accuracy comparison
ax=fig.add_subplot(gs[:,2])
cols=['SampEn','distance_cm']
s=d[d.fish.isin(SUB)]
a_sub=make_pipeline(StandardScaler(),LDA()).fit(s[cols],s.y).score(s[cols],s.y)*100
a_all=make_pipeline(StandardScaler(),LDA()).fit(d[cols],d.y).score(d[cols],d.y)*100
yt,yp=[],[]
for f in d.fish.unique():
    tr,te=d[d.fish!=f],d[d.fish==f]
    m=make_pipeline(StandardScaler(),LDA()).fit(tr[cols],tr.y)
    yp.extend(m.predict(te[cols])); yt.extend(te.y)
yt,yp=np.array(yt),np.array(yp); a_cv=(yt==yp).mean()*100; sens=yp[yt==1].mean()*100
base=100*(1-d.y.mean())
vals=[a_sub,a_all,a_cv]
lbl=['2 vs 2 subset, in-sample',
     'all animals, in-sample',
     'leave-one-animal-out\ncross-validation\n(sensitivity %.0f%%)'%sens]
ypos=[0.0,1.0,2.1]
ax.barh(ypos,vals,color=['#d1495b','#e8a33d','#2c6fb5'],ec='k',lw=.6,height=.34)
ax.axvline(base,color='k',ls='--',lw=.9,zorder=0)
ax.text(base,2.72,'majority-class\nbaseline',fontsize=6.2,ha='center',va='top',
        color='0.25',linespacing=1.25)
for i,(v,yy) in enumerate(zip(vals,ypos)):
    ax.text(v-2.5,yy,f'{v:.1f}%',ha='right',va='center',fontsize=7.6,
            fontweight='bold',color='white')
    ax.text(0,yy-0.40,lbl[i],fontsize=6.0,va='bottom',ha='left',color='0.15',
            linespacing=1.25)
ax.set_yticks([])
ax.set_xlim(0,100); ax.set_ylim(3.05,-0.62); ax.invert_yaxis(); ax.invert_yaxis()
ax.set_xlabel('accuracy (%)')
ax.set_title('(e) effect of the protocol',loc='left')
for sp in ('top','right','left'): ax.spines[sp].set_visible(False)
fig.savefig('fig_subset.pdf'); plt.close(fig)
json.dump(dict(acc_subset=a_sub,acc_all=a_all,acc_cv=a_cv,sens_cv=sens,base=base),
          open('clf_numbers.json','w'))
print('ok fig_subset.pdf |',{k:round(v,1) for k,v in
      dict(sub=a_sub,all=a_all,cv=a_cv,sens=sens,base=base).items()})
