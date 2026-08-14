"""IAAFT surrogate testing, 39 surrogates per trial, resumable."""
import sys, os, time, json, warnings
import numpy as np
warnings.filterwarnings('ignore')
import nld, nld2, pipeline as P

T="/sessions/tender-affectionate-carson/mnt/Article Zebrafish/zebrafish material and code/TRAJECTORY RESULTS"
OUT='surrogates.jsonl'; NS=39
BUDGET=float(sys.argv[1]) if len(sys.argv)>1 else 95.0
ALT=dict(Trev1='two-sided',TrevL='two-sided',PE_C5='greater',PE_H5='less',
         PE_missing5='greater',SampEn='less',DET='greater',ENTR='greater')

def stats(u,fs,tau):
    r={}
    r['Trev1']=nld2.time_reversal_asymmetry(u,1)
    r['TrevL']=nld2.time_reversal_asymmetry(u,max(1,int(0.1*fs)))
    H,C=nld2.statistical_complexity(u,d=5); r['PE_H5']=H; r['PE_C5']=C
    r['PE_missing5']=nld2.missing_patterns(u,d=5)
    r['SampEn']=nld.sample_entropy(u[:1500],2,0.2)
    X=nld.embed(u,4,tau); st=max(1,len(X)//1000)
    q=nld.rqa(X[::st][:1000],0.05,max(1,int(fs/2)//st))
    r['DET']=q['DET']; r['ENTR']=q['ENTR']
    return r

done=set()
if os.path.exists(OUT):
    for line in open(OUT):
        try: d=json.loads(line); done.add((d['fish'],d['trial']))
        except: pass
trs=[t for t in P.find_trials(T) if (t[0],t[1]) not in done]
t0=time.time(); n=0
with open(OUT,'a') as fh:
    for f,t,h in trs:
        if time.time()-t0>BUDGET: break
        try:
            pts,fs,_=P.load_points(h); th=P.bending_angle(pts,fs)
            tau=max(1,nld.first_min_tau(nld.average_mutual_information(th,150)))
            obs=stats(th,fs,tau)
            seed=abs(hash((f,t)))%10000
            sur={k:[] for k in obs}
            for s in range(NS):
                us=nld.iaaft(th,n_iter=120,rng=seed*100+s)
                rs=stats(us,fs,tau)
                for k in obs: sur[k].append(rs[k])
            row=dict(fish=f,trial=t,tau=int(tau))
            for k,o in obs.items():
                arr=np.array(sur[k],float)
                row[k]=o
                row['p_'+k]=nld.surrogate_rank_p(o,arr,ALT[k])
                row['z_'+k]=float((o-np.nanmean(arr))/(np.nanstd(arr)+1e-12))
                row['sm_'+k]=float(np.nanmean(arr))
            r=row
        except Exception as e:
            r=dict(fish=f,trial=t,error=repr(e))
        fh.write(json.dumps({k:(None if isinstance(x,float) and not np.isfinite(x) else x)
                             for k,x in r.items()})+"\n"); fh.flush(); n+=1
print(f"processats {n} | resten {len(trs)-n} | {time.time()-t0:.0f}s")
