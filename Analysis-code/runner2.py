"""Second-battery feature extraction, resumable within a wall-time budget."""
import sys, os, time, json, warnings
import numpy as np
warnings.filterwarnings('ignore')
import nld, nld2, pipeline as P

T="/sessions/tender-affectionate-carson/mnt/Article Zebrafish/zebrafish material and code/TRAJECTORY RESULTS"
OUT='features2.jsonl'
BUDGET=float(sys.argv[1]) if len(sys.argv)>1 else 95.0

def battery(theta, v, fs):
    r={}
    r.update(nld2.lateral_asymmetry(theta, fs))                       # H1: asimetria
    for d in (4,5,6):                                                 # H2: pla H-C
        H,C=nld2.statistical_complexity(theta,d=d,tau=1)
        r[f'PE_H{d}']=H; r[f'PE_C{d}']=C
    r['PE_missing5']=nld2.missing_patterns(theta,d=5)
    r['Trev_1']=nld2.time_reversal_asymmetry(theta,1)
    r['Trev_10']=nld2.time_reversal_asymmetry(theta,max(1,int(0.1*fs)))
    r['DFA_alpha']=nld2.dfa(theta)
    r.update({k:v2 for k,v2 in nld2.mfdfa(theta).items()})
    r.update(nld2.ami_profile(theta,max_lag=min(250,len(theta)//8)))
    r['AIS']=nld2.active_information_storage(theta,d=4)
    r['TE_theta_to_v']=nld2.transfer_entropy(theta,v,d=3)
    r['TE_v_to_theta']=nld2.transfer_entropy(v,theta,d=3)
    cur=nld2.rcmse(theta,scales=list(range(1,16)),m=2,r=0.15)
    r.update(nld2.mse_summary(cur))
    r['_mse']=[cur[k] for k in sorted(cur)]
    tau=max(1,nld.first_min_tau(nld.average_mutual_information(theta,150)))
    X=nld.embed(theta,4,tau)
    r.update(nld2.recurrence_network(X,0.05,theiler=max(int(fs/2),4*tau)))
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
            pts,fs,conf=P.load_points(h)
            _,v=P.head_kinematics(pts,fs)
            th=P.bending_angle(pts,fs)
            r=battery(th,v,fs); r['fish']=f; r['trial']=t
        except Exception as e:
            r=dict(fish=f,trial=t,error=repr(e))
        fh.write(json.dumps({k:(None if isinstance(x,float) and not np.isfinite(x) else x)
                             for k,x in r.items()})+"\n"); fh.flush(); n+=1
print(f"processats {n} | resten {len(trs)-n} | {time.time()-t0:.0f}s")
