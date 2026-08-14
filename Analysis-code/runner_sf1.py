"""Full battery + surrogates for the third scoliotic animal (sfs4)."""
import sys, os, time, json, glob, warnings
import numpy as np
warnings.filterwarnings('ignore')
import nld, nld2, pipeline as P
V="/sessions/tender-affectionate-carson/mnt/Article Zebrafish/zebrafish material and code/VIDEO_DOWNSPLD/S4/sf"
OUT='features_sfs4.jsonl'; SUR='surrogates_sfs4.jsonl'
BUDGET=float(sys.argv[1]) if len(sys.argv)>1 else 95.0
MODE=sys.argv[2] if len(sys.argv)>2 else 'feat'

def battery(theta,v,fs):
    r={}
    r.update(nld2.lateral_asymmetry(theta,fs))
    for dd in (4,5,6):
        H,C=nld2.statistical_complexity(theta,d=dd,tau=1); r[f'PE_H{dd}']=H; r[f'PE_C{dd}']=C
    r['PE_missing5']=nld2.missing_patterns(theta,d=5)
    r['Trev_1']=nld2.time_reversal_asymmetry(theta,1)
    r['Trev_10']=nld2.time_reversal_asymmetry(theta,max(1,int(0.1*fs)))
    r['DFA_alpha']=nld2.dfa(theta); r.update(nld2.mfdfa(theta))
    r.update(nld2.ami_profile(theta,max_lag=min(250,len(theta)//8)))
    r['AIS']=nld2.active_information_storage(theta,d=4)
    r['TE_theta_to_v']=nld2.transfer_entropy(theta,v,d=3)
    r['TE_v_to_theta']=nld2.transfer_entropy(v,theta,d=3)
    cur=nld2.rcmse(theta,scales=list(range(1,16)),m=2,r=0.15)
    r.update(nld2.mse_summary(cur)); r['_mse']=[cur[k] for k in sorted(cur)]
    tau=max(1,nld.first_min_tau(nld.average_mutual_information(theta,150)))
    r.update(nld2.recurrence_network(nld.embed(theta,4,tau),0.05,theiler=max(int(fs/2),4*tau)))
    return r

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

trs=[]
for h in sorted(glob.glob(os.path.join(V,'*','*_filtered.h5'))):
    if os.path.basename(h).startswith('._'): continue
    trs.append(('sfs4', os.path.basename(os.path.dirname(h)), h))
trs.sort(key=lambda t:(len(t[1]),t[1]))

target = OUT if MODE=='feat' else SUR
done=set()
if os.path.exists(target):
    for line in open(target):
        try: d=json.loads(line); done.add((d['fish'],d['trial']))
        except: pass
todo=[t for t in trs if (t[0],t[1]) not in done]
t0=time.time(); n=0
with open(target,'a') as fh:
    for f,t,h in todo:
        if time.time()-t0>BUDGET: break
        try:
            if MODE=='feat':
                r,_=P.process_trial(h,f,t,do_surrogates=False)
                pts,fs,_=P.load_points(h); _,v=P.head_kinematics(pts,fs); th=P.bending_angle(pts,fs)
                r.update(battery(th,v,fs))
            else:
                pts,fs,_=P.load_points(h); th=P.bending_angle(pts,fs)
                tau=max(1,nld.first_min_tau(nld.average_mutual_information(th,150)))
                obs=stats(th,fs,tau); seed=abs(hash((f,t)))%10000
                sur={k:[] for k in obs}
                for s in range(39):
                    us=nld.iaaft(th,n_iter=120,rng=seed*100+s)
                    rs=stats(us,fs,tau)
                    for k in obs: sur[k].append(rs[k])
                r=dict(fish=f,trial=t,tau=int(tau))
                for k,o in obs.items():
                    arr=np.array(sur[k],float); r[k]=o
                    r['p_'+k]=nld.surrogate_rank_p(o,arr,ALT[k])
                    r['z_'+k]=float((o-np.nanmean(arr))/(np.nanstd(arr)+1e-12))
                    r['sm_'+k]=float(np.nanmean(arr))
        except Exception as e:
            r=dict(fish=f,trial=t,error=repr(e))
        fh.write(json.dumps({k:(None if isinstance(x,float) and not np.isfinite(x) else x)
                             for k,x in r.items()})+"\n"); fh.flush(); n+=1
print(f"[{MODE}] processats {n} | resten {len(todo)-n} | {time.time()-t0:.0f}s")
