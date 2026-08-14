"""Resumable batch runner: works for a bounded wall-time, then exits."""
import sys, os, time, json
import numpy as np, pandas as pd
import pipeline as P

T="/sessions/tender-affectionate-carson/mnt/Article Zebrafish/zebrafish material and code/TRAJECTORY RESULTS"
OUT=sys.argv[1] if len(sys.argv)>1 else 'features.jsonl'
SURR=(len(sys.argv)>2 and sys.argv[2]=='surr')
BUDGET=float(sys.argv[3]) if len(sys.argv)>3 else 95.0

done=set()
if os.path.exists(OUT):
    for line in open(OUT):
        try: d=json.loads(line); done.add((d['fish'],d['trial']))
        except: pass
trs=[t for t in P.find_trials(T) if (t[0],t[1]) not in done]
t0=time.time(); n=0
with open(OUT,'a') as fh:
    for i,(f,t,h) in enumerate(trs):
        if time.time()-t0 > BUDGET: break
        try:
            r,_=P.process_trial(h,f,t,do_surrogates=SURR,seed=abs(hash((f,t)))%10000)
        except Exception as e:
            r=dict(fish=f,trial=t,error=repr(e))
        fh.write(json.dumps({k:(None if isinstance(v,float) and not np.isfinite(v) else v)
                             for k,v in r.items()})+"\n"); fh.flush(); n+=1
print(f"processats {n} | resten {len(trs)-n} | {time.time()-t0:.0f}s")
