import numpy as np, pandas as pd, time, pipeline as P
from multiprocessing import Pool
T="/sessions/tender-affectionate-carson/mnt/Article Zebrafish/zebrafish material and code/TRAJECTORY RESULTS"
def job(a):
    f,t,h=a
    try: return P.process_trial(h,f,t,do_surrogates=False)[0]
    except Exception as e: return dict(fish=f,trial=t,error=str(e))
if __name__=="__main__":
    trs=P.find_trials(T); t0=time.time()
    with Pool(2) as p: rows=p.map(job,trs)
    df=pd.DataFrame(rows); df['group']=np.where(df.fish.str.startswith('sf'),'scoliotic','healthy')
    df.to_csv('features_nosurr.csv',index=False)
    print(f"fet en {time.time()-t0:.0f}s | files={len(df)} | errors={df.get('error',pd.Series(dtype=object)).notna().sum()}")
