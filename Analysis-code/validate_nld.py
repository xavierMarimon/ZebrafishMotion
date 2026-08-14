import numpy as np, time
from scipy.integrate import solve_ivp
import nld

rng=np.random.default_rng(1)
print("="*70); print("VALIDACIO CONTRA SISTEMES AMB INVARIANTS CONEGUTS"); print("="*70)

# ---------- 1. Logistic map r=4 : lambda1 = ln2 = 0.6931 ; D2 = 1 ----------
n=6000; x=np.empty(n); x[0]=0.2
for i in range(1,n): x[i]=4*x[i-1]*(1-x[i-1])
X=nld.embed(x,2,1)
t,y=nld.rosenstein(X,fs=1.0,theiler=10,k_max=20,nref=800)
lam,_=nld.lyapunov_from_curve(t,y,fit_frac=(0.0,0.35))
D2,_,_,_=nld.correlation_dimension(nld.embed(x,3,1),theiler=10,nref=800)
print(f"\n[Logistic r=4]  lambda1 = {lam:6.3f}  (teoric 0.693)   err {abs(lam-np.log(2))/np.log(2)*100:4.1f}%")
print(f"                D2      = {D2:6.3f}  (teoric ~1.00)")

# ---------- 2. Lorenz : lambda1 = 0.906 ; D2 = 2.05 ----------
def lor(t,s,sig=10,rho=28,beta=8/3):
    x,y,z=s; return [sig*(y-x), x*(rho-z)-y, x*y-beta*z]
dt=0.01; T=200.0
sol=solve_ivp(lor,[0,T],[1.,1.,1.],t_eval=np.arange(0,T,dt),rtol=1e-9,atol=1e-9)
xs=sol.y[0][2000:]                      # drop transient
fs=1/dt
ami=nld.average_mutual_information(xs,max_tau=60)
tau=nld.first_min_tau(ami)
fnn=nld.false_nearest_neighbours(xs,tau,max_m=8,nref=400)
m=nld.choose_m(fnn)
print(f"\n[Lorenz]  tau (1r min AMI) = {tau}  ->  {tau*dt:.2f} s")
print(f"          FNN: {np.round(fnn,4)}  -> m = {m}  (teoric 3)")
Xl=nld.embed(xs,m,tau)
sub=Xl[::3][:4000]
t,y=nld.rosenstein(sub,fs=fs/3,theiler=int(0.5*fs/3),k_max=int(1.5*fs/3),nref=600)
lam,_=nld.lyapunov_from_curve(t,y,fit_frac=(0.03,0.30))
D2,rs,C,win=nld.correlation_dimension(sub,theiler=int(0.5*fs/3),nref=700)
print(f"          lambda1 = {lam:6.3f}  (teoric 0.906)  err {abs(lam-0.906)/0.906*100:4.1f}%")
print(f"          D2      = {D2:6.3f}  (teoric 2.05)   err {abs(D2-2.05)/2.05*100:4.1f}%")
r=nld.rqa(sub[::4][:1200],rr_target=0.05,theiler=int(0.5*fs/3/4))
print(f"          RQA DET = {r['DET']:.3f} (determinista -> proper a 1)")

# ---------- 3. Soroll blanc gaussia : D2 ~ m ; DET baix ----------
w=rng.standard_normal(4000)
for m_ in (3,5):
    D2w,_,_,_=nld.correlation_dimension(nld.embed(w,m_,1),theiler=1,nref=700)
    print(f"\n[Soroll blanc m={m_}]  D2 = {D2w:5.2f}  (ha de creixer amb m, sense saturar)")
rw=nld.rqa(nld.embed(w,3,1)[:1200],rr_target=0.05,theiler=1)
print(f"[Soroll blanc]  RQA DET = {rw['DET']:.3f}  (ha de ser baix)")
sew=nld.sample_entropy(w[:2000],m=2,r=0.2); sel=nld.sample_entropy(x[:2000],m=2,r=0.2)
print(f"\n[SampEn] soroll blanc = {sew:.3f} (alt) | logistic = {sel:.3f} (baix, determinista)")

# ---------- 4. Surrogats IAAFT : conserven espectre i distribucio ----------
s=nld.iaaft(xs[:3000],n_iter=200,rng=3)
p1=np.abs(np.fft.rfft(xs[:3000]))**2; p2=np.abs(np.fft.rfft(s))**2
print(f"\n[IAAFT] correlacio espectres = {np.corrcoef(p1,p2)[0,1]:.4f} (ha de ser ~1)")
print(f"        distribucions identiques = {np.allclose(np.sort(s),np.sort(xs[:3000]))}")
D2s,_,_,_=nld.correlation_dimension(nld.embed(s,m,tau),theiler=int(0.5*fs),nref=600)
D2o,_,_,_=nld.correlation_dimension(nld.embed(xs[:3000],m,tau),theiler=int(0.5*fs),nref=600)
print(f"        D2 original = {D2o:.2f}  vs  D2 surrogat = {D2s:.2f}  (el surrogat ha de ser mes alt)")
