# -Adaptive-Reservoir-Canon-v6.2-
# =========================================================
# 🧬 FINAL CANON v6.2 — MULTITHREAD / FAST GOLDEN ZONE
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, json, hashlib
from datetime import datetime
from joblib import Parallel, delayed

# ================= REPRODUCIBILITY =================
SEED = 777
np.random.seed(SEED)

# ================= CONFIG =================
CONFIG = {
    "N_esn": 300,
    "N_phoenix": 500,
    "N_adaptive": 600,
    "wash": 100,
    "train": 800,
    "test": 400,
    "horizons": [1,5,10,20],
    "runs": 3,
    "ensemble": 3,
    "sr_list": [0.97, 0.99, 1.01],
    "leak_list": [0.44, 0.45, 0.46],
    "ridge": [1e-6,1e-5,1e-4],
    "adaptive": {
        "sr": 0.98,
        "leak": 0.45,
        "eta": 0.01,
        "fb": 0.05
    },
    "n_jobs": -1  # auto all cores
}

# ================= HASH & SAVE =================
def hash_config(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

RUN_HASH = hash_config(CONFIG)
SAVE_DIR = f"./FINAL_V6_2_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{RUN_HASH[:6]}"
os.makedirs(SAVE_DIR, exist_ok=True)
with open(os.path.join(SAVE_DIR,"config.json"),"w") as f:
    json.dump(CONFIG,f,indent=2)

# ================= DATA =================
def mackey_glass(length, tau=30):
    x = np.zeros(length)
    x[0] = 1.2
    for t in range(1,length):
        xtau = x[t-tau] if t>=tau else 1.2
        x[t] = x[t-1] + 0.1*(0.2*xtau/(1+xtau**10) - 0.1*x[t-1])
    return x

def normalize(x):
    return (x - np.mean(x)) / (np.std(x)+1e-12)

def nmse(y,yhat):
    return np.mean((y-yhat)**2)/(np.var(y)+1e-12)

# ================= RESERVOIR =================
class Reservoir:
    def __init__(self,N,sr,leak):
        W = np.random.randn(N,N)/np.sqrt(N)
        W *= sr/(np.max(np.abs(np.linalg.eigvals(W)))+1e-12)
        self.W=W
        self.Win=0.1*np.random.randn(N)
        self.state=np.zeros(N)
        self.leak=leak

    def step(self,u):
        x=np.tanh(self.W@self.state + self.Win*u)
        self.state=(1-self.leak)*self.state + self.leak*x
        return self.state

    def run(self,u):
        return np.array([self.step(ui) for ui in u]).T

# ================= ADAPTIVE =================
class AdaptiveReservoir:
    def __init__(self,N,sr,leak,eta,fb):
        W=np.random.randn(N,N)/np.sqrt(N)
        W*=sr/(np.max(np.abs(np.linalg.eigvals(W)))+1e-12)
        self.W=W
        self.target_sr=sr
        self.Win=0.1*np.random.randn(N)
        self.state=np.zeros(N)
        self.leak=leak
        self.eta=eta
        self.fb=fb
        self.prev=0.01

    def step(self,u,error=0):
        pre=self.W@self.state + self.Win*u
        pre+=self.fb*np.tanh(error)
        x=np.tanh(pre)
        new=(1-self.leak)*self.state + self.leak*x

        act=np.mean(np.abs(new))
        ratio=act/(self.prev+1e-8)
        adapt=(1-ratio)*(1+np.abs(np.tanh(error)))
        self.W*=(1+self.eta*adapt)

        eig=np.linalg.eigvals(self.W)
        sr=np.max(np.abs(eig))
        self.W*=self.target_sr/(sr+1e-12)

        self.prev=act
        self.state=new
        return self.state

    def run(self,u,y=None,beta=None,split=None):
        states=[]
        for t,ui in enumerate(u):
            err=0
            if y is not None and beta is not None and split and t>=split:
                err=y[t]-beta@self.state
            states.append(self.step(ui,err))
        return np.array(states).T

# ================= TRAIN =================
def train_readout(X,y):
    best=None; best_score=1e9
    for lam in CONFIG["ridge"]:
        beta=np.linalg.solve(X.T@X + lam*np.eye(X.shape[1]), X.T@y)
        score=nmse(y,X@beta)
        if score<best_score:
            best_score=score
            best=beta
    return best

# ================= LYAPUNOV =================
def lyapunov_qr(W, states):
    Q = np.eye(states.shape[0])
    lyap = 0.0
    steps = min(200, states.shape[1])
    for t in range(steps):
        J = W @ np.diag(1 - states[:,t]**2)
        M = J @ Q
        Q,R = np.linalg.qr(M)
        lyap += np.log(np.abs(R[0,0])+1e-12)
    return lyap/steps

# ================= SINGLE RUN FUNCTION =================
def single_run(seed,H):
    np.random.seed(SEED+seed)
    data_full = normalize(mackey_glass(CONFIG["wash"]+CONFIG["train"]+CONFIG["test"]+50+max(CONFIG["horizons"])))
    u = data_full[:-H]; y = data_full[H:]
    recs=[]

    for sr in CONFIG["sr_list"]:
        for leak in CONFIG["leak_list"]:
            # ESN
            res=Reservoir(CONFIG["N_esn"],sr,leak)
            states=res.run(u)
            Xtr=states[:,CONFIG["wash"]:CONFIG["wash"]+CONFIG["train"]].T
            ytr=y[CONFIG["wash"]:CONFIG["wash"]+CONFIG["train"]]
            beta=train_readout(Xtr,ytr)
            Xte=states[:,CONFIG["wash"]+CONFIG["train"]:]
            pred=Xte.T@beta
            score=nmse(y[CONFIG["wash"]+CONFIG["train"]:],pred)
            if score<1: recs.append(["ESN",sr,leak,H,score,lyapunov_qr(res.W,Xte)])

            # PHOENIX
            res=Reservoir(CONFIG["N_phoenix"],sr,leak)
            states=res.run(u)
            Xtr=states[:,CONFIG["wash"]:CONFIG["wash"]+CONFIG["train"]].T
            ytr=y[CONFIG["wash"]:CONFIG["wash"]+CONFIG["train"]]
            beta=train_readout(Xtr,ytr)
            beta=0.98*beta + 0.02*np.tanh(beta)
            Xte=states[:,CONFIG["wash"]+CONFIG["train"]:]
            pred=Xte.T@beta
            score=nmse(y[CONFIG["wash"]+CONFIG["train"]:],pred)
            if score<1: recs.append(["PHOENIX",sr,leak,H,score,lyapunov_qr(res.W,Xte)])

    # ADAPTIVE
    nmse_list=[]; lyap_list=[]
    for ens in range(CONFIG["ensemble"]):
        np.random.seed(SEED+seed+ens)
        res=AdaptiveReservoir(CONFIG["N_adaptive"],CONFIG["adaptive"]["sr"],CONFIG["adaptive"]["leak"],CONFIG["adaptive"]["eta"],CONFIG["adaptive"]["fb"])
        split = CONFIG["wash"]+CONFIG["train"]
        states=res.run(u[:split])
        Xtr=states[:,CONFIG["wash"]:split].T
        ytr=y[CONFIG["wash"]:split]
        beta=train_readout(Xtr,ytr)
        states=res.run(u[split:],y,beta,split=split)
        pred=states.T@beta
        score=nmse(y[split:],pred)
        if score<1:
            nmse_list.append(score)
            lyap_list.append(lyapunov_qr(res.W,states))
    if nmse_list:
        recs.append(["ADAPTIVE",0,0,H,np.median(nmse_list),np.median(lyap_list)])
    return recs

# ================= PARALLEL RUN =================
results=[]
for H in CONFIG["horizons"]:
    recs_list=Parallel(n_jobs=CONFIG["n_jobs"])(delayed(single_run)(seed,H) for seed in range(CONFIG["runs"]))
    for recs in recs_list: results.extend(recs)

# ================= SAVE =================
df=pd.DataFrame(results,columns=["Model","SR","Leak","Horizon","NMSE","Lyapunov"])
df.to_csv(os.path.join(SAVE_DIR,"results.csv"),index=False)
print(df.groupby("Model")[["NMSE","Lyapunov"]].agg(["mean","median","std"]))

# ================= HISTOGRAM =================
plt.figure(figsize=(12,5))
for i,m in enumerate(["ESN","PHOENIX","ADAPTIVE"],1):
    plt.subplot(1,3,i)
    sub=df[df.Model==m]["NMSE"]
    plt.hist(sub,bins=30)
    plt.title(m)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,"hist.png"))

# ================= BOXPLOT =================
plt.figure()
data_plot=[df[df.Model==m]["NMSE"] for m in ["ESN","PHOENIX","ADAPTIVE"]]
plt.boxplot(data_plot, labels=["ESN","PHOENIX","ADAPTIVE"])
plt.yscale("log")
plt.savefig(os.path.join(SAVE_DIR,"boxplot.png"))

print(f"\n✅ FINAL v6.2 COMPLETE\n📁 {SAVE_DIR}")
