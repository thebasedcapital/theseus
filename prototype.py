#!/usr/bin/env python3
"""Theseus V0: deterministic, no-LLM model-optionality smoke test."""
from __future__ import annotations

import copy, csv, json, math, random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

SEED = 7
OUT = Path(__file__).resolve().parent
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(1)

class TinyReLU(nn.Module):
    def __init__(self, hidden=64):
        super().__init__(); self.fc1 = nn.Linear(64, hidden); self.fc2 = nn.Linear(hidden, 10)
    def forward(self, x): return self.fc2(torch.relu(self.fc1(x)))

@dataclass
class Result:
    name: str; current_accuracy: float; max_logit_diff_vs_base: float; gauge_imbalance_max: float
    q4_accuracy: float; q4_pass: bool; prune40_accuracy: float; prune40_pass: bool
    adapt_pass: bool; adapt_min_steps: int | None; adapt_lr: float | None
    adapt_shift_accuracy: float | None; adapt_retained_accuracy: float | None
    merge_pass: bool; merge_alpha: float | None; merge_original_accuracy: float | None
    merge_rotated_accuracy: float | None; optionality_passes: int; optionality_total: int

def data():
    x, y = load_digits(return_X_y=True); x = x.astype(np.float32) / 16
    a,b,c,d = train_test_split(x,y,test_size=.25,random_state=SEED,stratify=y)
    return torch.tensor(a), torch.tensor(b), torch.tensor(c,dtype=torch.long), torch.tensor(d,dtype=torch.long)

def rot(x): return torch.rot90(x.view(-1,8,8),1,[1,2]).reshape(-1,64)
def shift(x): return torch.roll(x.view(-1,8,8),1,2).reshape(-1,64)
def acc(m,x,y):
    with torch.no_grad(): return (m(x).argmax(1)==y).float().mean().item()
def logit_diff(a,b,x):
    with torch.no_grad(): return (a(x)-b(x)).abs().max().item()

def train(m,x,y,epochs=70,lr=1e-2):
    o=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4)
    for _ in range(epochs):
        for idx in torch.randperm(len(x)).split(128):
            loss=nn.functional.cross_entropy(m(x[idx]),y[idx]); o.zero_grad(); loss.backward(); o.step()
    return m

def specialist(base,x,xr,y):
    m=copy.deepcopy(base); o=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=1e-4)
    for _ in range(35):
        for xb in (xr,xr,x):
            for idx in torch.randperm(len(xb)).split(128):
                loss=nn.functional.cross_entropy(m(xb[idx]),y[idx]); o.zero_grad(); loss.backward(); o.step()
    return m

def bad_gauge(m,spread=100.):
    m=copy.deepcopy(m); h=m.fc1.out_features
    d=torch.logspace(-math.log10(spread),math.log10(spread),h)
    d=d[torch.randperm(h,generator=torch.Generator().manual_seed(123))]
    with torch.no_grad(): m.fc1.weight.mul_(d[:,None]); m.fc1.bias.mul_(d); m.fc2.weight.div_(d[None,:])
    return m

def gauge_fix(m,eps=1e-12):
    m=copy.deepcopy(m)
    with torch.no_grad():
        a=torch.sqrt((m.fc1.weight**2).sum(1)+m.fc1.bias**2+eps)
        b=torch.sqrt((m.fc2.weight**2).sum(0)+eps); s=torch.sqrt(b/a)
        m.fc1.weight.mul_(s[:,None]); m.fc1.bias.mul_(s); m.fc2.weight.div_(s[None,:])
    return m

def imbalance(m,eps=1e-12):
    with torch.no_grad():
        a=torch.sqrt((m.fc1.weight**2).sum(1)+m.fc1.bias**2+eps)
        b=torch.sqrt((m.fc2.weight**2).sum(0)+eps)
        return torch.abs(torch.log(a/b)).max().item()

def qt(t,bits=4):
    qmax=2**(bits-1)-1; mx=t.abs().max()
    if mx==0: return t.clone()
    s=mx/qmax; return torch.clamp(torch.round(t/s),-qmax,qmax)*s

def quant(m,bits=4):
    q=copy.deepcopy(m)
    with torch.no_grad():
        for p in q.parameters(): p.copy_(qt(p,bits))
    return q

def prune(m,f=.4):
    p=copy.deepcopy(m); ws=[p.fc1.weight,p.fc2.weight]
    vals=torch.cat([w.detach().abs().flatten() for w in ws]); k=max(1,int(f*vals.numel()))
    th=torch.kthvalue(vals,k).values
    with torch.no_grad():
        for w in ws: w.mul_(w.abs()>th)
    return p

def adapt_probe(m,xs,xst,xo,y,yt):
    for lr in [1e-4,3e-4,1e-3,3e-3,1e-2,3e-2]:
        c=copy.deepcopy(m); o=torch.optim.SGD(c.parameters(),lr=lr); g=torch.Generator().manual_seed(999)
        for step in range(1,101):
            idx=torch.randint(0,len(xs),(128,),generator=g)
            loss=nn.functional.cross_entropy(c(xs[idx]),y[idx]); o.zero_grad(); loss.backward(); o.step()
            if not math.isfinite(loss.item()): break
            if step%5==0:
                ta,re=acc(c,xst,yt),acc(c,xo,yt)
                if ta>=.52 and re>=.95: return dict(steps=step,lr=lr,target_accuracy=ta,retained_accuracy=re)
    return None

def merge(a,b,alpha):
    o=copy.deepcopy(a)
    with torch.no_grad():
        for po,pa,pb in zip(o.parameters(),a.parameters(),b.parameters()): po.copy_((1-alpha)*pa+alpha*pb)
    return o

def merge_probe(m,s,xo,xr,y):
    for alpha in np.linspace(.5,.9,9):
        z=merge(m,s,float(alpha)); ao,ar=acc(z,xo,y),acc(z,xr,y)
        if ao>=.90 and ar>=.80: return dict(alpha=float(alpha),original_accuracy=ao,rotated_accuracy=ar)
    return None

def evaluate(name,m,base,s,pack):
    xtr,xte,ytr,yte,xr,xre,xs,xse=pack
    cur=acc(m,xte,yte); q=acc(quant(m),xte,yte); p=acc(prune(m),xte,yte)
    a=adapt_probe(m,xs,xse,xte,ytr,yte); z=merge_probe(m,s,xte,xre,yte)
    flags=(q>=.95,p>=.95,a is not None,z is not None)
    return Result(name,cur,logit_diff(base,m,xte),imbalance(m),q,flags[0],p,flags[1],flags[2],
                  a['steps'] if a else None,a['lr'] if a else None,a['target_accuracy'] if a else None,
                  a['retained_accuracy'] if a else None,flags[3],z['alpha'] if z else None,
                  z['original_accuracy'] if z else None,z['rotated_accuracy'] if z else None,sum(flags),4)

def save(rs,spec):
    payload={'seed':SEED,'definition':{
        'q4_pass':'original test accuracy >= 0.95 after per-tensor symmetric int4 quantization',
        'prune40_pass':'original test accuracy >= 0.95 after 40% global magnitude pruning',
        'adapt_pass':'within 100 SGD steps and LR grid, shifted-domain accuracy >= 0.52 while original accuracy >= 0.95',
        'merge_pass':'for alpha in [0.50, 0.90], original accuracy >= 0.90 and rotated-domain accuracy >= 0.80'},
        'specialist':spec,'results':[asdict(r) for r in rs]}
    (OUT/'results.json').write_text(json.dumps(payload,indent=2))
    with (OUT/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(asdict(rs[0]))); w.writeheader(); [w.writerow(asdict(r)) for r in rs]
    lines=['Theseus V0 smoke test','=====================','']
    for r in rs:
        lines += [r.name,f'  current accuracy      : {r.current_accuracy:.4f}',
          f'  max logit diff vs base: {r.max_logit_diff_vs_base:.3e}',f'  gauge imbalance max   : {r.gauge_imbalance_max:.3f}',
          f'  Q4 accuracy/pass      : {r.q4_accuracy:.4f} / {r.q4_pass}',f'  prune40 accuracy/pass : {r.prune40_accuracy:.4f} / {r.prune40_pass}',
          f'  adapt capture         : {r.adapt_pass} steps={r.adapt_min_steps} lr={r.adapt_lr}',
          f'  merge capture         : {r.merge_pass} alpha={r.merge_alpha}',f'  optionality           : {r.optionality_passes}/{r.optionality_total}','']
    (OUT/'smoke_output.txt').write_text('\n'.join(lines))
    plt.figure(figsize=(8,4.5)); plt.bar([r.name for r in rs],[r.optionality_passes/4 for r in rs]); plt.ylim(0,1.05)
    plt.ylabel('Fraction of future surgery capture tests passed'); plt.title('Same present behavior, different model optionality')
    plt.tight_layout(); plt.savefig(OUT/'optionality.png',dpi=180); plt.close()

def main():
    xtr,xte,ytr,yte=data(); xr,xre=rot(xtr),rot(xte); xs,xse=shift(xtr),shift(xte)
    base=train(TinyReLU(),xtr,ytr); s=specialist(base,xtr,xr,ytr); bad=bad_gauge(base); fixed=gauge_fix(bad)
    pack=(xtr,xte,ytr,yte,xr,xre,xs,xse)
    rs=[evaluate('base',base,base,s,pack),evaluate('same-function / bad-gauge',bad,base,s,pack),evaluate('gauge-fixed',fixed,base,s,pack)]
    save(rs,{'original_accuracy':acc(s,xte,yte),'rotated_accuracy':acc(s,xre,yte)})
    assert rs[1].max_logit_diff_vs_base<1e-4 and abs(rs[1].current_accuracy-rs[0].current_accuracy)<1e-9
    assert [r.optionality_passes for r in rs]==[4,0,4]
    print((OUT/'smoke_output.txt').read_text())

if __name__=='__main__': main()
