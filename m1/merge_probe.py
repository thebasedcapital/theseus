#!/usr/bin/env python3
"""Task-vector merge reserve: deterministic LoRA specialist, linear and TIES merges."""
from __future__ import annotations
import argparse, json, random, subprocess, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import common
common._LOAD_KW = {}
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 2718
RANK = 32
ALPHA = 32
STEPS = 600
BATCH_SIZE = 2
SEQ_LEN = 128
TRAIN_N = 512
HELDOUT_N = 128
DENSITY = 0.2
ALPHAS = (0.3, 0.5, 0.7)
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SPECIALIST_DIR = common.WORK / "specialist"
MARKER = SPECIALIST_DIR / "specialist.json"
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int):
        super().__init__(); self.base = base; self.base.weight.requires_grad_(False)
        if base.bias is not None: base.bias.requires_grad_(False)
        dev = base.weight.device
        self.a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=dev))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=dev))
        nn.init.kaiming_uniform_(self.a, a=5**0.5); self.scale = alpha / rank
    def forward(self, x):
        return self.base(x) + (F.linear(F.linear(x.float(), self.a), self.b)*self.scale).to(x.dtype)

def replace_targets(model):
    for n,c in list(model.named_children()):
        if isinstance(c,nn.Linear) and n in TARGETS: setattr(model,n,LoRALinear(c,RANK,ALPHA))
        else: replace_targets(c)

def set_seed():
    random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def examples(n):
    rng=random.Random(SEED); out=[]
    for _ in range(n):
        key=''.join(rng.choice('abcdefghij') for _ in range(8)); val=''.join(rng.choice('0123456789') for _ in range(8)); out.append((key,val))
    return out

def make_data(tok, pairs):
    rows=[]; labels=[]; masks=[]; pad=tok.eos_token_id
    for key,val in pairs:
        pre=f"kv: {key}={val} => "; tgt=f"{key}: {val}"; p=tok(pre,add_special_tokens=False)['input_ids']; t=tok(tgt,add_special_tokens=False)['input_ids']+[pad]
        x=(p+t)[:SEQ_LEN]; y=([-100]*len(p)+t)[:SEQ_LEN]; m=[1]*len(x)
        rows.append(x+[pad]*(SEQ_LEN-len(x))); labels.append(y+[-100]*(SEQ_LEN-len(y))); masks.append(m+[0]*(SEQ_LEN-len(m)))
    return tuple(torch.tensor(z,dtype=torch.long) for z in (rows,labels,masks))

def task_loss(model,data,device):
    x,y,m=(z.to(device) for z in data); total=n=0; model.eval()
    with torch.no_grad():
        for i in range(0,len(x),BATCH_SIZE):
            o=model(input_ids=x[i:i+BATCH_SIZE],attention_mask=m[i:i+BATCH_SIZE]); lg=o.logits[:,:-1].float(); tg=y[i:i+BATCH_SIZE,1:]
            total+=F.cross_entropy(lg.reshape(-1,lg.size(-1)),tg.reshape(-1),ignore_index=-100,reduction='sum').item(); n+=int((tg!=-100).sum())
    return total/max(1,n)

def train_specialist(tok,train,held,device):
    set_seed(); model=common.load_model(common.REF_MODEL,dtype=torch.bfloat16,device=device); model.config.use_cache=False; replace_targets(model)
    params=[p for p in model.parameters() if p.requires_grad]; opt=torch.optim.AdamW(params,lr=3e-3,weight_decay=0.0)
    x,y,m=train; model.train(); t0=time.perf_counter()
    for step in range(STEPS):
        i=(step*BATCH_SIZE)%len(x); xb,yb,mb=x[i:i+BATCH_SIZE].to(device),y[i:i+BATCH_SIZE].to(device),m[i:i+BATCH_SIZE].to(device)
        o=model(input_ids=xb,attention_mask=mb); loss=F.cross_entropy(o.logits[:,:-1].float().reshape(-1,o.logits.size(-1)),yb[:,1:].reshape(-1),ignore_index=-100); loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step(); opt.zero_grad(set_to_none=True)
    if device=='cuda': torch.cuda.synchronize()
    runtime=time.perf_counter()-t0
    # Plain BA merge in-place into a bf16 state dict.
    sd=common.load_state(common.REF_MODEL)
    with torch.no_grad():
        for name,mod in model.named_modules():
            if isinstance(mod,LoRALinear):
                key=name+'.weight'; ba=mod.b.float().cpu() @ mod.a.float().cpu(); sd[key]=(sd[key].float()+ba * mod.scale).to(torch.bfloat16); del ba
    spec_sd={k:v.detach().to("cpu").clone() for k,v in sd.items()}
    common.save_state(spec_sd,SPECIALIST_DIR,common.REF_MODEL)
    quality=task_loss(model,held,device)
    batches=[b.to(device) for b in common.eval_batches(common.REF_MODEL,ntokens=2048,seqlen=512)]; ppl=common.perplexity(model,batches)
    batches=[b.to(device) for b in common.eval_batches(common.REF_MODEL,ntokens=2048,seqlen=128)]; ppl=common.perplexity(model,batches)
    marker={'seed':SEED,'rank':RANK,'alpha':ALPHA,'steps':STEPS,'batch_size':BATCH_SIZE,'seq_len':SEQ_LEN,'train_examples':TRAIN_N,'heldout_examples':HELDOUT_N,'rule':'key:value reformat: kv: KEY=VALUE => KEY: VALUE','rule_loss':quality,'heldout_rule_loss':quality,'eval_ppl':ppl,'runtime_s':runtime,'max_memory_allocated_gb':torch.cuda.max_memory_allocated()/1e9 if device=='cuda' else 0.0}
    del opt,model,sd,spec_sd
    common.release(device)
    return marker

def ensure_specialist(tok,train,held,device):
    if (SPECIALIST_DIR/'model.safetensors').exists() and MARKER.exists():
        try:
            m=common.rjson(MARKER); expected={'seed':SEED,'rank':RANK,'alpha':ALPHA,'steps':STEPS,'batch_size':BATCH_SIZE,'seq_len':SEQ_LEN,'train_examples':TRAIN_N,'heldout_examples':HELDOUT_N}
            if all(m.get(k)==v for k,v in expected.items()): return m
        except Exception: pass
    return train_specialist(tok,train,held,device)

    batches=[b.to(device) for b in common.eval_batches(common.REF_MODEL,ntokens=2048,seqlen=128)]; cand_sd={k:v.to("cpu") for k,v in cand_sd.items()}; specialist_sd={k:v.to("cpu") for k,v in specialist_sd.items()}; base=common.state_to_model(cand_sd,model_dir,dtype=torch.bfloat16,device=device); base_ppl=common.perplexity(base,batches); del base
    common.release(device)
    out=[]
    for alpha in alphas:
        merged=common.merge_sd(cand_sd,specialist_sd,alpha,ties=ties,density=DENSITY)
        model=common.state_to_model(merged,model_dir,dtype=torch.bfloat16,device=device)
        ppl=common.perplexity(model,batches); loss=task_loss(model,HELD_DATA,device)
        out.append({'alpha':alpha,'eval_ppl':ppl,'specialist_rule_loss':loss,'pass':ppl<=1.05*base_ppl and loss<=0.7*SPECIALIST_QUALITY['rule_loss']})
        del model,merged
        common.release(device)
    return base_ppl,out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model-dir',required=True); ap.add_argument('--out',required=True); ap.add_argument('--tags',default='')
    args=ap.parse_args(); t0=time.perf_counter(); model_dir=Path(args.model_dir).expanduser().resolve(); out=Path(args.out)
    result={'script':'merge_probe.py','model_dir':str(model_dir),'tag':model_dir.name if common.WORK in model_dir.parents else 'base','git_head':subprocess.check_output(['git','-C','/home/admin/theseus','rev-parse','HEAD'],text=True).strip(),'torch':torch.__version__,'results':{},'duration_s':None}
    try:
        global HELD_DATA,SPECIALIST_QUALITY
        set_seed(); device=common.pick_device(3.2); device_note=device if device == 'cuda' else 'cpu: insufficient free CUDA memory or CUDA unavailable'
        with common.lock("gpu"):
            while True:
                device=common.pick_device(2.4)
                if device == "cuda": break
                common.log("waiting for gpu")
                time.sleep(30)
            device_note=device
            tok=common.load_tokenizer(common.REF_MODEL); ex=examples(TRAIN_N+HELDOUT_N)
            train=make_data(tok,ex[:TRAIN_N]); HELD_DATA=make_data(tok,ex[TRAIN_N:])
            SPECIALIST_QUALITY=ensure_specialist(tok,train,HELD_DATA,device)
            cand=common.load_state(model_dir); spec=common.load_state(SPECIALIST_DIR)
            linear_base,linear=evaluate_merge(cand,spec,model_dir,ALPHAS,device,False)
            del cand, spec
            common.release(device)
            ties_base,ties=evaluate_merge(common.load_state(model_dir),common.load_state(SPECIALIST_DIR),model_dir,ALPHAS,device,True)
            def smallest(rows): return min((r['alpha'] for r in rows if r['pass']),default=None)
            result['results']={'seed':SEED,'rule':SPECIALIST_QUALITY['rule'],'steps':STEPS,'lora_rank':RANK,'lora_alpha':ALPHA,'density':DENSITY,'alphas':list(ALPHAS),'specialist':SPECIALIST_QUALITY,'candidate_ppl':linear_base,'linear':{'matrix':linear,'smallest_passing_alpha':smallest(linear)},'ties':{'matrix':ties,'smallest_passing_alpha':smallest(ties)},'device':device,'device_note':device_note}
    except Exception as e: result['error']={'type':type(e).__name__,'message':str(e)}
    result['duration_s']=time.perf_counter()-t0; common.wjson(out,result); print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__': main()
