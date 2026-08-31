#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import torch
M1=Path(__file__).resolve().parent; REPO=M1.parent; EVAL=M1/'data'/'eval_wikitext.txt'; ROOT=M1/'work'/'fidelity'
TAGS=('base','g4_perm','g5_c8','g5_c8_rep','prep_base'); NBYTES=8192

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=M1/'work'/'fidelity.json'); ap.add_argument('--root',type=Path,default=ROOT); ap.add_argument('--tags',default=','.join(TAGS)); a=ap.parse_args()
 os.environ['OMP_NUM_THREADS']='4'; torch.set_num_threads(4); sys.path.insert(0,str(M1)); import common
 raw=EVAL.read_bytes()[:NBYTES]; results={}; cmds=[]; tstart=time.monotonic()
 for tag in [x.strip() for x in a.tags.split(',') if x.strip()]:
  d=common.REF_MODEL if tag=='base' else a.root/tag; d=Path(d).resolve(); tok=common.load_tokenizer(d); ids=torch.tensor(tok(raw.decode('utf-8',errors='replace'),add_special_tokens=False)['input_ids'],dtype=torch.long)
  if ids.numel()<2: results[tag]={'status':'UNAVAILABLE','torch_tokens':int(ids.numel()),'gguf_ppl':None,'gguf_tokens':None}; continue
  m=common.load_model(d,dtype=torch.bfloat16,device='cpu'); t=time.monotonic()
  with torch.inference_mode():
   lg=m(input_ids=ids[None,:]).logits[:, :-1].float(); target=ids[None,1:]; loss=torch.nn.functional.cross_entropy(lg.reshape(-1,lg.shape[-1]),target.reshape(-1)); ppl=float(loss.exp().item())
  elapsed=time.monotonic()-t; del m; common.release('cpu'); n=int(target.numel()); results[tag]={'status':'OK','model_dir':str(d),'torch_ppl':ppl,'torch_tokens':n,'gguf_ppl':None,'gguf_tokens':None,'ratio':None,'torch_elapsed_s':elapsed}; cmds.append(f'OMP_NUM_THREADS=4 {sys.executable} m1/fidelity.py --root {a.root} --tags {tag} --out {a.out}')
 obj={'script':'fidelity.py','git_head':subprocess.run(['git','-C',str(REPO),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),'torch':torch.__version__,'slice':{'source':str(EVAL),'byte_start':0,'byte_end_exclusive':len(raw),'bytes':len(raw)},'results':results,'cmds':cmds,'duration_s':time.monotonic()-tstart,'notes':['CPU torch half; GGUF fields null until llama phase','bf16 model, CPU, OMP_NUM_THREADS=4']}; a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n'); print(json.dumps(obj,indent=2,sort_keys=True))
if __name__=='__main__': main()
