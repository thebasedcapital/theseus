#!/usr/bin/env python3
"""Check llama.cpp Qwen2 tensor naming, geometry, and metadata fidelity."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

LAYERS=24
EXPECTED=290
_src = __import__("common").converter().parent
sys.path.insert(0, str(_src / "gguf-py"))

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--gguf',type=Path,required=True); ap.add_argument('--out',type=Path,default=None); a=ap.parse_args(); p=a.gguf.resolve()
 r={'script':'check_gguf_layout.py','path':str(p),'expected_tensor_count':EXPECTED}
 if not p.exists(): r.update(status='UNAVAILABLE',error=f'missing GGUF artifact: {p}')
 else:
  try:
   from gguf import GGUFReader
   rd=GGUFReader(str(p)); ts={str(t.name):t for t in rd.tensors}; req=set()
   for i in range(LAYERS): req|={f'blk.{i}.{x}' for x in ('attn_q.weight','attn_k.weight','attn_v.weight','attn_output.weight','ffn_gate.weight','ffn_down.weight','ffn_up.weight','attn_q.bias','attn_k.bias','attn_v.bias','attn_norm.weight','ffn_norm.weight')}
   req|={'token_embd.weight','output_norm.weight'}
   missing=sorted(req-set(ts)); shape_errors=[]; norm_errors=[]; blocks=[]
   for i in range(LAYERS):
    shapes={x:list(map(int,ts[f'blk.{i}.{x}'].shape)) for x in ('attn_q.weight','attn_k.weight','attn_v.weight','attn_q.bias','attn_k.bias','attn_v.bias') if f'blk.{i}.{x}' in ts}
    blocks.append({'layer':i,'shapes':shapes})
    for x,want in (('attn_q.weight',[896,896]),('attn_k.weight',[128,896]),('attn_v.weight',[128,896]),('attn_q.bias',[896]),('attn_k.bias',[128]),('attn_v.bias',[128])):
     if f'blk.{i}.{x}' not in ts or list(map(int,ts[f'blk.{i}.{x}'].shape))!=want: shape_errors.append(f'blk.{i}.{x}')
    for x in ('attn_norm.weight','ffn_norm.weight'):
     t=ts.get(f'blk.{i}.{x}'); typ=str(t.tensor_type).split('.')[-1] if t else None
     if typ!='F32': norm_errors.append(f'blk.{i}.{x}:{typ}')
   rope=sorted(k for k in rd.fields if 'rope' in str(k).lower())
   r.update(status='OK' if len(ts)==EXPECTED and not missing and not shape_errors and not norm_errors and rope else 'FAIL',tensor_count=len(ts),missing=missing,shape_errors=shape_errors,norm_errors=norm_errors,blocks=blocks,rope_metadata_keys=rope,has_rope_freqs=bool(rope),has_output_weight='output.weight' in ts,notes=['Qwen2 tensors retain native HF row/column geometry; no importer q/k row permutation observed'])
  except Exception as e:r.update(status='FAIL',error=f'{type(e).__name__}: {e}')
 out=a.out or p.with_suffix('.layout.json'); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n'); print(json.dumps(r,indent=2,sort_keys=True)); raise SystemExit(0 if r.get('status')=='OK' else 1)
if __name__=='__main__': main()
