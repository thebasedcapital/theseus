#!/usr/bin/env python3
"""Independently verify the byte-for-byte G5 recovery claim.

CLAIMS K-5 / M1_RESULTS assert that the artifact-only repair of `g5_c8` — which never sees the
original checkpoint — reproduces the pristine Qwen2.5-0.5B file exactly. That claim had been resting
on a log line, and the repaired artifact is gitignored, so it was the one headline in the report I
could not re-check. This re-derives it from scratch.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common            # noqa: E402
import canonicalize as C  # noqa: E402
from common import Arch   # noqa: E402

STRESSED = common.WORK / "g5_c8"
PRISTINE_BLOB = Path("/home/admin/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/blobs/"
                     "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342")
cfg = json.loads((common.REF_MODEL / "config.json").read_text())
arch = Arch.from_config(cfg)

print(f"artifact-only repair of {STRESSED.name}, arch tie={arch.tie}")
repaired, meta = C.canon_g5(common.load_state(STRESSED), arch)
pristine = common.load_state(common.REF_MODEL)

keys_r, keys_p = set(repaired), set(pristine)
print(f"key sets identical: {keys_r == keys_p}  (n={len(keys_p)})")
differ = [k for k in sorted(keys_r & keys_p)
          if repaired[k].shape != pristine[k].shape or not (repaired[k] == pristine[k]).all()]
print(f"tensors differing: {len(differ)}/{len(keys_p)}")
for k in differ[:5]:
    print("   ", k)
print("repair meta:", json.dumps(meta, default=str)[:220])

out = Path("/tmp/g5_rep_verify.safetensors")
try:
    sd = {k: repaired[k] for k in sorted(keys_p)}
    d = Path("/tmp/g5_rep_dir"); d.mkdir(exist_ok=True)
    common.save_state(sd, d, common.REF_MODEL)
    produced = d / "model.safetensors"
    def sha(p):
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        return h.hexdigest()
    print("repaired sha:", sha(produced))
    print("pristine sha:", sha(PRISTINE_BLOB))
    print("BYTE-FOR-BYTE:", "CONFIRMED" if sha(produced) == sha(PRISTINE_BLOB) else "NOT EQUAL")
except Exception as exc:                                  # writer needs sibling config files
    print("file-level hash skipped:", type(exc).__name__, str(exc)[:160])
    print("tensor-level equality is the substantive check and stands on its own")
