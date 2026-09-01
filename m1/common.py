#!/usr/bin/env python3
"""M1 shared plumbing: Qwen2 (decoder-only, GQA, RMSNorm, SwiGLU, tied head) artifact I/O + metrics.

Every M1 script is CLI-shaped: it takes `--model-dir <HF dir>` and writes a JSON result.
That keeps the gauge library, the equivalence verifier, and the surgery probes independent
of each other: a *variant* is just a directory of safetensors.

Interpreter: create the project's own environment; nothing here reaches into a sibling
checkout.

    python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
    ./.venv/bin/python m1/<script>.py ...
    # or point at any suitable interpreter:  THESEUS_PY=/path/to/python

requirements.txt is CPU torch by default (~1.7 GB); the recorded CUDA cells need a CUDA build to
reproduce byte-for-byte, which is stated in that file rather than assumed.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

M1 = Path(__file__).resolve().parent
REPO = M1.parent
# THESEUS_WORK exists so a second architecture cannot write cells into the first one's evidence
# directory. Incident #20 showed means computed across different code are meaningless; a shared
# m1/work would produce the same failure across different MODELS, silently, with colliding variant
# names. Default is unchanged, so every recorded Qwen2 cell keeps resolving.
WORK = Path(os.environ.get("THESEUS_WORK") or (M1 / "work"))  # scratch (gitignored)
DATA = M1 / "data"

# THESSUS_REF_MODEL is the knob weakness #2 was missing: REF_MODEL was a hardcoded literal, so the
# whole pipeline could only ever measure one checkpoint and the cross-architecture work stayed a
# paper audit. Default is the same Qwen2.5-0.5B snapshot, unchanged, so existing cells stay valid.
REF_MODEL = Path(os.path.expanduser(os.environ.get(
    "THESEUS_REF_MODEL",
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
)))
EVAL_TEXT = DATA / "eval_wikitext.txt"

# --- state-dict tensor names that carry meaning for the Qwen2 gauge families -------------
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight$")
NORM_KEYS = ("input_layernorm.weight", "post_attention_layernorm.weight")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def pick_device(need_gb: float = 3.6) -> str:
    """The GPU is shared with the desktop and sibling agents: take it only if it really
    has room, otherwise run on CPU (slow but correct) instead of OOM-killing a sibling."""
    if os.environ.get("TSX_CPU") or not torch.cuda.is_available():
        return "cpu"
    try:
        free, _ = torch.cuda.mem_get_info()
    except Exception:
        return "cpu"
    return "cuda" if free >= need_gb * 1e9 else "cpu"


def release(dev: str | None = None):
    import gc
    gc.collect()
    if (dev or "").startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


class Lock:
    """mkdir-based advisory mutex so sibling processes serialize GPU/CPU-heavy phases.

    One 8 GB GPU and 8 cores are shared between the desktop, llama.cpp and several torch
    probes; concurrent big loads OOM each other and thrash cores, so contention is made
    cooperative. Staleness (default 30 min) lets a crashed holder's lock be stolen rather
    than deadlock. Names in use: "gpu", "cpu".
    """

    def __init__(self, name: str = "gpu", timeout: float = 5400.0, stale_s: float = 1800.0):
        self.path = WORK / f"{name}.lock"
        self.timeout, self.stale_s, self.held = timeout, stale_s, False

    def __enter__(self):
        t0 = time.time()
        WORK.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                self.path.mkdir()
                (self.path / "owner").write_text(
                    f"pid={os.getpid()} start={time.strftime('%F %T')} "
                    f"cmd={' '.join(sys.argv[:3])}\n")
                self.held = True
                return self
            except FileExistsError:
                if self._holder_dead():
                    log(f"lock {self.path.name}: holder is dead, stealing")
                    shutil.rmtree(self.path, ignore_errors=True)
                    continue
                try:
                    age = time.time() - self.path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age > self.stale_s:
                    log(f"lock {self.path.name}: stealing after {age:.0f}s idle")
                    shutil.rmtree(self.path, ignore_errors=True)
                    continue
                if time.time() - t0 > self.timeout:
                    raise TimeoutError(f"could not acquire {self.path} in {self.timeout}s")
                time.sleep(5)

    def _holder_dead(self) -> bool:
        """True when the recorded owner process no longer exists (or its cmdline no longer
        matches what it claimed). A crashed holder must not stall the panel for stale_s."""
        try:
            fields = (self.path / "owner").read_text().split()
            pid = int(fields[0].split("=", 1)[1])
            claimed = " ".join(fields[2:]) if len(fields) > 2 else ""
        except Exception:                                    # noqa: BLE001
            return False
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            return True                                      # pid gone
        except Exception:                                    # noqa: BLE001
            return False
        if not cmdline.strip():
            return False                                     # zombie/parent: leave it alone
        if claimed:
            head = claimed.split()[0]
            return head not in cmdline
        return False

    def __exit__(self, *exc):
        if self.held:
            shutil.rmtree(self.path, ignore_errors=True)
            self.held = False
        return False


def lock(name: str = "gpu", **kw) -> Lock:
    return Lock(name, **kw)


@dataclass
class Arch:
    """Geometry needed to apply exact gauges. Read from config, never assumed."""
    hidden: int
    n_q: int
    n_kv: int
    head_dim: int
    layers: int
    intermediate: int
    group: int          # q heads per kv head
    tie: bool

    @staticmethod
    def from_config(cfg: dict) -> "Arch":
        hidden = int(cfg["hidden_size"])
        n_q = int(cfg["num_attention_heads"])
        n_kv = int(cfg.get("num_key_value_heads", n_q))
        head_dim = int(cfg.get("head_dim") or hidden // n_q)
        return Arch(
            hidden=hidden, n_q=n_q, n_kv=n_kv, head_dim=head_dim,
            layers=int(cfg["num_hidden_layers"]),
            intermediate=int(cfg["intermediate_size"]),
            group=n_q // n_kv, tie=bool(cfg.get("tie_word_embeddings", False)),
        )


def read_arch(model_dir: Path) -> Arch:
    return Arch.from_config(json.loads((Path(model_dir) / "config.json").read_text()))


def n_layers(sd: dict) -> int:
    return 1 + max(int(LAYER_RE.match(k).group(1)) for k in sd if LAYER_RE.match(k))


def head_of(h: int, arch: Arch) -> int:
    """kv-group serving query head h (matches HF `repeat_kv` interleaving)."""
    return h // arch.group


# --- artifact I/O ------------------------------------------------------------------------

_LOAD_KW = dict(disable_tqdm=True)


def load_state(model_dir: Path, dtype=torch.bfloat16) -> dict:
    """State dict (safetensors shards merged), copied out and cast."""
    from safetensors.torch import load_file
    model_dir = Path(model_dir)
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no safetensors in {model_dir}")
    sd = {}
    for f in files:
        for k, v in load_file(str(f)).items():
            sd[k] = v.to(dtype).clone()
    return sd


def save_state(sd: dict, out_dir: Path, ref_dir: Path = REF_MODEL,
               config_patch: dict | None = None) -> Path:
    """Write a loadable HF dir: weights + config/tokenizer copied from `ref_dir`."""
    from safetensors.torch import save_file
    from transformers import AutoConfig
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = AutoConfig.from_pretrained(str(ref_dir))
    for k, v in (config_patch or {}).items():
        setattr(cfg, k, v)
    # tied artifacts store embed_tokens only; untied must store lm_head too
    tie = bool(getattr(cfg, "tie_word_embeddings", False))
    tensors = dict(sd)
    if tie:
        tensors.pop("lm_head.weight", None)
    save_file({k: v.contiguous() for k, v in tensors.items()},
              str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    cfg.save_pretrained(str(out_dir))
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                 "special_tokens_map.json", "chat_template.jinja", "generation_config.json",
                 "added_tokens.json"):
        src = Path(ref_dir) / name
        if src.exists():
            (out_dir / name).write_bytes(src.read_bytes())
    return out_dir


_LOAD_KW: dict = {}


def load_model(model_dir: Path, dtype=torch.float32, device: str | None = None):
    from transformers import AutoModelForCausalLM
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dtype, **_LOAD_KW)
    return m.to(device).eval()


def load_tokenizer(model_dir: Path = REF_MODEL):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model_dir))


def state_to_model(sd: dict, model_dir: Path, dtype=torch.float32, device: str | None = None):
    """Load a live model straight from a state dict (no scratch write)."""
    m = load_model(model_dir, dtype=dtype, device=device)
    sd = dict(sd)
    exp = set(m.state_dict())
    if "lm_head.weight" in exp and "lm_head.weight" not in sd and "model.embed_tokens.weight" in sd:
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"]        # honour the tie
    missing, unexpected = m.load_state_dict(sd, strict=False)
    bad = [k for k in list(missing) + list(unexpected) if "rotary" not in k and "inv_freq" not in k]
    if bad:
        raise SystemExit(f"state_dict/model key mismatch: missing={missing} unexpected={unexpected}")
    return m.to(device).eval()


# --- eval data ---------------------------------------------------------------------------

def eval_batches(model_dir: Path = REF_MODEL, ntokens: int = 8192, seqlen: int = 512,
                 dtype=torch.bfloat16):
    """Deterministic fixed-token batches from the pinned eval corpus (no shuffling)."""
    if not EVAL_TEXT.exists():
        raise SystemExit(f"missing {EVAL_TEXT}; run m1/prep_data.py once")
    tok = load_tokenizer(model_dir)
    ids = tok(EVAL_TEXT.read_text(), add_special_tokens=False, return_tensors=None)["input_ids"]
    ids = ids[: ntokens - (ntokens % seqlen)]
    x = torch.tensor(ids, dtype=torch.long).view(-1, seqlen)
    return [x[i:i + 1] for i in range(0, len(x))]                   # batch 1: GPU is shared


def corpus_tokens(model_dir: Path = REF_MODEL, ntokens: int = 8192):
    tok = load_tokenizer(model_dir)
    ids = tok(EVAL_TEXT.read_text(), add_special_tokens=False, return_tensors=None)["input_ids"]
    return torch.tensor(ids[:ntokens], dtype=torch.long)


# --- metrics ------------------------------------------------------------------------------
#
# The GPU on this box is shared with the desktop and other agents, so verification never
# holds two models at once: logits for checkpoint A are computed in batch-1 slices and
# parked in host RAM (fp32), A is freed, then B runs, then the two are compared in chunks.

@torch.no_grad()
def forward_logits(model, batch):
    return model(input_ids=batch).logits.float()


@torch.no_grad()
def perplexity(model, batches) -> float:
    lp, n = 0.0, 0
    for b in batches:
        lg = model(input_ids=b).logits[:, :-1, :].float()
        tgt = b[:, 1:]
        lp += torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)),
                                               tgt.reshape(-1), reduction="sum").item()
        n += tgt.numel()
    return float(torch.exp(torch.tensor(lp / n)))


def cached_logits(model_dir: Path, batches, dtype=torch.float32,
                  device: str | None = None) -> torch.Tensor:
    dev = device or pick_device()
    m = load_model(model_dir, dtype=dtype, device=dev)
    out = []
    for b in batches:
        out.append(forward_logits(m, b.to(dev)).cpu())
    del m
    release(dev)
    return torch.cat(out, 0)                      # [n_batches*seqlen, vocab]


def compare_logits(la: torch.Tensor, lb: torch.Tensor, device: str | None = None,
                   chunk: int = 256) -> dict:
    """Equivalence evidence on identical tokens (fp32 forwards, chunked compare)."""
    dev = device or pick_device()
    dev = device or "cpu"
    mx, kl, agree, n = 0.0, 0.0, 0, 0
    for i in range(0, la.size(0), chunk):
        a, b = la[i:i + chunk], lb[i:i + chunk]
        mx = max(mx, (a - b).abs().max().item())
        lp, lq = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
        kl += (lp.exp() * (lp - lq)).sum().item()           # over every position in chunk
        agree += int((a.argmax(-1) == b.argmax(-1)).sum().item())
        n += a.numel() // a.size(-1)
        del a, b, lp, lq
    release(dev)
    return {"max_dlogit": mx, "kl_mean_nats": kl / max(1, n), "top1_agree": agree / max(1, n),
            "n_positions": n}


def ppl_from_logits(lg: torch.Tensor, ids: torch.Tensor, device="cpu") -> float:
    lp, n = 0.0, 0
    for i in range(lg.size(0)):
        a = lg[i, :-1].to(device).float()
        t = ids[i, 1:].to(device)
        lp += torch.nn.functional.cross_entropy(a, t, reduction="sum").item()
        n += t.numel()
    return float(torch.exp(torch.tensor(lp / n)))


def equivalence_report(dir_a: Path, dir_b: Path, ntokens=8192, seqlen=512,
                       dtype=torch.float32) -> dict:
    """bf16 artifacts in, function-equivalence evidence out. No two models resident."""
    dev = pick_device()
    x = corpus_tokens(REF_MODEL, ntokens).view(-1, seqlen)
    batches = [x[i:i + 1] for i in range(x.size(0))]
    la = cached_logits(Path(dir_a), batches, dtype, dev)
    lb = cached_logits(Path(dir_b), batches, dtype, dev)
    out = compare_logits(la, lb, dev)
    out["ppl_a"] = ppl_from_logits(la, x, dev)
    out["ppl_b"] = ppl_from_logits(lb, x, dev)
    out["ntokens"], out["seqlen"], out["device"] = ntokens, seqlen, dev
    del la, lb
    return out


# --- small helpers ------------------------------------------------------------------------

def wjson(path: Path, obj) -> Path:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    return Path(path)


def rjson(path: Path):
    return json.loads(Path(path).read_text())


def merge_sd(a: dict, b: dict, alpha: float, ties: bool = False, density: float = 0.2) -> dict:
    """Task-vector merge of candidate `a` with specialist `b`: a + alpha*(b - a).

    Tied HF artifacts omit ``lm_head.weight``; untied gauge representatives store it. Expand
    the tied representation from ``embed_tokens`` on either side before comparing keys. Any
    other key mismatch is an incompatible architecture/layout and fails closed.

    ties=True additionally magnitude-trims the single delta (S - a) to the top `density`
    fraction per tensor. HONEST LABEL: this is the TIES *trim* stage only. Elect-sign and
    sign-consensus need two independent task vectors against a shared base; here the merge
    operand is one delta (specialist minus candidate), so those stages are vacuous by
    construction. That is itself part of the finding: TIES/DARE assume the parents share a
    parameter coordinate system, which is exactly what a gauge-transformed checkpoint breaks.
    """
    a, b = dict(a), dict(b)
    for sd in (a, b):
        if "lm_head.weight" not in sd and "model.embed_tokens.weight" in sd:
            sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    if set(a) != set(b):
        raise ValueError(f"merge state key mismatch: only_a={sorted(set(a)-set(b))} "
                         f"only_b={sorted(set(b)-set(a))}")
    if not ties:
        return {k: (1 - alpha) * a[k] + alpha * b[k] for k in a}
    out = {}
    for k in a:
        d = (b[k].float() - a[k].float())
        if d.ndim >= 2 and density < 1.0:
            # torch.quantile refuses >16M elements, and embed/lm_head are 136M here: estimate the
            # trim threshold from a deterministic subsample instead of the full sort.
            ad = d.abs().flatten()
            if ad.numel() > 4_000_000:
                idx = torch.randperm(ad.numel(), generator=torch.Generator().manual_seed(0),
                                     device="cpu")[:2_000_000]
                thr = torch.quantile(ad.cpu()[idx], 1.0 - density).to(ad.device)
            else:
                thr = torch.quantile(ad, 1.0 - density)
            d = d * (ad.view_as(d) >= thr)
        out[k] = (a[k].float() + alpha * d).to(a[k].dtype)
    return out
