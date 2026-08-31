#!/usr/bin/env python3
# HarvestLineage (agent HarvestLineage, /home/admin/theseus/harvest): declared-lineage population
# enumerator for Theseus base rates. Metadata-first; NEVER downloads weights unless asked with
# --fetch, which refuses to exceed its byte budget. Stdlib only (urllib/json/hashlib/shutil).
"""Build the Theseus declared-lineage population from the public HF hub.

Produces, all under harvest/cache/:

    manifest.jsonl   one JSON line per artifact (Contract manifest schema, SCHEMA.md §1)
    edges.jsonl      {"child": <id>, "parent": <id>|null, "relation": ..., "declared_in": ...}
    usage.json       cache byte accounting + df capture
    POPULATION.md    composition counts, resolvable-parent fraction, arch histogram, self-bias
    raw/<repo>.json  cached HF /api/models/<repo>?blobs=true payloads (idempotency, audit)
    probes/*.json    cached API listing / HEAD responses (idempotency, lady politeness)
    w/               downloaded weight files (--fetch only; LRU, hard budget)

Kind and lineage-source resolution
----------------------------------
kinds: base | finetune | instruct | merge | quant | adapter | mlx (Contract enum)
lineage_source / edge.declared_in: card | config | filename | none

Declared-lineage source priority (strongest first):
  1. HF `base_model:` tags   (base_model:finetune:|:quantized:|:adapter:|:merge:) -> "card"
  2. cardData.base_model / base_models (mergekit)                                 -> "card"
  3. config.peft.base_model_name_or_path (PEFT adapters)                          -> "config"
  4. raw config.json `_name_or_path` (only fetched when 1-3 are absent)            -> "config"
  5. quant filename matching an in-set base exactly                                -> "filename"

A filename-inferred relation is a HYPOTHESIS: it stays declared_in=filename and is never
upgraded to card/config. Records with no lineage are kept only when explicitly seeded
(they classify as base); otherwise dropped.

API politeness: sequential requests only; exponential backoff on 429/5xx capped at 60 s per
request; responses cached so re-runs do zero network I/O. If HF refuses, we stop and report —
a repo, size, or sha is never invented (fail closed).

Interpreter: /usr/bin/python3 (stdlib only). No torch, no huggingface_hub call path (the hub JSON
API is the data plane; huggingface_hub is not imported).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
W = CACHE / "w"
RAW = CACHE / "raw"
PROBES = CACHE / "probes"
MANIFEST = CACHE / "manifest.jsonl"
EDGES = CACHE / "edges.jsonl"
USAGE = CACHE / "usage.json"
POP = CACHE / "POPULATION.md"
REACH = CACHE / "reach.json"
LRU_JOURNAL = W / ".lru.json"

API = "https://huggingface.co"
UA = "theseus-harvest/0.1 (metadata-only lineage enumerator; Theseus/thebasedcapital)"
TIMEOUT = 40
BACKOFF_CAP_S = 60
BUDGET_BYTES = 1 << 30           # hard cache cap (1 GiB) enforced by selfcheck

KINDS = ("base", "finetune", "instruct", "merge", "quant", "adapter", "mlx")
EDGE_REL = {"finetune": "finetune_of", "quantized": "quant_of",
            "adapter": "adapter_on", "merge": "merge_of"}

# decoder-only causal-LM architecture names. A *ForCausalLM machine is decoder-only; a few
# legacy decoders are named otherwise (GPT2LMHeadModel, BloomForCausalLM covers, ...).
DECODER_CAUSAL_SUFFIX = "ForCausalLM"
DECODER_EXACT = {"GPT2LMHeadModel", "GPT2DoubleHeadsModel", "BloomForCausalLM",
                 "FalconForCausalLM", "GPTNeoForCausalLM", "GPTJForCausalLM",
                 "GPTNeoXForCausalLM", "MPTForCausalLM", "CodeGenForCausalLM",
                 "GPTBigCodeForCausalLM", "JambaForCausalLM"}
# conditional-generation / image-text machine names that must never pass the causal gate
DECODER_HARD_FAIL = ("ForConditionalGeneration", "ConditionalGeneration")
WEIGHT_PATTERNS = [
    ("model",        r"^model\.safetensors$"),
    ("model-shard",  r"^model-\d+-of-\d+\.safetensors$"),
    ("gguf",         r"^.+\.gguf$"),
    ("adapter",      r"^adapter_model\.(safetensors|bin)$"),
    ("npz",          r"^.+\.npz$"),
    ("safetensors",  r"^.+\.safetensors$"),
    ("bin",          r"^pytorch_model(-\d+-of-\d+)?\.bin$"),
    ("bin-generic",  r"^.+\.bin$"),
]

# strong markers that make a name *a quantized variant* (fp8/bf16 are merely dtypes)
KIND_QUANT_RE = re.compile(
    r"(?i)(q2_k|q3_k|q4_0|q4_1|q4_k_s|q4_k_m|q5_0|q5_1|q5_k_s|q5_k_m|q6_k|q8_0|q8_1|"
    r"-gptq|-awq|-exl2|-mlx|-gguf|bnb-4bit|\dbit)")
# full label regex (adds dtype-only tokens for the `quant` field text)
QUANT_NAME_RE = re.compile(
    r"(?i)(q2_k|q3_k|q4_0|q4_1|q4_k_s|q4_k_m|q5_0|q5_1|q5_k_s|q5_k_m|q6_k|q8_0|q8_1|"
    r"-gptq|-awq|-exl2|-mlx|-gguf|bnb-4bit|\dbit|fp8|-bf16|-f16|-i1|-i2|-i4)")
INSTRUCT_NAME_RE = re.compile(r"(?i)(instruct|chat|[-_.]it$|it\b)")
NAME_HYP_MARKERS = ("-GGUF", "-GPTQ", "-AWQ", "-ExL2", "-MLX", "-Q4_K_M", "-Q5_K_M",
                    "-Q8_0", "-4bit", "-8bit", "-bnb-4bit", "-unsloth-bnb-4bit",
                    "-fp8", "-BF16")
FAMILY_TAG_HINTS = ("gguf", "gptq", "awq", "bitsandbytes", "mlx", "exl2", "fp8")
QUANT_FAMILY_TAGS = ("gguf", "gptq", "awq", "bitsandbytes", "mlx", "exl2")  # kind-level signals

# Seed repos: explicit starter set. kind is NOT forced — each is classified from its OWN
# declared metadata. Guarantees architecture diversity and gives derived clusters a root.
SEEDS = [
    "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2-7B", "Qwen/Qwen2-0.5B", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen2.5-Coder-7B",
    "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-2-7b-hf", "NousResearch/Meta-Llama-3.1-8B",
    "mistralai/Mistral-7B-v0.3", "mistralai/Mistral-7B-v0.1", "mistralai/Mixtral-8x7B-v0.1",
    "google/gemma-2-2b", "google/gemma-2-9b", "google/gemma-2-27b", "google/gemma-3-4b-pt",
    "microsoft/phi-4", "microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3.5-mini-instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "allenai/OLMo-2-7B", "ibm-granite/granite-3.0-8b-base", "ibm-granite/granite-7b-base",
    "HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-1.7B",
    "tiiuae/falcon-7b", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "teknium/OpenHermes-2.5-Mistral-7B", "NousResearch/Hermes-3-Llama-3.1-8B",
    "HuggingFaceH4/zephyr-7b-beta",
]
CLUSTER_K = {s: 6 for s in SEEDS}
for s in SEEDS[22:]:
    CLUSTER_K[s] = 3
SWEEPS = [
    ("merge", 45), ("gguf", 40), ("gptq", 12), ("awq", 10),
    ("peft", 22), ("lora", 15), ("mlx", 12),
]


# ================================================================ httplib (polite, cached)
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None  # stop at the 302 so we can read x-linked-* headers


def _probe_key(url: str) -> Path:
    return PROBES / (hashlib.sha1(url.encode()).hexdigest() + ".json")


def _http(method: str, url: str, cached: bool = True, follow: bool = True):
    """GET/HEAD with backoff (429/5xx, cap 60 s). Returns (status, body, headers).

    follow=False stops at redirects (LFS 302) so x-linked-size / x-linked-etag are readable.
    Cached responses short-circuit identical URLs — re-runs are offline.
    """
    if cached:
        pk = _probe_key(url)
        if pk.exists():
            o = json.loads(pk.read_text())
            return o["status"], o.get("body"), o.get("headers") or {}
    opener = urllib.request.build_opener() if follow else urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, method=method, headers={"User-Agent": UA})
    wait, t0 = 1.0, time.time()
    while True:
        try:
            r = opener.open(req, timeout=TIMEOUT)
            body = r.read()
            keep = {k: v for k, v in ((h.lower(), r.headers.get(h)) for h in
                                      ("x-linked-size", "x-linked-etag", "x-repo-commit"))
                    if v}
            if cached:
                _cache_probe(pk, r.status, body, keep)
            return r.status, body, keep
        except urllib.error.HTTPError as e:
            if e.code in (302, 301) and not follow:  # redirect = public LFS pointer
                keep = {k: e.headers.get(k) for k in
                        ("x-linked-size", "x-linked-etag", "x-repo-commit")}
                keep = {k: v for k, v in keep.items() if v}
                if cached:
                    _cache_probe(pk, e.code, None, keep)
                return e.code, None, keep
            if e.code in (429, 500, 502, 503, 504) and time.time() - t0 < BACKOFF_CAP_S:
                time.sleep(wait + random.random() * 0.5)
                wait = min(wait * 2, 32)
                continue
            if cached:
                _cache_probe(pk, e.code, None, {})
            return e.code, None, {}
        except urllib.error.URLError:
            if time.time() - t0 < BACKOFF_CAP_S:
                time.sleep(wait + random.random() * 0.5)
                wait = min(wait * 2, 32)
                continue
            raise


def _cache_probe(pk: Path, status: int, body, hdrs: dict) -> None:
    pk.write_text(json.dumps({"status": status,
                              "body": body.decode("utf-8", "replace") if body else None,
                              "headers": hdrs}))


def api_listing(filter_tag: str, position: int, limit: int):
    url = (f"{API}/api/models?limit={limit}&position={position}&sort=downloads"
           f"&direction=-1&filter={urllib.parse.quote(filter_tag)}")
    st, body, _ = _http("GET", url)
    if st != 200:
        raise RuntimeError(f"HF listing refused ({st}) for filter={filter_tag!r} — stopping")
    return json.loads(body)


def fetch_blobs(repo: str):
    """GET /api/models/<repo>?blobs=true (cached). None if gone/not public (fail closed)."""
    path = RAW / (repo.replace("/", "__") + ".json")
    if path.exists():
        return json.loads(path.read_text())
    url = f"{API}/api/models/{urllib.parse.quote(repo, safe='/')}?blobs=true"
    st, body, _ = _http("GET", url)
    if st != 200:
        path.write_text(json.dumps({"error": st, "repo": repo}))
        return None
    meta = json.loads(body)
    path.write_text(json.dumps(meta))
    return meta


def head_weight(repo: str, rev: str, fname: str):
    """HEAD one weight file, stop at the LFS 302. Returns (status, x_size, x_etag). Cached."""
    url = f"{API}/{repo}/resolve/{urllib.parse.quote(rev or 'main')}/{urllib.parse.quote(fname)}"
    st, _, hdrs = _http("HEAD", url, follow=False)
    sz = hdrs.get("x-linked-size")
    etag = (hdrs.get("x-linked-etag") or "").strip('"')
    return st, (int(sz) if sz else None), (etag or None)


# ================================================================ classification
def _base_relations(meta: dict):
    """Declared lineage from base_model: tags; relation-prefixed entries win."""
    rel = []
    for t in meta.get("tags") or []:
        if not t.startswith("base_model:"):
            continue
        rest = t[len("base_model:"):]
        if ":" in rest:
            r, repo = rest.split(":", 1)
            if r in EDGE_REL:
                rel.append((EDGE_REL[r], repo))
        else:
            rel.append(("finetune_of", rest))
    seen, out = set(), []
    for k in rel:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _card_bases(meta: dict):
    cd = meta.get("cardData") or {}
    bm = cd.get("base_model")
    if bm is None:
        return None
    if isinstance(bm, str):
        if "," not in bm:
            return [("finetune_of", bm.strip())]
        return [("merge_of", x.strip()) for x in bm.split(",") if x.strip()]
    if isinstance(bm, list):
        return [("merge_of", x) for x in bm if isinstance(x, str) and x]
    return None


def _config_bases(meta: dict):
    cfg = meta.get("config") or {}
    peft = cfg.get("peft") or {}
    b = peft.get("base_model_name_or_path") or peft.get("base_model")
    return [("adapter_on", b)] if isinstance(b, str) and b else None


def _first_arch(meta: dict):
    cfg = meta.get("config") or {}
    archs = cfg.get("architectures") or []
    if archs and isinstance(archs[0], str) and archs[0]:
        return archs[0]
    mt = cfg.get("model_type")
    return mt if isinstance(mt, str) else None


def _decoder_ok(arch) -> bool:
    if not arch:
        return True  # unknown arch is permitted (adapters; recorded as null)
    if any(h in arch for h in DECODER_HARD_FAIL):
        return False  # conditional-generation / vision machines are not causal decoders
    return arch.endswith(DECODER_CAUSAL_SUFFIX) or arch in DECODER_EXACT


def _is_instruct(repo: str) -> bool:
    return INSTRUCT_NAME_RE.search(repo.split("/")[-1]) is not None


def _qnorm(name: str, tags):
    """Normalized quant label ('GGUF-Q4_K_M', 'GPTQ-Int4', 'MLX-4bit', 'BNB-4bit')."""
    tl = {t.lower() for t in (tags or [])}
    tag = next((t for t in ("gguf", "gptq", "awq", "mlx", "exl2", "fp8", "bitsandbytes")
                if t in tl), None)
    m = QUANT_NAME_RE.search(name)
    if tag == "bitsandbytes":
        tag = "bnb"
    if not tag and not m:
        return None
    piece = (m.group(0).strip("-") if m else "")
    lab = (tag or "").upper()
    if lab and piece and piece.lower() == lab.lower():  # tag and name both say GGUF — no doubling
        piece = ""
    return f"{lab}-{piece}" if lab and piece else (lab or piece.upper())

def _name_hyp_short(short: str):
    for marker in NAME_HYP_MARKERS:
        if short.endswith(marker):
            b = short[: -len(marker)].rstrip("-_ .")
            return b if len(b) > 1 else None
    return None


def _weight_files(meta: dict, kind: str):
    """Pick weight files by pattern priority. Returns ([(name, size, sha),(...)], pattern)."""
    sibs = meta.get("siblings") or []
    tagl = {t.lower() for t in meta.get("tags") or []}
    if kind == "adapter":
        order = ["adapter"]
    elif kind == "mlx":
        order = ["npz", "safetensors"]
    elif "gguf" in tagl or any(s.get("rfilename", "").endswith(".gguf") for s in sibs):
        order = ["gguf"]
    else:
        order = ["model", "model-shard", "adapter", "npz", "safetensors", "bin", "bin-generic"]
    for pname, rx_s in WEIGHT_PATTERNS:
        if pname not in order:
            continue
        rx = re.compile(rx_s)
        sel = [s for s in sibs if rx.fullmatch(s.get("rfilename", ""))]
        if sel:
            sel.sort(key=lambda s: s.get("size") or 0, reverse=True)
            return [(s["rfilename"], s.get("size"),
                     (s.get("lfs") or {}).get("sha256")) for s in sel], pname
    return [], None


def classify(meta: dict, is_seed: bool = False, is_parent: bool = False):
    """Return (kind, [(relation, parent_repo)], lineage_source, quant) or (None,None,None,None,drop)."""
    repo = meta.get("id") or ""
    short = repo.split("/")[-1]
    tagl = {t.lower() for t in meta.get("tags") or []}
    tags_raw = meta.get("tags") or []
    sibs = meta.get("siblings") or []
    cfg = meta.get("config") or {}

    rel = _base_relations(meta)
    card = _card_bases(meta)
    cfg_b = _config_bases(meta)

    gguf_files = any(s.get("rfilename", "").endswith(".gguf") for s in sibs)
    has_adapter_files = any(s.get("rfilename", "").startswith("adapter_") for s in sibs)
    is_peft = "peft" in tagl or has_adapter_files or (cfg.get("peft") is not None)
    mlx_marked = "mlx" in tagl or repo.startswith("mlx-community/")
    q_tag = next((t for t in QUANT_FAMILY_TAGS if t in tagl), None)
    name_quant = KIND_QUANT_RE.search(short) is not None
    qname = _qnorm(short, tags_raw)

    kind = bases = source = quant = None

    # ------ declared lineages, strongest first ------
    if rel:
        source = "card"
        qt = [x for x in rel if x[0] == "quant_of"]
        at = [x for x in rel if x[0] == "adapter_on"]
        mt = [x for x in rel if x[0] == "merge_of"]
        ft = [x for x in rel if x[0] == "finetune_of"]
        if at:
            kind, bases = "adapter", at
        elif qt:
            kind = "mlx" if (mlx_marked or re.search(r"(?i)(\dbit|mlx)", short)) else "quant"
            bases = qt
        elif mt:
            kind, bases = "merge", mt
        elif ft:
            kind = "mlx" if (mlx_marked and re.search(r"(?i)(\dbit)", short)) else \
                ("instruct" if _is_instruct(repo) else "finetune")
            bases = ft
        else:
            kind, bases = "finetune", []
    elif card:
        source = "card"
        if len(card) >= 2 or any(x[0] == "merge_of" for x in card):
            kind, bases = "merge", card[:4]
        else:
            kind = "instruct" if _is_instruct(repo) else "finetune"
            bases = card
    elif cfg_b:
        source = "config"
        kind, bases = "adapter", cfg_b
    else:
        # ------ no declared lineage: only keep things whose kind a NAME/TAG alone justifies -
        if is_peft:
            kind = "adapter"
        elif q_tag or name_quant or gguf_files:
            kind = "mlx" if mlx_marked else "quant"
        elif "merge" in tagl or re.search(r"(?i)-merge(?:d)?$", short) or re.search(r"(?i)\+.*\+", short):
            kind = "merge"
        elif is_seed or is_parent:
            kind = "base"
        else:
            return None, None, None, None, "no declared lineage and no kind marker"

    quant = qname if kind in ("quant", "mlx") else None
    arch = _first_arch(meta)
    if kind != "adapter" and not _decoder_ok(arch):
        return None, None, None, None, f"non-decoder arch {arch!r}"
    return kind, (bases or []), (source or "none"), quant, None


# ================================================================ population build
class Population:
    def __init__(self, parent_inject: bool = True):
        self.records: dict = {}   # repo -> {meta, kind, bases, source, quant}
        self.drops = []           # (repo, reason)
        self.parent_inject = parent_inject

    def add(self, repo: str, is_seed: bool = False, is_parent: bool = False, why: str = ""):
        if repo in self.records:
            return
        meta = fetch_blobs(repo)
        if meta is None:
            self.drops.append((repo, why or "not found / not public"))
            return
        if meta.get("id") != repo:
            self.drops.append((repo, f"id mismatch {meta.get('id')!r}"))
            return
        kind, bases, source, quant, drop = classify(meta, is_seed, is_parent)
        if kind is None:
            self.drops.append((repo, drop or why or "unclassifiable"))
            return
        self.records[repo] = {"meta": meta, "kind": kind, "bases": bases,
                              "source": source, "quant": quant}

    def inject_parents(self):
        if not self.parent_inject:
            return
        for r in list(self.records.values()):
            for _, parent in r["bases"]:
                if parent not in self.records:
                    self.add(parent, is_seed=parent in SEEDS, is_parent=True, why="declared parent")


def _record_body(r: dict):
    meta, kind = r["meta"], r["kind"]
    arch = _first_arch(meta)
    saf = meta.get("safetensors") or {}
    params = saf.get("total") if isinstance(saf, dict) else None
    dtypes = saf.get("parameters") if isinstance(saf, dict) else None
    dtype = max(dtypes, key=dtypes.get) if isinstance(dtypes, dict) and dtypes else None

    files_n, _pat = _weight_files(meta, kind)
    files = {f: (sz or 0) for f, sz, _ in files_n}
    prim = files_n[0][2] if files_n else None
    sha16 = (prim or "")[:16]

    return {
        "repo": meta.get("id"),
        "revision": meta.get("sha"),
        "kind": kind,
        "declared_base": [p for _, p in r["bases"]] if r["bases"] else None,
        "lineage_source": r["source"],
        "arch": arch,
        "params": params,
        "dtype": dtype,
        "quant": r["quant"],
        "files": files,
        "weights_present": bool(files),
        "sha256_16": sha16,
        "downloads": meta.get("downloads"),
        "created": meta.get("createdAt"),
    }


def _master_id(body: dict) -> str:
    canon = json.dumps({k: body[k] for k in sorted(body)}, sort_keys=True,
                       separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:12]


def run_list(args):
    P = Population(parent_inject=not args.no_parent_inject)

    # 1. seeds
    for repo in SEEDS:
        P.add(repo, is_seed=True, why="seed")
    # 2. derived clusters (top downloads, filter=base_model:<seed>)
    for repo in SEEDS:
        k = CLUSTER_K[repo]
        page = api_listing(f"base_model:{repo}", 0, k)
        for m in page:
            P.add(m["id"], why=f"cluster-of {repo}")
    # 3. global kind sweeps
    for ftag, limit in SWEEPS:
        page = api_listing(ftag, 0, limit)
        for m in page:
            P.add(m["id"], why=f"sweep:{ftag}")
    # 4. parent injection -> resolvable matched pairs
    P.inject_parents()

    # ---- filename-hypothesis resolution: quant/mlx repo whose short name exactly matches an
    # in-set base's short name + quant marker => hypothesis edge, declared_in=filename
    short_to_repos = {}
    for repo in sorted(P.records):
        short_to_repos.setdefault(repo.split("/")[-1], []).append(repo)
    for repo in sorted(P.records):
        r = P.records[repo]
        if r["kind"] in ("quant", "mlx") and not r["bases"]:
            hyp = _name_hyp_short(repo.split("/")[-1])
            if hyp:
                targets = [x for x in short_to_repos.get(hyp, []) if x not in (repo,)
                           and _first_arch(P.records[x]["meta"]) is not None]
                if targets:
                    t = targets[0]
                    r["bases"] = [("quant_of", t)]
                    r["source"] = "filename"  # HYPOTHESIS — never upgraded to card/config

    # ---- deterministic records + ids ----
    repos = sorted(P.records)
    lines, idtab = [], {}
    for repo in repos:
        body = _record_body(P.records[repo])
        rid = _master_id(body)
        body = {"id": rid, **body}
        idtab[repo] = rid
        lines.append(body)

    # ---- edges (resolved + dangling) ----
    edges, dangling = [], []
    for repo in repos:
        r = P.records[repo]
        for rel, parent in r["bases"]:
            if parent in idtab:
                edges.append({"child": idtab[repo], "parent": idtab[parent],
                              "relation": rel, "declared_in": r["source"]})
            else:
                dangling.append({"child": idtab[repo], "parent": None, "parent_repo": parent,
                                 "relation": rel, "declared_in": r["source"],
                                 "reason": "declared parent has no manifest record"})

    MANIFEST.write_text("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in lines))
    EDGES.write_text("".join(json.dumps(x, separators=(",", ":")) + "\n"
                             for x in edges + dangling))

    build_reach(lines)
    record_usage()
    write_population(lines, edges, dangling, P.drops)

    n_par = sum(1 for r in P.records.values() if r["bases"])
    res = sum(1 for e in edges if e["parent"])
    print(f"manifest {len(lines)} records | bases-referencing {n_par} | resolved edges {len(edges)}"
          f" ({100 * res / max(n_par, 1):.0f}% of referencers) | dangling {len(dangling)} | "
          f"dropped {len(P.drops)}")
    return 0


# ================================================================ reachability / usage
def build_reach(lines):
    out = {}
    for x in lines:
        repo, r = x["repo"], x
        if not r["files"]:
            out[repo] = {"files_authorized": False, "reason": "no weight files declared"}
            continue
        meta = fetch_blobs(repo)
        if meta is None:
            out[repo] = {"files_authorized": False, "reason": "metadata missing"}
            continue
        if meta.get("private") or meta.get("gated"):
            out[repo] = {"files_authorized": False,
                         "reason": "gated" if meta.get("gated") else "private"}
            continue
        fname = next(iter(r["files"]))
        st, sz, etag = head_weight(repo, r["revision"] or "main", fname)
        ok = st in (200, 302, 301, 206)
        out[repo] = {"files_authorized": ok, "reason": None if ok else f"HTTP {st}",
                     "head_file": fname}
    REACH.write_text(json.dumps(out, indent=1, sort_keys=True))


def record_usage():
    lines = load_manifest()
    cache_bytes = sum(f.stat().st_size for f in CACHE.rglob("*") if f.is_file())
    w_bytes = sum(f.stat().st_size for f in W.rglob("*")
                  if f.is_file() and f.name != ".lru.json")
    df = shutil.disk_usage(str(HERE.parent))
    prior = json.loads(USAGE.read_text()) if USAGE.exists() else {}
    usage = {"schema": 1, "cache_bytes": cache_bytes, "w_bytes": w_bytes,
             "manifest_records": len(lines), "budget_bytes": BUDGET_BYTES,
             "df": {"avail_bytes": df.free, "used_bytes": df.used, "total_bytes": df.total},
             "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if prior.get("last_fetch"):  # keep df-before/after evidence from the last fetch
        usage["last_fetch"] = prior["last_fetch"]
    USAGE.write_text(json.dumps(usage, indent=1, sort_keys=True) + "\n")


# ================================================================ POPULATION.md
def write_population(lines, edges, dangling, drops):
    kinds = Counter(x["kind"] for x in lines)
    archs = Counter((x["arch"] or "null") for x in lines)
    rec_ids = {x["id"] for x in lines}
    all_edges = edges + dangling
    referencers = {e["child"] for e in all_edges}                       # children w/ declared parent
    res_children = {e["child"] for e in all_edges if e["parent"] and e["parent"] in rec_ids}
    resolved_edges = sum(1 for e in all_edges if e["parent"] and e["parent"] in rec_ids)
    frac = (len(res_children) / len(referencers) * 100) if referencers else 0.0
    dang = Counter(d["relation"] for d in dangling)
    # resolved quant-chains base->instruct->quant present
    chains = 0

    L = []
    w = L.append
    w("# Theseus harvest population — GENERATED, do not edit by hand")
    w("")
    w("owner: HarvestLineage (harvest/lineage.py); describes harvest/cache/manifest.jsonl "
      "at regeneration time.")
    w(f"regenerated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    w(f"records: {len(lines)} | resolved edges: {resolved_edges} | resolved-parent children: "
      f"{len(res_children)} | dangling edges: {len(dangling)} | dropped: {len(drops)}")
    w("")
    w("## 1. Composition by kind\n")
    w("| kind | count |")
    w("|---|---|")
    for k in KINDS:
        w(f"| {k} | {kinds.get(k, 0)} |")
    w("")
    w("## 2. Resolvable parent (what enables matched pairs)\n")
    w(f"- records declaring a parent: {len(referencers)} "
      f"({100 * len(referencers) / len(lines):.1f}% of the population)")
    w(f"- of those, at least one declared parent is inside the population: {len(res_children)} "
      f"({frac:.1f}% of parent-declaring records)")
    w(f"- resolved edge lines: {resolved_edges}; dangling edge lines: {len(dangling)}")
    w("- a parent is resolvable when both the child and the declared parent have manifest "
      "records; that is the substrate for base->instruct->quant matched pairs without further "
      "discovery.")
    if dang:
        w("\nDangling (declared parent absent from the population), by relation:")
        for rel, n in dang.most_common():
            w(f"- {rel}: {n}")
    w("")
    w("## 3. Top architectures\n")
    w("| arch | count |")
    w("|---|---|")
    for a, n in archs.most_common(14):
        w(f"| {a} | {n} |")
    w("")
    w("## 4. Selection bias of this sampling (self-declared; analysis is the next slice)\n")
    w("1. **`sort=downloads` over-represents popularity.** Every probe page is sorted by "
      "download count, so the population skews to famous English instruction/chat models derived "
      "from a handful of mega-bases (Qwen2.5, Llama-3.1). Long-tail finetunes, non-English "
      "models, and small bases appear only via their cluster page. Any base rate computed here is "
      "a rate over *popular* hub artifacts, not over all checkpoints.")
    w("2. **Quant pool is concentrated on the same popular bases.** GGUF/GPTQ/AWQ publishers "
      "re-release the same ~20 popular instruct models under many quant tags; downloads rank "
      "those first. `quant`-kind rows therefore describe a few base distributions repeatedly — "
      "great for matched chains, weak for a general quant-vs-base claim.")
    w("3. **Resolvable-parent fraction is deliberately inflated.** Declared parents are injected "
      "into the manifest regardless of popularity, so ~90%+ of declared parents resolve. That is "
      "a design guarantee for the matched-pairs use case, NOT an independent estimate of how "
      "often HF publishers declare lineage.")
    w("4. **instruct vs finetune is a name heuristic.** HF tags both as `base_model:finetune:`. "
      "We label `instruct` only when the repo name matches instruct|chat|it\\b; e.g. "
      "'OpenHermes' or 'Zephyr' (instruction-tuned) land in `finetune`. Splits on that boundary "
      "carry classification error.")
    w("5. **Survivorship.** The population is what the hub returns today. Deleted, renamed, or "
      "private parents become dangling edges; historic artifacts that were removed are simply "
      "absent.")
    w("6. **Kind/arch filtering.** Non-decoder architectures (vision, embedding, video, "
      "speech) are dropped; a no-lineage repo is admitted only when a name/tag marker justifies "
      "its kind and it passes the decoder gate. The population is therefore scoped to what "
      "Theseus's static scanner can actually read.")
    w("")
    w("## 5. Sample drops (top reasons)\n")
    def drop_label(r):
        if r and r.startswith("non-decoder"):
            return "non-decoder arch"
        if r and r.startswith("id mismatch"):
            return "id mismatch"
        if r and r.startswith("not found"):
            return "not found / not public"
        return "no declared lineage / kind marker"
    dropr = Counter(drop_label(d[1]) for d in drops)
    w("| reason | count |")
    w("|---|---|")
    for r, n in dropr.most_common(12):
        w(f"| {r} | {n} |")
    POP.write_text("\n".join(L) + "\n")


# ================================================================ load helpers
def load_manifest():
    if not MANIFEST.exists():
        return []
    return [json.loads(ln) for ln in MANIFEST.read_text().splitlines() if ln.strip()]


def load_edges():
    if not EDGES.exists():
        return []
    return [json.loads(ln) for ln in EDGES.read_text().splitlines() if ln.strip()]


# ================================================================ fetch (LRU + hard budget)
def cmd_fetch(ids, budget=0.8, vacate=False):
    _args = argparse.Namespace(ids=ids, budget_gb=budget, vacate=vacate)
    args = _args
    budget = int(args.budget_gb * (1 << 30))
    records = {x["id"]: x for x in load_manifest()}
    lru = json.loads(LRU_JOURNAL.read_text()) if LRU_JOURNAL.exists() else {}

    if args.vacate:
        n = 0
        for f in W.rglob("*"):
            if f.is_file() and f.name != ".lru.json":
                f.unlink()
                n += 1
        LRU_JOURNAL.write_text("{}")
        record_usage()
        print(f"vacated {n} weight file(s)")
        return 0

    missing = [i for i in args.ids if i not in records]
    if missing:
        print(f"REFUSE: unknown manifest id(s): {missing}", file=sys.stderr)
        return 1
    if not args.ids:
        print("REFUSE: --fetch requires at least one manifest id", file=sys.stderr)
        return 1

    wanted = []
    for rid in args.ids:
        for fname, sz in records[rid]["files"].items():
            wanted.append((rid, fname, sz))
    if not wanted:
        print(f"REFUSE: no weight files declared for {args.ids}", file=sys.stderr)
        return 1

    now = {}
    for f in W.rglob("*"):
        if f.is_file() and f.name != ".lru.json":
            now[str(f.relative_to(W))] = f.stat().st_size
    cur = sum(now.values())
    incoming = sum(w[2] for w in wanted)
    total = cur + incoming

    df_before = shutil.disk_usage(str(HERE.parent))
    # LRU evict oldest by journal touch time while over budget
    order = sorted((ts, fn) for fn, ts in lru.items())
    evicted = []
    while total > budget and order:
        _ts, fn = order.pop(0)
        p = W / fn
        if p.exists():
            total -= p.stat().st_size
            p.unlink()
            evicted.append(fn)
            now.pop(fn, None)
        lru.pop(fn, None)
    if total > budget:
        over = total - budget
        print(f"REFUSE: {incoming} bytes requested, {args.budget_gb} GiB budget, "
              f"{cur} current; even after evicting {len(evicted)} LRU file(s), "
              f"{over} bytes would exceed the cap — increase --budget-gb or free space.",
              file=sys.stderr)
        LRU_JOURNAL.write_text(json.dumps(lru))
        return 2

    # budget fits: download (sequential, sized chunks)
    dl = 0
    for rid, fname, _sz in wanted:
        r = records[rid]
        destdir = W / rid
        dest = destdir / fname
        if dest.exists():
            lru[f"{rid}/{fname}"] = time.time()
            continue
        destdir.mkdir(parents=True, exist_ok=True)
        url = (f"{API}/{r['repo']}/resolve/{urllib.parse.quote(r['revision'] or 'main')}"
               f"/{urllib.parse.quote(fname)}")
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=5 * 60) as resp, dest.open("wb") as fh:
            shutil.copyfileobj(resp, fh, 1 << 16)
        expect = _blob_sha(r["repo"], fname)
        if expect:
            h = hashlib.sha256()
            with dest.open("rb") as fh:
                for c in iter(lambda: fh.read(1 << 20), b""):
                    h.update(c)
            if h.hexdigest() != expect:
                dest.unlink()
                print(f"REFUSE: sha256 mismatch on {rid}/{fname}; file deleted",
                      file=sys.stderr)
                return 3
        lru[f"{rid}/{fname}"] = time.time()
        dl += 1

    LRU_JOURNAL.write_text(json.dumps(lru))
    df_after = shutil.disk_usage(str(HERE.parent))
    usage = json.loads(USAGE.read_text()) if USAGE.exists() else {}
    usage["last_fetch"] = {"downloaded": dl, "incoming_bytes": incoming, "evicted": evicted,
                           "w_bytes": total, "df_before_avail_bytes": df_before.free,
                           "df_after_avail_bytes": df_after.free,
                           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    USAGE.write_text(json.dumps(usage, indent=1, sort_keys=True) + "\n")
    print(f"fetched {dl} weight file(s) ({incoming} bytes); cache/w now {total} bytes "
          f"(budget {budget}); evicted {len(evicted)}")
    return 0


def _blob_sha(repo: str, fname: str):
    p = RAW / (repo.replace("/", "__") + ".json")
    if not p.exists():
        return None
    for s in json.loads(p.read_text()).get("siblings") or []:
        if s.get("rfilename") == fname:
            return (s.get("lfs") or {}).get("sha256")
    return None


# ================================================================ selfcheck
def selfcheck():
    errs = []
    lines = load_manifest()
    edges = load_edges()

    if not lines:
        errs.append("manifest is empty or missing")
    seen, by_id = set(), {}
    for i, x in enumerate(lines):
        if not isinstance(x, dict):
            errs.append(f"manifest line {i}: not a JSON object"); continue
        if x.get("kind") not in KINDS:
            errs.append(f"manifest line {i} {x.get('id')}: kind not in enum: {x.get('kind')!r}")
        if x.get("id") in seen:
            errs.append(f"duplicate manifest id {x.get('id')}")
        seen.add(x.get("id"))
        by_id[x.get("id")] = x
        for f in ("repo", "revision", "declared_base", "lineage_source", "arch", "params",
                  "dtype", "quant", "files", "weights_present", "sha256_16", "downloads",
                  "created"):
            if f not in x:
                errs.append(f"manifest line {i} {x.get('id')}: missing {f}")
        if x.get("lineage_source") not in ("card", "config", "filename", "none"):
            errs.append(f"manifest line {i}: bad lineage_source {x.get('lineage_source')!r}")

    for i, e in enumerate(edges):
        if not isinstance(e, dict):
            errs.append(f"edge line {i}: not a JSON object"); continue
        c = e.get("child")
        if c not in by_id:
            errs.append(f"edge {i}: child {c!r} not in manifest")
        if e.get("relation") not in set(EDGE_REL.values()):
            errs.append(f"edge {i}: bad relation {e.get('relation')!r}")
        if e.get("declared_in") not in ("card", "config", "filename", "none"):
            errs.append(f"edge {i}: bad declared_in {e.get('declared_in')!r}")
        p = e.get("parent")
        if p is None:
            if not e.get("reason"):
                errs.append(f"edge {i}: dangling parent without reason")
        elif p not in by_id:
            errs.append(f"edge {i}: parent {p!r} not in manifest")

    cache_bytes = sum(f.stat().st_size for f in CACHE.rglob("*") if f.is_file())
    if cache_bytes > (1 << 30):
        errs.append(f"cache {cache_bytes} bytes exceeds 1 GiB")
    if USAGE.exists():
        u = json.loads(USAGE.read_text())
        if u.get("cache_bytes", 0) > (1 << 30):
            errs.append("usage.json reports cache over 1 GiB")

    if errs:
        print(f"selfcheck: {len(errs)} problem(s)")
        for e in errs:
            print("  FAIL " + e)
        return 1
    print(f"selfcheck OK: {len(by_id)} unique manifest ids, {len(edges)} edges, "
          f"cache {cache_bytes / 1e6:.1f} MB")
    return 0


# ================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="Theseus harvest lineage enumerator")
    ap.add_argument("--selfcheck", action="store_true", help="validate cache contents")
    ap.add_argument("--fetch", action="store_true",
                    help="download weights (also: bare `fetch` subcommand)")
    ap.add_argument("--budget-gb", type=float, default=0.8,
                    help="hard cap for cache/w in GiB (fetch refuses to exceed it)")
    ap.add_argument("--vacate", action="store_true", help="remove all downloaded weights")
    ap.add_argument("--no-parent-inject", action="store_true",
                    help="do not add declared parents to the population")
    ap.add_argument("ids", nargs="*", help="manifest ids (fetch target; `list`/`fetch` markers "
                                           "are accepted and ignored)")
    args = ap.parse_args(argv)

    if args.selfcheck:
        return selfcheck()
    ids = [i for i in args.ids if i not in ("list", "fetch")]  # tolerate markers
    if args.fetch or args.vacate or ids:
        return cmd_fetch(ids, args.budget_gb, args.vacate)
    return run_list(args)


if __name__ == "__main__":
    sys.exit(main())
