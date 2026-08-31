#!/usr/bin/env python3
"""Pin the M1 eval corpus: deterministic slice of WikiText-2 (raw) test split.

Committed so every later run is offline and byte-identical.
"""
from __future__ import annotations

import json
from pathlib import Path

import common

OUT = common.DATA / "eval_wikitext.txt"
PROV = common.DATA / "PROVENANCE.json"
MAX_CHARS = 400_000


def main():
    from datasets import load_dataset
    common.DATA.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    lines, total = [], 0
    for r in ds:
        t = r["text"].strip()
        if len(t) < 80 or t.startswith("= "):      # skip blanks / section stubs
            continue
        lines.append(t)
        total += len(t)
        if total >= MAX_CHARS:
            break
    text = "\n\n".join(lines) + "\n"
    OUT.write_text(text)
    tok = common.load_tokenizer()
    ntok = len(tok(text, add_special_tokens=False)["input_ids"])
    PROV.write_text(json.dumps({
        "source": "Salesforce/wikitext wikitext-2-raw-v1 split=test (HF datasets)",
        "license": "CC BY-SA 4.0",
        "selection": "first paragraphs with >=80 chars, section stubs dropped, until 400k chars",
        "chars": len(text), "paragraphs": len(lines), "qwen_tokens": ntok,
    }, indent=2) + "\n")
    common.log(f"wrote {OUT} chars={len(text)} qwen_tokens={ntok}")


if __name__ == "__main__":
    main()
