# pre_fn.py
from __future__ import annotations
import os, uuid
from typing import Dict, Any

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel

from shared import pack, unpack
from dataset import build_char_vocab
from moe_model import build_pre

DEVICE = os.getenv("DEVICE", "cpu")
DATA_PATH = os.getenv("DATA_PATH", "input.txt")
VOCAB_PATH = os.getenv("VOCAB_PATH", "vocab.json")

D_MODEL = int(os.getenv("EMB_DIM", "256"))
NUM_EXPERTS = int(os.getenv("NUM_EXPERTS", "4"))
TOP_K = int(os.getenv("TOP_K", "2"))
LR_SHARED = float(os.getenv("LR_SHARED", "3e-4"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.0"))

# build vocab_size from vocab.json/input
stoi, _ = build_char_vocab(DATA_PATH, VOCAB_PATH)
CFG = {"vocab_size": len(stoi), "d_model": D_MODEL, "num_experts": NUM_EXPERTS, "top_k": TOP_K}

pre = build_pre(CFG).to(DEVICE).train()
opt = torch.optim.AdamW(pre.parameters(), lr=LR_SHARED, weight_decay=WEIGHT_DECAY)

CACHE: Dict[str, Dict[str, Any]] = {}

app = FastAPI()

class Req(BaseModel):
    trace_id: str | None = None
    payload: dict

@app.post("/fwd")
def fwd(req: Req):
    trace = req.trace_id or str(uuid.uuid4())
    obj = unpack(req.payload, map_location=DEVICE)
    x = obj["x"]  # [B,T] long
    x = x.to(DEVICE)

    x.requires_grad_(False)
    h, gate_logits = pre(x)
    probs = F.softmax(gate_logits, dim=-1)
    topk_vals, topk_idx = torch.topk(probs, k=TOP_K, dim=-1)  # [B,T,K]

    # keep graph for backward
    CACHE[trace] = {
        "h": h,
        "topk_vals": topk_vals,
        "topk_idx": topk_idx,
    }
    return {"trace_id": trace, "payload": pack({"h": h, "topk_vals": topk_vals, "topk_idx": topk_idx})}

@app.post("/bwd")
def bwd(req: Req):
    trace = req.trace_id
    assert trace in CACHE, f"unknown trace_id={trace}"
    obj = unpack(req.payload, map_location=DEVICE)
    grad_h = obj["grad_h"].to(DEVICE)                 # [B,T,D]
    grad_topk_vals = obj["grad_topk_vals"].to(DEVICE) # [B,T,K]

    h = CACHE[trace]["h"]
    topk_vals = CACHE[trace]["topk_vals"]

    # backprop to pre params (embed+gate)
    torch.autograd.backward([h, topk_vals], [grad_h, grad_topk_vals])
    return {"ok": True}

@app.post("/step")
def step(req: Req):
    opt.step()
    return {"ok": True}

@app.post("/zero")
def zero(req: Req):
    opt.zero_grad(set_to_none=True)
    return {"ok": True}

@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "vocab_size": len(stoi)}
