# post_fn.py
from __future__ import annotations
import os, uuid
from typing import Dict, Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from shared import pack, unpack
from dataset import build_char_vocab
from moe_model import build_post

DEVICE = os.getenv("DEVICE", "cpu")
DATA_PATH = os.getenv("DATA_PATH", "input.txt")
VOCAB_PATH = os.getenv("VOCAB_PATH", "vocab.json")

D_MODEL = int(os.getenv("EMB_DIM", "256"))
LR_SHARED = float(os.getenv("LR_SHARED", "3e-4"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.0"))

stoi, _ = build_char_vocab(DATA_PATH, VOCAB_PATH)
CFG = {"vocab_size": len(stoi), "d_model": D_MODEL}

post = build_post(CFG).to(DEVICE).train()
opt = torch.optim.AdamW(post.parameters(), lr=LR_SHARED, weight_decay=WEIGHT_DECAY)

CACHE: Dict[str, Dict[str, Any]] = {}

app = FastAPI()

class Req(BaseModel):
    trace_id: str | None = None
    payload: dict

@app.post("/fwd")
def fwd(req: Req):
    trace = req.trace_id or str(uuid.uuid4())
    obj = unpack(req.payload, map_location=DEVICE)
    combined = obj["combined"].to(DEVICE)   # [B,T,D]
    y = obj["y"].to(DEVICE)                 # [B,T]
    combined.requires_grad_(True)

    logits, loss = post(combined, targets=y)
    CACHE[trace] = {"combined": combined, "loss": loss, "logits": logits, "y": y}
    return {"trace_id": trace, "payload": pack({"loss": loss.detach(), "logits": logits.detach()})}

@app.post("/bwd")
def bwd(req: Req):
    trace = req.trace_id
    assert trace in CACHE, f"unknown trace_id={trace}"
    loss = CACHE[trace]["loss"]
    combined = CACHE[trace]["combined"]

    loss.backward()
    grad_combined = combined.grad.detach()
    return {"trace_id": trace, "payload": pack({"grad_combined": grad_combined})}

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
