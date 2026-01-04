# expert_app.py
from __future__ import annotations
import os, uuid
from typing import Dict, Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from shared import pack, unpack
from dataset import build_char_vocab
from moe_model import build_expert

DEVICE = os.getenv("DEVICE", "cpu")
DATA_PATH = os.getenv("DATA_PATH", "input.txt")
VOCAB_PATH = os.getenv("VOCAB_PATH", "vocab.json")

D_MODEL = int(os.getenv("EMB_DIM", "256"))
NUM_EXPERTS = int(os.getenv("NUM_EXPERTS", "4"))
TOP_K = int(os.getenv("TOP_K", "2"))
HIDDEN_MULT = int(os.getenv("HIDDEN_MULT", "2"))
ACT = os.getenv("ACT", "relu")

LR_EXPERT = float(os.getenv("LR_EXPERT", "3e-4"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.0"))

EXPERT_ID = int(os.getenv("EXPERT_ID", "0"))

stoi, _ = build_char_vocab(DATA_PATH, VOCAB_PATH)
CFG = {"vocab_size": len(stoi), "d_model": D_MODEL, "num_experts": NUM_EXPERTS, "top_k": TOP_K, "hidden_mult": HIDDEN_MULT, "act": ACT}

expert = build_expert(CFG, expert_id=EXPERT_ID).to(DEVICE).train()
opt = torch.optim.AdamW(expert.parameters(), lr=LR_EXPERT, weight_decay=WEIGHT_DECAY)

CACHE: Dict[str, Dict[str, Any]] = {}

app = FastAPI()

class Req(BaseModel):
    trace_id: str | None = None
    payload: dict

@app.post("/fwd")
def fwd(req: Req):
    trace = req.trace_id or str(uuid.uuid4())
    obj = unpack(req.payload, map_location=DEVICE)
    inp = obj["inp"].to(DEVICE)  # [N,D]
    inp.requires_grad_(True)
    out = expert(inp)            # [N,D]
    CACHE[trace] = {"inp": inp, "out": out}
    return {"trace_id": trace, "payload": pack({"out": out})}

@app.post("/bwd")
def bwd(req: Req):
    trace = req.trace_id
    assert trace in CACHE, f"unknown trace_id={trace}"
    obj = unpack(req.payload, map_location=DEVICE)
    grad_out = obj["grad_out"].to(DEVICE)  # [N,D]
    out = CACHE[trace]["out"]
    inp = CACHE[trace]["inp"]

    out.backward(grad_out)  # accum expert grads, inp.grad available
    grad_inp = inp.grad.detach()
    return {"trace_id": trace, "payload": pack({"grad_inp": grad_inp})}

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
    return {"ok": True, "expert_id": EXPERT_ID, "device": DEVICE}
