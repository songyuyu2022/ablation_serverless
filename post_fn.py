# post_fn.py (ROBUST COVER)
from __future__ import annotations
import os, uuid
from typing import Dict, Any, Optional

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
    # ✅ 允许 body 为空/只带 trace_id，避免 422
    trace_id: Optional[str] = None
    payload: Dict[str, Any] = {}

def _get_trace_and_obj(req: Req) -> tuple[str, Dict[str, Any]]:
    """
    兼容 controller 的两种写法：
    - trace_id 在外层（推荐）
    - trace_id 也可能被放在 payload 里（兜底）
    同时对 payload 执行 unpack(map_location=DEVICE)
    """
    trace = req.trace_id or str(req.payload.get("trace_id") or req.payload.get("trace") or uuid.uuid4())
    try:
        obj = unpack(req.payload, map_location=DEVICE)
        if not isinstance(obj, dict):
            obj = {"_payload": obj}
    except Exception:
        # 如果 payload 不是 pack 格式，就原样用
        obj = dict(req.payload) if isinstance(req.payload, dict) else {"_raw": req.payload}
    return trace, obj

@app.post("/fwd")
def fwd(req: Req):
    trace, obj = _get_trace_and_obj(req)

    combined = obj["combined"].to(DEVICE)   # [B,T,D]
    y = obj["y"].to(DEVICE)                 # [B,T]
    combined.requires_grad_(True)

    logits, loss = post(combined, targets=y)
    CACHE[trace] = {"combined": combined, "loss": loss, "logits": logits, "y": y}

    # 返回给 controller：loss/logits 用 detach（不把图传回去）
    return {"trace_id": trace, "payload": pack({"loss": loss.detach(), "logits": logits.detach()})}

@app.post("/bwd")
def bwd(req: Req):
    trace, obj = _get_trace_and_obj(req)

    if trace not in CACHE:
        # ✅ 不要 assert 直接崩 500，给出可读错误
        return {"ok": False, "trace_id": trace, "error": f"unknown trace_id={trace}"}

    loss = CACHE[trace]["loss"]
    combined = CACHE[trace]["combined"]

    loss.backward()
    grad_combined = combined.grad.detach()

    # ✅ 清理缓存，避免内存增长
    del CACHE[trace]

    return {"trace_id": trace, "payload": pack({"grad_combined": grad_combined})}

@app.post("/step")
def step(req: Req):
    # ✅ 支持可选的 scale（不影响你现在逻辑，但后面冷累计很有用）
    trace, obj = _get_trace_and_obj(req)
    scale = obj.get("scale", 1.0)
    try:
        s = float(scale)
        if s != 1.0:
            for p in post.parameters():
                if p.grad is not None:
                    p.grad.mul_(s)
    except Exception:
        pass

    opt.step()
    return {"ok": True}

@app.post("/zero")
def zero(req: Req):
    opt.zero_grad(set_to_none=True)
    return {"ok": True}

@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "vocab_size": len(stoi)}
