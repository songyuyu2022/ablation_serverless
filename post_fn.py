# post_fn.py (UPDATED - REAL SERVERLESS MODE)
from __future__ import annotations
import os, uuid, traceback
from typing import Dict, Any, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel

# ✅ 引入通信组件
from shared import pack, unpack, tensor_to_pack, pack_to_tensor
from comm import CommManager
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

# ✅ 废弃 CACHE
comm = CommManager()
# CACHE: Dict[str, Dict[str, Any]] = {}

app = FastAPI()


class Req(BaseModel):
    trace_id: Optional[str] = None
    payload: Dict[str, Any] = {}


def _get_trace_and_obj(req: Req) -> tuple[str, Dict[str, Any]]:
    trace = req.trace_id or str(req.payload.get("trace_id") or req.payload.get("trace") or uuid.uuid4())
    try:
        obj = unpack(req.payload, map_location=DEVICE)
        if not isinstance(obj, dict):
            obj = {"_payload": obj}
    except Exception:
        obj = dict(req.payload) if isinstance(req.payload, dict) else {"_raw": req.payload}
    return trace, obj


@app.post("/fwd")
def fwd(req: Req):
    try:
        trace, obj = _get_trace_and_obj(req)

        combined = obj["combined"].to(DEVICE)  # [B,T,D]
        y = obj["y"].to(DEVICE)  # [B,T]
        combined.requires_grad_(True)

        logits, loss = post(combined, targets=y)

        # [修改] 存入 Hot Store
        # Post 阶段必须保存 combined (大张量) 和 y 用于重计算
        save_key = f"{trace}_post"
        save_data = {
            "combined": tensor_to_pack(combined.detach()),
            "y": tensor_to_pack(y.detach())
        }
        comm.send_hot(save_key, save_data)

        return {"trace_id": trace, "payload": pack({"loss": loss.detach(), "logits": logits.detach()})}
    except Exception as e:
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@app.post("/bwd")
def bwd(req: Req):
    try:
        trace, obj = _get_trace_and_obj(req)

        # 1. [修改] 读取 Input
        save_key = f"{trace}_post"
        saved_data = comm.pull_hot(save_key, delete=True)

        if saved_data is None:
            return {"ok": False, "trace_id": trace, "error": f"Post trace expired: {trace}"}

        # 2. 恢复 Tensor 并重计算
        combined_data = saved_data["combined"]
        y_data = saved_data["y"]

        if isinstance(combined_data, dict):
            combined = pack_to_tensor(combined_data, map_location=DEVICE)
        else:
            combined = combined_data.to(DEVICE)

        if isinstance(y_data, dict):
            y = pack_to_tensor(y_data, map_location=DEVICE)
        else:
            y = y_data.to(DEVICE)

        combined.requires_grad_(True)

        # Re-compute (Post FWD)
        logits, loss = post(combined, targets=y)

        # 3. Backward
        # Post 是最后一站，Loss 直接 Backward，不需要上游梯度
        loss.backward()

        grad_combined = combined.grad.detach()

        return {"trace_id": trace, "payload": pack({"grad_combined": grad_combined})}
    except Exception as e:
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@app.post("/step")
def step(req: Req):
    trace, obj = _get_trace_and_obj(req)
    # 处理 scale 逻辑 (与 Expert 一致)
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
    return {"ok": True, "device": DEVICE}