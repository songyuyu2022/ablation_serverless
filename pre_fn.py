# pre_fn.py (UPDATED - REAL SERVERLESS MODE)
from __future__ import annotations

import os
import uuid
import traceback
from typing import Dict, Any, Tuple

import torch
import torch.nn.functional as F
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ✅ 引入通信组件
from shared import pack, unpack, tensor_to_pack, pack_to_tensor
from comm import CommManager
from dataset import build_char_vocab
from moe_model import build_pre

# -------------------------
# Env / Config
# -------------------------
DEVICE = os.getenv("DEVICE", "cpu")
DATA_PATH = os.getenv("DATA_PATH", "input.txt")
VOCAB_PATH = os.getenv("VOCAB_PATH", "vocab.json")

D_MODEL = int(os.getenv("EMB_DIM", "256"))
NUM_EXPERTS = int(os.getenv("NUM_EXPERTS", "4"))
TOP_K = int(os.getenv("TOP_K", "2"))
LR_SHARED = float(os.getenv("LR_SHARED", "3e-4"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.0"))

stoi, _ = build_char_vocab(DATA_PATH, VOCAB_PATH)
CFG = {
    "vocab_size": len(stoi),
    "d_model": D_MODEL,
    "num_experts": NUM_EXPERTS,
    "top_k": TOP_K,
}

pre = build_pre(CFG).to(DEVICE)
pre.train()

opt = torch.optim.AdamW(pre.parameters(), lr=LR_SHARED, weight_decay=WEIGHT_DECAY)

# ✅ 初始化通信管理器，废弃 CACHE
comm = CommManager()
# CACHE: Dict[str, Dict[str, Any]] = {}

app = FastAPI()


# -------------------------
# Helpers
# -------------------------
def _normalize_body(body: Any) -> Tuple[str, Dict[str, Any]]:
    if not isinstance(body, dict):
        trace = str(uuid.uuid4())
        return trace, {"_raw": body}
    trace = body.get("trace_id") or body.get("trace")
    if "payload" in body and isinstance(body["payload"], dict):
        payload = body["payload"]
        if trace is None:
            trace = payload.get("trace_id") or payload.get("trace")
        trace = trace or str(uuid.uuid4())
        return trace, payload
    payload = dict(body)
    payload.pop("trace_id", None)
    payload.pop("trace", None)
    trace = trace or str(uuid.uuid4())
    return trace, payload


def _safe_unpack(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        obj = unpack(payload, map_location=DEVICE)
        return obj if isinstance(obj, dict) else {"_unpacked": obj}
    except Exception:
        return payload


def _to_tensor(x: Any, device: str) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    t = torch.as_tensor(x)
    return t.to(device)


def _extract_pre_outputs(out: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        return out[0], out[1]
    if isinstance(out, dict):
        h = out.get("h")
        g = out.get("gate_logits", out.get("logits", out.get("gate", None)))
        if h is None or g is None:
            raise ValueError(f"build_pre returned dict but missing keys, got keys={list(out.keys())}")
        return h, g
    raise ValueError(f"Unsupported build_pre output type: {type(out)}")


# -------------------------
# Routes
# -------------------------
@app.post("/fwd")
async def fwd(request: Request):
    try:
        body = await request.json()
        trace, payload = _normalize_body(body)
        obj = _safe_unpack(payload)

        if "x" not in obj:
            return JSONResponse(status_code=400, content={"ok": False, "trace_id": trace, "error": "missing 'x'"})

        x = _to_tensor(obj["x"], DEVICE)
        if x.dtype != torch.long: x = x.long()

        # 1. Forward
        out = pre(x)
        h, gate_logits = _extract_pre_outputs(out)

        probs = F.softmax(gate_logits, dim=-1)
        topk_vals, topk_idx = torch.topk(probs, k=TOP_K, dim=-1)

        # 2. [修改] 存入 Hot Store (Redis)
        # Pre 阶段只需存 Input x 即可重计算
        save_key = f"{trace}_pre"
        save_data = {"x": tensor_to_pack(x.detach())}
        comm.send_hot(save_key, save_data)

        return {
            "ok": True,
            "trace_id": trace,
            "payload": pack({"h": h, "topk_vals": topk_vals, "topk_idx": topk_idx}),
        }

    except Exception as e:
        tb = traceback.format_exc(limit=50)
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e), "traceback": tb})


@app.post("/bwd")
async def bwd(request: Request):
    try:
        body = await request.json()
        trace, payload = _normalize_body(body)

        # 1. [修改] 从 Hot Store 读取
        save_key = f"{trace}_pre"
        saved_data = comm.pull_hot(save_key, delete=True)

        if saved_data is None:
            return JSONResponse(status_code=400, content={"ok": False, "trace_id": trace, "error": "Pre trace expired"})

        # 2. 恢复 Input 并重计算
        x_data = saved_data["x"]
        if isinstance(x_data, dict):
            x = pack_to_tensor(x_data, map_location=DEVICE)
        else:
            x = _to_tensor(x_data, DEVICE)

        if x.dtype != torch.long: x = x.long()

        # Re-compute Forward (构建计算图)
        out = pre(x)
        h, gate_logits = _extract_pre_outputs(out)

        # 为了求导 gate_logits，我们需要重演 softmax 和 topk 吗？
        # 通常 grad_h 和 grad_topk_vals 是从 controller 传回来的
        # 如果我们只对 h 和 gate_logits 求导，直接 backward 即可
        # 注意：controller 传回的是 grad_topk_vals，这意味着我们需要连 topk 操作也重算一遍
        # 才能把梯度传回 gate_logits

        probs = F.softmax(gate_logits, dim=-1)
        topk_vals, topk_idx = torch.topk(probs, k=TOP_K, dim=-1)

        # 3. 接收梯度
        obj = _safe_unpack(payload)
        if "grad_h" not in obj:
            return JSONResponse(status_code=400, content={"ok": False, "error": "missing grad_h"})

        grad_h = _to_tensor(obj["grad_h"], DEVICE)

        # 简单的 backward (假设 grad_topk_vals 是可选的，或者你主要关注 embedding 梯度)
        # 如果 controller 实现了对 gate 的梯度回传：
        tensors_to_grad = [h]
        grads = [grad_h]

        if "grad_topk_vals" in obj:
            grad_topk = _to_tensor(obj["grad_topk_vals"], DEVICE)
            tensors_to_grad.append(topk_vals)
            grads.append(grad_topk)

        torch.autograd.backward(tensors_to_grad, grads)

        return {"ok": True, "trace_id": trace}

    except Exception as e:
        tb = traceback.format_exc(limit=50)
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e), "traceback": tb})


@app.post("/step")
async def step(request: Request):
    opt.step()
    return {"ok": True}


@app.post("/zero")
async def zero(request: Request):
    opt.zero_grad(set_to_none=True)
    return {"ok": True}


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE}