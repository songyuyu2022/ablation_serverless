# expert_app.py (COVER)
from __future__ import annotations

import os
import uuid
import traceback
from typing import Dict, Any, Tuple

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from shared import pack, unpack
from dataset import build_char_vocab
from moe_model import build_expert

# ============================================================
# Env
# ============================================================
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

# vocab_size
stoi, _ = build_char_vocab(DATA_PATH, VOCAB_PATH)
CFG = {
    "vocab_size": len(stoi),
    "d_model": D_MODEL,
    "num_experts": NUM_EXPERTS,
    "top_k": TOP_K,
    "hidden_mult": HIDDEN_MULT,
    "act": ACT,
}

def _build_expert_model() -> torch.nn.Module:
    for fn in (
        lambda: build_expert(CFG, expert_id=EXPERT_ID),
        lambda: build_expert(CFG, EXPERT_ID),
        lambda: build_expert(CFG),
    ):
        try:
            m = fn()
            if isinstance(m, torch.nn.Module):
                return m
        except TypeError:
            continue
    return build_expert(CFG, EXPERT_ID)

expert = _build_expert_model().to(DEVICE)
expert.train()

opt = torch.optim.AdamW(expert.parameters(), lr=LR_EXPERT, weight_decay=WEIGHT_DECAY)

# trace_id -> cached tensors for backward
CACHE: Dict[str, Dict[str, Any]] = {}

app = FastAPI()

# ============================================================
# Helpers
# ============================================================
def _normalize(body: Any) -> Tuple[str, Dict[str, Any]]:
    """
    接受多种形状：
      1) {"trace_id": "...", "payload": {...}}   （推荐）
      2) {"payload": {...}}
      3) {"trace_id": "...", ...payload fields...}
      4) {...payload fields...}
    """
    if not isinstance(body, dict):
        trace = str(uuid.uuid4())
        return trace, {"_raw": body}

    trace = body.get("trace_id") or body.get("trace")
    if "payload" in body and isinstance(body["payload"], dict):
        payload = body["payload"]
        if not trace:
            trace = payload.get("trace_id") or payload.get("trace")
        return (trace or str(uuid.uuid4())), payload

    payload = dict(body)
    payload.pop("trace_id", None)
    payload.pop("trace", None)
    return (trace or str(uuid.uuid4())), payload


def _safe_unpack(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        obj = unpack(payload, map_location=DEVICE)
        return obj if isinstance(obj, dict) else {"_unpacked": obj}
    except Exception:
        return payload


def _to_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(DEVICE)
    return torch.as_tensor(x, device=DEVICE)


def _apply_scale_to_grads(scale: float):
    if scale == 1.0:
        return
    if not (isinstance(scale, (int, float)) and math.isfinite(float(scale))):
        return
    s = float(scale)
    for p in expert.parameters():
        if p.grad is not None:
            p.grad.mul_(s)


# ============================================================
# Routes
# ============================================================
@app.post("/fwd")
async def fwd(request: Request):
    try:
        body = await request.json()
        trace, payload = _normalize(body)
        obj = _safe_unpack(payload)

        if "inp" not in obj:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "trace_id": trace, "error": "missing field 'inp'", "recv_keys": list(obj.keys())},
            )

        inp = _to_tensor(obj["inp"])
        if inp.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            inp = inp.float()

        inp = inp.detach().requires_grad_(True)
        out = expert(inp)

        CACHE[trace] = {"inp": inp, "out": out}

        return {"ok": True, "trace_id": trace, "payload": pack({"out": out})}

    except Exception as e:
        tb = traceback.format_exc(limit=120)
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e), "traceback": tb})


@app.post("/bwd")
async def bwd(request: Request):
    try:
        body = await request.json()
        trace, payload = _normalize(body)
        obj = _safe_unpack(payload)

        if trace not in CACHE:
            return JSONResponse(status_code=400, content={"ok": False, "trace_id": trace, "error": "unknown trace_id"})

        if "grad_out" not in obj:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "trace_id": trace, "error": "missing field 'grad_out'", "recv_keys": list(obj.keys())},
            )

        grad_out = _to_tensor(obj["grad_out"])
        if grad_out.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            grad_out = grad_out.float()

        inp = CACHE[trace]["inp"]
        out = CACHE[trace]["out"]

        torch.autograd.backward(out, grad_out)
        grad_inp = inp.grad

        # ✅ 防止 cache 无限增长（安全释放）
        del CACHE[trace]

        return {"ok": True, "trace_id": trace, "payload": pack({"grad_inp": grad_inp})}

    except Exception as e:
        tb = traceback.format_exc(limit=120)
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e), "traceback": tb})


@app.post("/step")
async def step(request: Request):
    """
    支持 controller 传入 {"payload":{"scale":...}}：
    - 冷专家累计更新：controller 会传 scale=1/COLD_ACC_STEPS，避免梯度等效放大
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    try:
        trace, payload = _normalize(body)
        obj = _safe_unpack(payload)
        scale = obj.get("scale", 1.0)
        # 在 step 前缩放梯度
        try:
            s = float(scale)
            if s != 1.0:
                for p in expert.parameters():
                    if p.grad is not None:
                        p.grad.mul_(s)
        except Exception:
            pass

        opt.step()
        return {"ok": True}
    except Exception as e:
        tb = traceback.format_exc(limit=120)
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e), "traceback": tb})


@app.post("/zero")
async def zero(request: Request):
    opt.zero_grad(set_to_none=True)
    return {"ok": True}


@app.get("/health")
def health():
    return {"ok": True, "expert_id": EXPERT_ID, "device": DEVICE, "d_model": D_MODEL}
