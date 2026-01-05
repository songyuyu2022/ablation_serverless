# pre_fn.py (robust drop-in)
from __future__ import annotations

import os
import uuid
import traceback
from typing import Dict, Any, Tuple

import torch
import torch.nn.functional as F
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from shared import pack, unpack
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

# Forward cache for backward
CACHE: Dict[str, Dict[str, Any]] = {}

app = FastAPI()


# -------------------------
# Helpers
# -------------------------
def _normalize_body(body: Any) -> Tuple[str, Dict[str, Any]]:
    """
    Accept multiple shapes:
      1) {"trace_id": "...", "payload": {...}}
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
    # Try shared.unpack first; if already raw, return as-is
    try:
        obj = unpack(payload, map_location=DEVICE)
        return obj if isinstance(obj, dict) else {"_unpacked": obj}
    except Exception:
        return payload


def _to_tensor(x: Any, device: str) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    # common: list -> tensor
    t = torch.as_tensor(x)
    return t.to(device)


def _extract_pre_outputs(out: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Support multiple build_pre return conventions:
      - (h, gate_logits)
      - {"h":..., "gate_logits":...}
      - {"h":..., "logits":...}
    """
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
            return JSONResponse(
                status_code=400,
                content={"ok": False, "trace_id": trace, "error": "missing field 'x'", "recv_keys": list(obj.keys())},
            )

        x = _to_tensor(obj["x"], DEVICE)

        # Ensure [B,T] long for embedding
        if x.dtype != torch.long:
            x = x.long()

        # forward
        out = pre(x)
        h, gate_logits = _extract_pre_outputs(out)

        probs = F.softmax(gate_logits, dim=-1)
        topk_vals, topk_idx = torch.topk(probs, k=TOP_K, dim=-1)

        CACHE[trace] = {"h": h, "topk_vals": topk_vals, "topk_idx": topk_idx}

        return {
            "ok": True,
            "trace_id": trace,
            "payload": pack({"h": h, "topk_vals": topk_vals, "topk_idx": topk_idx}),
        }

    except Exception as e:
        # Return traceback so you can see real root cause in controller response body
        tb = traceback.format_exc(limit=50)
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e), "traceback": tb},
        )


@app.post("/bwd")
async def bwd(request: Request):
    try:
        body = await request.json()
        trace, payload = _normalize_body(body)

        if trace not in CACHE:
            return JSONResponse(status_code=400, content={"ok": False, "trace_id": trace, "error": "unknown trace_id"})

        obj = _safe_unpack(payload)

        if "grad_h" not in obj or "grad_topk_vals" not in obj:
            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "trace_id": trace,
                    "error": "missing grad_h or grad_topk_vals",
                    "recv_keys": list(obj.keys()) if isinstance(obj, dict) else str(type(obj)),
                },
            )

        grad_h = _to_tensor(obj["grad_h"], DEVICE)
        grad_topk_vals = _to_tensor(obj["grad_topk_vals"], DEVICE)

        h = CACHE[trace]["h"]
        topk_vals = CACHE[trace]["topk_vals"]

        torch.autograd.backward([h, topk_vals], [grad_h, grad_topk_vals])

        return {"ok": True, "trace_id": trace}

    except Exception as e:
        tb = traceback.format_exc(limit=50)
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e), "traceback": tb})


@app.post("/step")
async def step(request: Request):
    # ignore body
    opt.step()
    return {"ok": True}


@app.post("/zero")
async def zero(request: Request):
    opt.zero_grad(set_to_none=True)
    return {"ok": True}


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "vocab_size": len(stoi), "top_k": TOP_K, "num_experts": NUM_EXPERTS}
