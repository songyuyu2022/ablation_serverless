# expert_app.py (UPDATED - REAL SERVERLESS MODE)
from __future__ import annotations

import os
import uuid
import traceback
from typing import Dict, Any, Tuple

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ✅ 引入与 controller 一致的通信和序列化工具
from shared import pack, unpack, tensor_to_pack, pack_to_tensor
from comm import CommManager
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

# [修改点 1] 初始化通信管理器，废弃 CACHE 内存字典
comm = CommManager()
# CACHE: Dict[str, Dict[str, Any]] = {}  <-- 删除这个作弊的字典

app = FastAPI()


# ============================================================
# Helpers
# ============================================================
def _normalize(body: Any) -> Tuple[str, Dict[str, Any]]:
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
            return JSONResponse(status_code=400, content={"ok": False, "trace_id": trace, "error": "missing 'inp'"})

        inp = _to_tensor(obj["inp"])
        if inp.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            inp = inp.float()

        # 1. Forward 计算
        inp = inp.detach().requires_grad_(True)
        out = expert(inp)

        # [修改点 2] 真实存储：将 Input 存入模拟的 Hot Store (Redis)
        # 使用 trace_id + expert_id 作为唯一键，这与 LocalExecutor 逻辑一致
        save_key = f"{trace}_exp_{EXPERT_ID}"
        # 注意：这里我们只存 inp，模拟“无状态”，不存 out 对象
        save_data = {"inp": tensor_to_pack(inp.detach())}

        # 存入 Hot 通道（模拟 Redis）
        comm.send_hot(save_key, save_data)

        # 返回结果
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

        if "grad_out" not in obj:
            return JSONResponse(status_code=400,
                                content={"ok": False, "trace_id": trace, "error": "missing 'grad_out'"})

        grad_out = _to_tensor(obj["grad_out"])
        if grad_out.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            grad_out = grad_out.float()

        # [修改点 3] 真实读取与重计算
        save_key = f"{trace}_exp_{EXPERT_ID}"

        # 从 Hot Store 拉取 Input (delete=True 模拟读后即焚，节省空间)
        saved_data = comm.pull_hot(save_key, delete=True)

        if saved_data is None:
            # 容错：如果 Hot 里没有，可能被驱逐或者存错了，尝试 Cold (虽然 FWD 强制存 Hot)
            saved_data = comm.pull_cold(save_key, delete=True)

        if saved_data is None:
            return JSONResponse(status_code=400,
                                content={"ok": False, "trace_id": trace, "error": "Activation expired/not found"})

        # A. 恢复 Input 并开启梯度
        inp_data = saved_data["inp"]
        # 注意：这里需要 unpack 出来
        if isinstance(inp_data, dict):  # 如果是 pack 格式
            inp = pack_to_tensor(inp_data).to(DEVICE)
        else:
            inp = _to_tensor(inp_data)

        inp.requires_grad_(True)

        # B. 重计算 Forward (Re-computation)
        # 这是 Serverless BWD 的核心开销所在
        out = expert(inp)

        # C. 执行 Backward
        torch.autograd.backward(out, grad_out)
        grad_inp = inp.grad

        return {"ok": True, "trace_id": trace, "payload": pack({"grad_inp": grad_inp})}

    except Exception as e:
        tb = traceback.format_exc(limit=120)
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e), "traceback": tb})


@app.post("/step")
async def step(request: Request):
    # Step 逻辑基本不变，但去掉了复杂的 unpack 检查，使其更健壮
    try:
        body = await request.json()
    except Exception:
        body = {}

    try:
        trace, payload = _normalize(body)
        obj = _safe_unpack(payload)
        scale = obj.get("scale", 1.0)

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