# pre_fn.py
import os
import torch
import torch.nn as nn
from fastapi import FastAPI, Request, Response

from shared import dumps, loads, tensor_to_pack, pack_to_tensor
from moe_config import load_moe_config
from utils.logger import log

# --- 核心修改：引入 moe_model ---
from moe_model import PreStage

app = FastAPI()

DEVICE = os.getenv("DEVICE", "cpu")
# 如果有 GPU 可用且没被禁用
if torch.cuda.is_available() and os.getenv("CUDA_VISIBLE_DEVICES", "") != "":
    DEVICE = "cuda"


def init_pre_model():
    moe_cfg = load_moe_config()
    log("pre-fn", f"Initializing PreStage (Real MoE). Vocab={moe_cfg.vocab_size}, Dim={moe_cfg.d_model}")

    # 直接初始化 PreStage
    model = PreStage(
        vocab_size=moe_cfg.vocab_size,
        d_model=moe_cfg.d_model,
        num_experts=moe_cfg.num_experts
    ).to(DEVICE)

    # 简单优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


pre_model, optimizer = init_pre_model()


@app.post("/fwd")
async def pre_forward(req: Request):
    """
    输入: {"x": tensor[B, T]}
    输出: {"hidden": tensor[B, T, D], "gate": tensor[B, T, E]}
    """
    try:
        body = await req.body()
        obj = loads(body)

        x_pack = obj.get("x")
        if x_pack is None:
            return Response(content="Missing 'x'", status_code=400)

        x = pack_to_tensor(x_pack).to(DEVICE).long()

        # --- 真实前向计算 ---
        h, router_logits = pre_model(x)

        # 将结果返回给 Controller (或者下一级)
        # Controller 那边会自己做 Softmax/TopK，这里返回 logits 即可
        # 或者为了保持一致性，也可以返回 softmax 后的 probs，这里直接返回 logits 更灵活

        resp_data = {
            "hidden": tensor_to_pack(h.cpu()),
            "gate": tensor_to_pack(router_logits.cpu())  # 注意：这里是 logits
        }
        return Response(content=dumps(resp_data), media_type="application/msgpack")

    except Exception as e:
        log("pre-fn", f"FWD Error: {e}")
        return Response(content=f"Internal Error: {e}", status_code=500)


@app.post("/bwd")
async def pre_backward(req: Request):
    try:
        body = await req.body()
        obj = loads(body)

        # 接收梯度（在真实训练中，这里会接收 dL/dh 并进行 backward）
        # 目前版本仅做占位或简单打印
        # if "grads" in obj:
        #    grads = pack_to_tensor(obj["grads"]).to(DEVICE)
        #    ... backward logic ...

        return Response(content=dumps({"ok": True}), media_type="application/msgpack")

    except Exception as e:
        log("pre-fn", f"BWD Error: {e}")
        return Response(content=f"Internal Error: {e}", status_code=500)