# post_fn.py
import os
import torch
import torch.nn as nn
from fastapi import FastAPI, Request, Response

from shared import dumps, loads, tensor_to_pack, pack_to_tensor
from utils.logger import log
from moe_config import load_moe_config

# --- 核心修改：引入 moe_model ---
from moe_model import PostStage

app = FastAPI()

DEVICE = os.getenv("DEVICE", "cpu")
if torch.cuda.is_available() and os.getenv("CUDA_VISIBLE_DEVICES", "") != "":
    DEVICE = "cuda"


def init_post_model():
    moe_cfg = load_moe_config()
    log("post-fn", f"Initializing PostStage (Real MoE). Dim={moe_cfg.d_model}")

    # 直接初始化 PostStage
    model = PostStage(
        d_model=moe_cfg.d_model,
        vocab_size=moe_cfg.vocab_size
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


post_model, optimizer = init_post_model()


@app.post("/fwd")
async def post_forward(req: Request):
    """
    输入: {"y": hidden_states, "targets": optional labels, "train": bool}
    输出: {"loss": float, "acc_top1": float...}
    """
    try:
        body = await req.body()
        obj = loads(body)

        y_pack = obj.get("y")  # Aggregated hidden states
        if y_pack is None:
            return Response(content="Missing 'y'", status_code=400)

        y = pack_to_tensor(y_pack).to(DEVICE)

        targets = None
        if "targets" in obj:
            targets = pack_to_tensor(obj["targets"]).to(DEVICE).long()

        train_mode = obj.get("train", False)
        if train_mode:
            post_model.train()
        else:
            post_model.eval()

        # --- 真实前向计算 ---
        # 如果是训练模式，我们需要 targets 来计算 loss
        if train_mode and targets is None:
            # 如果没有 targets 但在训练，只能返回 logits，无法计算 loss
            logits, loss, acc1, acc5 = post_model(y, None)
        else:
            logits, loss, acc1, acc5 = post_model(y, targets)

        resp = {
            "loss": loss.item() if loss is not None else 0.0,
            "acc_top1": acc1,
            "acc_top5": acc5
        }

        # --- 真实反向传播 (简单版) ---
        if train_mode and loss is not None:
            optimizer.zero_grad()
            loss.backward()

            # 回传梯度给上一级 (Controller -> Experts)
            # 我们需要返回对输入 y 的梯度 dL/dy
            if y.grad is not None:
                resp["grads"] = tensor_to_pack(y.grad.cpu())
            else:
                # 某些层可能没有保留梯度，返回全0防止报错
                resp["grads"] = tensor_to_pack(torch.zeros_like(y).cpu())

            optimizer.step()

        return Response(content=dumps(resp), media_type="application/msgpack")

    except Exception as e:
        log("post-fn", f"FWD Error: {e}")
        return Response(content=f"Internal Error: {e}", status_code=500)


@app.post("/bwd")
async def post_backward(req: Request):
    # PostStage 通常是最后一环，fwd 里已经包含了 loss backward
    # 除非有更后级的处理，否则这里可能不需要逻辑
    return Response(content=dumps({"ok": True}), media_type="application/msgpack")