# moe_model.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertMLP(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 2, act: str = "relu"):
        super().__init__()
        h = int(d_model) * int(hidden_mult)
        self.fc1 = nn.Linear(d_model, h)
        self.fc2 = nn.Linear(h, d_model)
        act = (act or "relu").lower()
        self.act = nn.ReLU() if act == "relu" else nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class PreStage(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_experts: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x_tok: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.embed(x_tok)        # [B,T,D]
        gate_logits = self.gate(h)   # [B,T,E]
        return h, gate_logits


class PostStage(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, h: torch.Tensor, targets: Optional[torch.Tensor] = None):
        z = self.ln(h)
        logits = self.lm_head(z)  # [B,T,V]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


@dataclass
class SimpleMoEConfig:
    vocab_size: int
    d_model: int
    num_experts: int
    top_k: int
    hidden_mult: int = 2
    act: str = "relu"


# ---- builder shims for serverless apps ----
def _get(d: Dict[str, Any], *keys: str, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def build_pre(cfg: Union[SimpleMoEConfig, Dict[str, Any]]) -> nn.Module:
    if isinstance(cfg, SimpleMoEConfig):
        vocab_size, d_model, num_experts = cfg.vocab_size, cfg.d_model, cfg.num_experts
    else:
        vocab_size = int(_get(cfg, "vocab_size", default=256))
        d_model = int(_get(cfg, "d_model", "emb_dim", default=256))
        num_experts = int(_get(cfg, "num_experts", default=4))
    return PreStage(vocab_size=vocab_size, d_model=d_model, num_experts=num_experts)


def build_post(cfg: Union[SimpleMoEConfig, Dict[str, Any]]) -> nn.Module:
    if isinstance(cfg, SimpleMoEConfig):
        vocab_size, d_model = cfg.vocab_size, cfg.d_model
    else:
        vocab_size = int(_get(cfg, "vocab_size", default=256))
        d_model = int(_get(cfg, "d_model", "emb_dim", default=256))
    return PostStage(d_model=d_model, vocab_size=vocab_size)


def build_expert(cfg: Union[SimpleMoEConfig, Dict[str, Any]], expert_id: int = 0) -> nn.Module:
    if isinstance(cfg, SimpleMoEConfig):
        d_model, hidden_mult, act = cfg.d_model, cfg.hidden_mult, cfg.act
    else:
        d_model = int(_get(cfg, "d_model", "emb_dim", default=256))
        hidden_mult = int(_get(cfg, "hidden_mult", default=2))
        act = str(_get(cfg, "act", default="relu"))
    return ExpertMLP(d_model=d_model, hidden_mult=hidden_mult, act=act)
