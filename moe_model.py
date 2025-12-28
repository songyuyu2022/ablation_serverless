# moe_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Expert 定义 (保持不变)
# ============================================================
class ExpertMLP(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 2, act: str = "relu"):
        super().__init__()
        hidden = int(d_model) * int(hidden_mult)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

        act = act.lower()
        if act == "relu":
            self.act = nn.ReLU()
        elif act == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def build_expert(d_model: int, *, hidden_mult: int = 2, act: str = "relu") -> nn.Module:
    return ExpertMLP(d_model, hidden_mult=hidden_mult, act=act)


# ============================================================
# PreStage: 负责 Embedding 和 Gating (给 pre_fn.py 使用)
# ============================================================
class PreStage(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_experts: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor):
        """
        Input: x [B, T] (indices)
        Output: hidden [B, T, D], router_logits [B, T, E]
        """
        h = self.embed(x)
        router_logits = self.gate(h)
        return h, router_logits


# ============================================================
# PostStage: 负责 Norm, Head, Loss (给 post_fn.py 使用)
# ============================================================
class PostStage(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        """
        Input: x [B, T, D] (aggregated hidden states)
        Output: logits, loss (optional), acc1, acc5
        """
        x = self.norm(x)
        logits = self.head(x)  # [B, T, V]

        loss = None
        acc1 = 0.0
        acc5 = 0.0

        if targets is not None:
            # Flatten for loss calculation
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)

            with torch.no_grad():
                pred = logits_flat.argmax(dim=-1)
                acc1 = (pred == targets_flat).float().mean().item()

                # Check top-5 accuracy
                if logits_flat.size(-1) >= 5:
                    _, top5 = logits_flat.topk(5, dim=-1)
                    # top5: [N, 5], targets: [N]
                    correct = top5.eq(targets_flat.unsqueeze(1).expand_as(top5))
                    acc5 = correct.sum().float().item() / targets_flat.size(0)
                else:
                    acc5 = acc1  # Fallback if vocab < 5

        return logits, loss, acc1, acc5


# ============================================================
# SimpleMoE: 统合类 (给 Controller 使用，实际上只是组合了 Pre/Post)
# ============================================================
class SimpleMoE(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts

        # 使用组件构建
        self.pre_stage = PreStage(vocab_size, d_model, num_experts)

        self.experts = nn.ModuleList([
            build_expert(d_model) for _ in range(num_experts)
        ])

        self.post_stage = PostStage(d_model, vocab_size)

    # 兼容 Controller 的 forward_pre 接口
    def forward_pre(self, x: torch.Tensor):
        h, logits = self.pre_stage(x)  # [B, T, D], [B, T, E]
        probs = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)
        return h, topk_vals, topk_idx

    def forward_single_expert(self, eid: int, x: torch.Tensor):
        return self.experts[eid](x)

    # 兼容 Controller 的 forward_post 接口
    def forward_post(self, combined_output: torch.Tensor):
        # Controller 调用 forward_post 通常只想要 logits
        logits, _, _, _ = self.post_stage(combined_output, targets=None)
        return logits