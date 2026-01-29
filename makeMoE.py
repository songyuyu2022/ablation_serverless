# makeMoE.py
import torch
import torch.nn as nn
from torch.nn import functional as F


class MakeMoEConfig:
    def __init__(self):
        self.batch_size = 16
        self.block_size = 32
        self.max_iters = 5000
        self.learning_rate = 1e-3
        self.eval_iters = 200
        self.n_embed = 128  # embedding dimension
        self.n_head = 4
        self.n_layer = 4
        self.dropout = 0.1
        self.num_experts = 4
        self.top_k = 2
        self.vocab_size = 65  # default, will be overwritten


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, config):
        super().__init__()
        self.key = nn.Linear(config.n_embed, head_size, bias=False)
        self.query = nn.Linear(config.n_embed, head_size, bias=False)
        self.value = nn.Linear(config.n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,16)
        q = self.query(x)  # (B,T,16)
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, 16) @ (B, 16, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, config) for _ in range(num_heads)])
        self.proj = nn.Linear(config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class Expert(nn.Module):
    """ An MLP is a simple linear layer followed by a non-linearity """

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            nn.ReLU(),
            nn.Linear(4 * config.n_embed, config.n_embed),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.top_k
        self.linear = nn.Linear(config.n_embed, config.num_experts)

    def forward(self, x):
        logits = self.linear(x)
        topk_vals, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        return logits, topk_vals, topk_indices  # Return logits for loss calculation if needed


class SparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = TopkRouter(config)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])

    def forward(self, x):
        # x: (B, T, C)
        # 这一步在 Serverless 架构中会被拆解，但在本地模型中保留逻辑
        logits, topk_vals, topk_indices = self.router(x)
        B, T, C = x.shape

        # Softmax over top-k
        topk_probs = F.softmax(topk_vals, dim=-1)

        # Weighted sum (Standard MoE logic)
        out = torch.zeros_like(x)
        # flatten for easier indexing
        x_flat = x.view(-1, C)
        topk_probs_flat = topk_probs.view(-1, self.router.top_k)
        topk_indices_flat = topk_indices.view(-1, self.router.top_k)

        # 简单循环实现 (实际部署时会被 controller 并行化替代)
        for i in range(self.router.top_k):
            idx = topk_indices_flat[:, i]
            prob = topk_probs_flat[:, i]

            # Mask for each expert
            for e_id, expert in enumerate(self.experts):
                mask = (idx == e_id)
                if mask.any():
                    inp = x_flat[mask]
                    e_out = expert(inp)
                    # Accumulate
                    # Note: scatter_add_ is cleaner but this is illustrative
                    # out_flat[mask] += e_out * prob[mask].unsqueeze(-1)
                    # For simplicity in this local block, we just assume non-parallel exec
                    pass

                    # NOTE: 在本地 Adapter 中，我们实际上不会调用 forward，而是直接访问 .router 和 .experts
        return x  # Placeholder


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, config):
        super().__init__()
        head_size = config.n_embed // config.n_head
        self.sa = MultiHeadAttention(config.n_head, head_size, config)
        self.ffwd = SparseMoeBlock(config)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        # 在适配器中，我们会在这里截断：PreStage 输出 ln2(x)，ExpertStage 处理 ffwd
        x = x + self.ffwd(self.ln2(x))
        return x


class SparseMoELanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


if __name__ == '__main__':
    # 原有的 main 测试代码
    pass