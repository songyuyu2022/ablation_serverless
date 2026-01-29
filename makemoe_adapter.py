# makemoe_adapter.py
import torch
import torch.nn as nn
from model_interface import MoEPartitionInterface
from makeMoE import SparseMoELanguageModel, MakeMoEConfig, Expert
import torch.nn.functional as F


class MakeMoEAdapter(MoEPartitionInterface):
    """
    适配器：将 makeMoE 的 Layer N 拆解为 Serverless 流水线。
    """

    def __init__(self, config: MakeMoEConfig, split_layer_idx: int = 1):
        self.config = config
        self.full_model = SparseMoELanguageModel(config)

        if split_layer_idx >= config.n_layer:
            split_layer_idx = config.n_layer - 1
        self.split_layer_idx = split_layer_idx

        # 组件拆分
        self.pre_stage = MakeMoEPreStage(self.full_model, split_layer_idx)
        self.post_stage = MakeMoEPostStage(self.full_model, split_layer_idx)
        self.experts = self.full_model.blocks[split_layer_idx].ffwd.experts

    def get_pre_stage(self) -> nn.Module:
        return self.pre_stage

    def get_expert_stage(self, expert_id: int) -> nn.Module:
        return self.experts[expert_id]

    def get_post_stage(self) -> nn.Module:
        return self.post_stage

    def create_expert_instance(self, expert_id: int) -> nn.Module:
        return Expert(self.config)


class MakeMoEPreStage(nn.Module):
    def __init__(self, full_model, split_idx):
        super().__init__()
        self.tok_emb = full_model.token_embedding_table
        self.pos_emb = full_model.position_embedding_table
        self.pre_blocks = full_model.blocks[:split_idx]
        target_block = full_model.blocks[split_idx]
        self.ln1 = target_block.ln1
        self.sa = target_block.sa
        self.ln2 = target_block.ln2
        self.router = target_block.ffwd.router

    def forward(self, x):
        B, T = x.shape
        x = self.tok_emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        for block in self.pre_blocks:
            x = block(x)

        # 拆分层前半部分 (Attention + Residual)
        x_attn = x + self.sa(self.ln1(x))
        h_to_expert = self.ln2(x_attn)

        # Routing
        logits, topk_vals, topk_indices = self.router(h_to_expert)
        weights = F.softmax(topk_vals, dim=-1)

        # 将数据切分发给各个专家
        # controller 期望的格式是 { "expert_id": tensor_packed }
        expert_inputs = {}
        unique_experts = torch.unique(topk_indices).cpu().numpy()
        for eid in unique_experts:
            expert_inputs[str(eid)] = h_to_expert  # 实际可只传对应的 tokens 以进一步优化

        return {
            "expert_inputs": expert_inputs,
            "context": {
                "residual": x_attn,  # 透传残差用于 PostStage 聚合
                "topk_idx": topk_indices,
                "topk_weights": weights
            },
            "router_logits": logits,
            "hidden_states": h_to_expert,  # 兼容旧版
            "expert_indices": topk_indices,
            "expert_weights": weights
        }


class MakeMoEPostStage(nn.Module):
    def __init__(self, full_model, split_idx):
        super().__init__()
        self.post_blocks = full_model.blocks[split_idx + 1:]
        self.ln_f = full_model.ln_f
        self.lm_head = full_model.lm_head

    def forward(self, expert_results_list, context, targets=None):
        # 1. 聚合专家输出 (Combine)
        # context 包含 PreStage 传来的路由信息
        res_tensor = context["residual"]
        topk_idx = context["topk_idx"]
        weights = context["topk_weights"]

        # 简单聚合：根据 indices 将 expert_results 累加回 x
        # 注意：此处简化处理，假设 expert_results_list 的顺序与 ID 一致
        combined_ffn = torch.zeros_like(res_tensor)

        # 这里对应 SparseMoeBlock 的逻辑
        for i in range(topk_idx.shape[-1]):  # top_k
            # 简化版：这里假设 expert_results_list 包含了所有专家的输出
            # 生产环境需根据 topk_idx 精确取出对应的专家输出
            for eid, e_out in enumerate(expert_results_list):
                mask = (topk_idx[:, :, i] == eid).unsqueeze(-1)
                combined_ffn += e_out * weights[:, :, i].unsqueeze(-1) * mask

        # 2. 残差连接
        x = res_tensor + combined_ffn

        # 3. 后续层处理
        for block in self.post_blocks:
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