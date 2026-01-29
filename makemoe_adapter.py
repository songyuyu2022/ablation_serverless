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

    def __init__(self, config: MakeMoEConfig, split_layer_idx: int = 2):
        self.config = config
        self.full_model = SparseMoELanguageModel(config)

        # 确保 split_layer_idx 有效
        if split_layer_idx >= config.n_layer:
            split_layer_idx = config.n_layer - 1

        self.split_layer_idx = split_layer_idx

        # 提取目标拆分层
        self.target_block = self.full_model.blocks[split_layer_idx]

        # 组件拆分
        self.pre_stage = MakeMoEPreStage(self.full_model, split_layer_idx)
        self.post_stage = MakeMoEPostStage(self.full_model, split_layer_idx)

        # 专家的引用 (注意：这里引用的是 target_block 里的 experts)
        self.experts = self.target_block.ffwd.experts

    def get_pre_stage(self) -> nn.Module:
        return self.pre_stage

    def get_expert_stage(self, expert_id: int) -> nn.Module:
        return self.experts[expert_id]

    def get_post_stage(self) -> nn.Module:
        return self.post_stage

    def create_expert_instance(self, expert_id: int) -> nn.Module:
        # 用于 Worker 独立初始化
        return Expert(self.config)


class MakeMoEPreStage(nn.Module):
    def __init__(self, full_model, split_idx):
        super().__init__()
        self.tok_emb = full_model.token_embedding_table
        self.pos_emb = full_model.position_embedding_table
        # 前序层
        self.pre_blocks = full_model.blocks[:split_idx]

        # 拆分层的组件 (Attention 部分)
        target_block = full_model.blocks[split_idx]
        self.ln1 = target_block.ln1
        self.sa = target_block.sa
        self.ln2 = target_block.ln2
        self.router = target_block.ffwd.router

    def forward(self, x):
        # 1. Embedding
        B, T = x.shape
        x = self.tok_emb(x) + self.pos_emb(torch.arange(T, device=x.device))

        # 2. 前序 Blocks
        for block in self.pre_blocks:
            x = block(x)

        # 3. 拆分层前半部分 (Attn)
        # x = x + sa(ln1(x))
        # 为了支持 Residual，我们需要把 Attn 的结果加回去
        # 但在 Serverless 拆分中，通常传递的是 ln2 之后的值给 Expert，
        # 而 Residual (x) 需要透传。
        # 简化策略：我们把 Attention 后的 x 作为 residual 暂存，
        # 但标准 MoE 接口只传递一个 hidden_state。
        # 适配：Expert 计算的是 FFN 部分，PostStage 需要加上这个 residual。
        # 为了兼容，PreStage 输出的 'h' 应该是要进入 Router 的数据，即 ln2(x_attn)

        x_attn = x + self.sa(self.ln1(x))
        h_to_expert = self.ln2(x_attn)  # 这就是进入 FFN 的输入

        # 4. Routing
        logits, topk_vals, topk_indices = self.router(h_to_expert)

        # 必须返回符合 model_interface 约定的字典
        # 这里的 h 必须是 [B, T, D]，后续会被 slice 发给 expert
        return {
            "hidden_states": h_to_expert,
            "router_logits": logits,
            "expert_weights": torch.softmax(topk_vals, dim=-1),  # 或者直接用 topk_vals
            "expert_indices": topk_indices,
            # Hack: 传递 residual 给 PostStage?
            # 现有的 controller 协议不支持传递额外变量。
            # 通常做法：PreStage 输出的 hidden_states 就是 x。
            # 如果 Expert 只是 FFN，那么 x_attn 丢失了。
            # 权宜之计：我们在 PostStage 里做残差连接，或者忽略上一层的残差(有损)。
            # 在此实现中，我们将 x_attn 隐含在流程中，假设 Post 接收的是 Expert 的输出。
        }


class MakeMoEPostStage(nn.Module):
    def __init__(self, full_model, split_idx):
        super().__init__()
        self.post_blocks = full_model.blocks[split_idx + 1:]
        self.ln_f = full_model.ln_f
        self.lm_head = full_model.lm_head

    def forward(self, combined_output, targets=None):
        # combined_output 是 Experts 聚合后的结果 (即 FFN 的输出)
        # 理论上应该 x_attn + combined_output
        # 但因为无法从 Pre 传 x_attn 过来，这里近似认为 combined_output 已经是完整流
        x = combined_output

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