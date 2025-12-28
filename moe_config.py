# moe_config.py
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class MoeConfig:
    """全局 MoE 配置，集中管理与专家相关的超参数。

    字段说明：
    - num_experts: 逻辑专家个数
    - top_k:       每个 token 选择的专家数量
    - d_model:     模型隐层维度
    - num_pre_layers:  pre_fn 中的 Transformer 层数
    - num_post_layers: post_fn 中的 Transformer 层数
    - vocab_size:  词表大小 [新增]
    """
    num_experts: int
    top_k: int
    d_model: int
    num_pre_layers: int
    num_post_layers: int
    vocab_size: int = 2000  # [新增] 默认词表大小
    model_name: str = "custom_moe"


# 默认 MoE 配置
DEFAULT_MOE_CONFIG = MoeConfig(
    num_experts=1,
    top_k=2,
    d_model=256,
    num_pre_layers=2,
    num_post_layers=2,
    vocab_size=2000,
)


def load_moe_config(expert_instances: Optional[Dict[str, Any]] = None) -> MoeConfig:
    """从环境变量 + experts.json 推断 MoE 配置。"""

    # 1. 基础维度配置
    d_model = int(os.getenv("EMB_DIM", str(DEFAULT_MOE_CONFIG.d_model)))

    # [新增] 读取环境变量中的 VOCAB_SIZE
    vocab_size = int(os.getenv("VOCAB_SIZE", str(DEFAULT_MOE_CONFIG.vocab_size)))

    num_pre_layers = int(
        os.getenv("N_LAYERS_PRE", str(DEFAULT_MOE_CONFIG.num_pre_layers))
    )
    num_post_layers = int(
        os.getenv("N_LAYERS_POST", os.getenv("N_LAYERS", str(DEFAULT_MOE_CONFIG.num_post_layers)))
    )

    # 2. 专家数量：优先 NUM_EXPERTS，其次 experts.json
    num_experts_env = os.getenv("NUM_EXPERTS")
    if num_experts_env is not None:
        try:
            num_experts = int(num_experts_env)
        except ValueError:
            num_experts = 1
    else:
        if expert_instances:
            num_experts = max(1, len(expert_instances))
        else:
            num_experts = DEFAULT_MOE_CONFIG.num_experts

    # 3. top-k
    top_k = int(os.getenv("TOP_K", str(DEFAULT_MOE_CONFIG.top_k)))

    return MoeConfig(
        num_experts=num_experts,
        top_k=top_k,
        d_model=d_model,
        num_pre_layers=num_pre_layers,
        num_post_layers=num_post_layers,
        vocab_size=vocab_size,
        model_name="custom_moe",
    )