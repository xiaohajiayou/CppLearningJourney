1. 修复 llm_engine.py 中的拼写错误
在 d:\sync\nano-vllm\nanovllm\engine\llm_engine.py 文件里，将 config_fileds 修正为 config_fields。

llm_engine.py
全部拒绝
全部采纳
from dataclasses import fields

class LLMEngine:
    def __init__(self, model, **kwargs):
        # 修正拼写错误
        config_fields = {field.name for field in fields(Config)} 
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # ... 已有代码 ...
2. 修复 block_manager.py 中 can_append 方法的逻辑问题
在 d:\sync\nano-vllm\nanovllm\engine\block_manager.py 文件中，假设实际需求是判断空闲块数量是否足够容纳 seq 新增的部分，可按如下方式修改：


block_manager.py
Apply
class BlockManager:
    def __init__(self, num_kvcache_blocks, kvcache_block_size):
        self.block_size = kvcache_block_size
        self.free_block_ids = list(range(num_kvcache_blocks))

    def can_append(self, seq: Sequence):
        # 计算 seq 新增部分需要的块数
        remaining_tokens = len(seq) % self.block_size
        if remaining_tokens == 0:
            required_blocks = 0
        else:
            required_blocks = 1
        return len(self.free_block_ids) >= required_blocks
3. 修复 sampler.py 中 forward 方法的除零问题
在 d:\sync\nano-vllm\nanovllm\layers\sampler.py 文件中，为避免除零错误，可给分母添加一个极小的常数。

sampler.py
全部拒绝
全部采纳
import torch
from torch import nn

class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.to(torch.float)
        greedy_tokens = logits.argmax(dim=-1)
        logits.div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # 添加极小常数避免除零错误
        eps = 1e-8
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + eps).argmax(dim=-1)
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
以上修复方案分别解决了拼写错误和逻辑潜在错误，你可以将这些修改应用到对应文件中进行验证。