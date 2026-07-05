# 梳理qwenvl-qwen2vl-qwen2.5vl-qwen3vl 系列进化
## Qwen2VL
模型架构
<img width="1288" height="544" alt="image" src="https://github.com/user-attachments/assets/4234f84e-7ec9-494e-87aa-c6e8bde4657c" />
共包含3大模块：
- vision encoder： 使用的是vit (权重来自于 Openclip’s ViT-bigG)
- llm : Qwen-7B
- adapter: 将图片特征映射到llm编码空间， 使用的是 a single-layer cross-attention module ，将图像特征压缩到固定长度256. 做法是设置一组可以学习的query，图片特征作为key和value。该attention中使用2D绝对位置编码。

训练方式：3阶段训练
- 1，2 是预训练阶段,
  stage1 冻结llm  使用1.4B图文数据对（原始有5B，清洗后得到1.4B，中英文：3:1）
  stage2 全部参数多任务
- 3 是指令微调阶段，得到了Qwen2VL-Chat。 该阶段冻结vision encoder

图像分辩率和图像特征长度
- stage1 224 * 224  256
- stage2 448 * 448  256

固定分辨率：图像需要resize

图像特征长度指的是图像输入到llm的token个数，由可学习的query决定


attention机制：报告中指出使用global attentioin 而非window attention, 因为计算速度差距不大但是window attention 收敛速度明显变慢

qwen2vl能力：
- 多语言
- 多图输入
- 细粒度理解

报告中还指出因为在训练阶段引入了纯文本预料，消融实验表明纯文本能力没有退化，相反还稍有提高。


cross attention 详解：
```
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualCrossAttention(nn.Module):
    """
    Qwen-VL 单层独立Cross-Attention视觉压缩层
    Q: 可学习固定256 Query Token
    K/V: ViT输出图像Patch特征
    """
    def __init__(
        self,
        vis_hidden_size: int = 1024,   # ViT输出维度
        llm_hidden_size: int = 4096,   # Qwen-7B LLM维度
        num_heads: int = 16,
        num_queries: int = 256,        # 固定压缩到256视觉token
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = llm_hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5

        # 可学习固定Query (唯一可训练查询向量，全局共享)
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, llm_hidden_size))

        # 投影层：视觉特征映射到LLM维度
        self.k_proj = nn.Linear(vis_hidden_size, llm_hidden_size)
        self.v_proj = nn.Linear(vis_hidden_size, llm_hidden_size)
        self.q_proj = nn.Linear(llm_hidden_size, llm_hidden_size)
        self.out_proj = nn.Linear(llm_hidden_size, llm_hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(llm_hidden_size)
        self.norm_kv = nn.LayerNorm(vis_hidden_size)

    def forward(self, visual_patch_features: torch.Tensor):
        """
        Args:
            visual_patch_features: [B, N_patch, vis_hidden_size] ViT输出图像patch
        Returns:
            compressed_vis_tokens: [B, 256, llm_hidden_size] 压缩后视觉token
        """
        B, N_patch, _ = visual_patch_features.shape

        # Step1: 归一化 + 投影K/V（图像侧）
        kv_norm = self.norm_kv(visual_patch_features)
        k = self.k_proj(kv_norm)  # [B, N_patch, D_llm]
        v = self.v_proj(kv_norm)  # [B, N_patch, D_llm]

        # Step2: 扩展可学习Query到batch维度
        q = self.query_tokens.expand(B, -1, -1)  # [B, 256, D_llm]
        q_norm = self.norm_q(q)
        q = self.q_proj(q_norm)

        # Step3: 多头拆分 (B, seq_len, head, head_dim)
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 256, d]
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N_patch, d]
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Step4: Scaled Dot Product Cross-Attention 核心计算
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v)  # [B, H, 256, d]
        attn_out = attn_out.transpose(1, 2).contiguous()  # [B, 256, H, d]
        attn_out = attn_out.view(B, -1, self.num_heads * self.head_dim)

        # Step5: 输出投影
        compressed_vis_tokens = self.out_proj(attn_out)
        return compressed_vis_tokens

```
```
class QwenVLVisualAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        # ViT视觉编码器（原版OpenCLIP ViT-L/14）
        self.vit = nn.Identity()  # 此处省略ViT完整实现，仅占位
        # 单层Cross-Attention压缩层（你提问的核心模块）
        self.cross_attn = VisualCrossAttention(
            vis_hidden_size=1024,
            llm_hidden_size=4096,
            num_heads=16,
            num_queries=256
        )

    def forward(self, pixel_values):
        # 1. ViT提取图像Patch特征
        patch_feats = self.vit(pixel_values)  # [B, N_patch, 1024]
        # 2. 单层Cross-Attention压缩到固定256 token
        vis_tokens = self.cross_attn(patch_feats)  # [B, 256, 4096]
        return vis_tokens
```

2D绝对位置编码伪代码
```
# 模拟ViT输出
visual_feat = torch.randn(B, N_patch, D)  # [2, 1024, 1024]

# 模拟2D网格位置编码
grid_pos = torch.randn(H, W, D)  # [32, 32, 1024]
pos_emb = grid_pos.reshape(-1, D)  # [1024, 1024]

# 扩展batch维度
pos_emb = pos_emb.unsqueeze(0)  # [1, 1024, 1024]

# 广播相加
visual_feat_with_pos = visual_feat + pos_emb
```



