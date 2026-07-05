# qwen2vl

模型架构

<img width="972" height="624" alt="image" src="https://github.com/user-attachments/assets/7bfa76fc-0842-4d5f-9766-16a0a4adaf81" />

依旧是3大组成部分：
- vision encoder:while the vision encoder of Qwen2-VL is initialized with the ViT derived from DFN.  the fixed position embedding in the original DFN’s ViT (Fang et al., 2023) is replaced by RoPE-2D.
- 重新训练，支持动态分辨率
- llm: qwen2 llm 系列， 使用M-ROPE
- adapter: 2层mlp

训练：3阶段训练
- stage1: 600Btoken, 训练vit和adapter
- stage2: 800B token  全参训练
- stage3: 指令微调，进训练LLM


主要改动：
- 重新训练vit,支持动态分辨率，支持图像和视频（在qwenvl中，所有图像都需要resize, 切压缩层固定长度256的token)

  具体来说：
    - For dynamic resolution, we only set min_pixels= 100 × 28 × 28 and max_pixels= 16384 × 28 × 28
    - 图片等比率放缩到像素范围区间，是28的整数倍
    - 关于28：
      - Qwen2-VL 两个硬性约束相乘：patch_size = 14：ViT 切图最小单元 14×14 像素
      - merge_size = 2：ViT 输出后 2×2 PatchMerger 压缩，每 4 个 Patch 合并成 1 个视觉 Token ，单块最终视觉 Token对应原图像素 = 14 × 2 = 28 像素，也就是说：1 个送入 LLM 的视觉 Token = 原图 28×28 像素区域。
- adapter使用了2层mlp和patcher_merger(4 个 Patch 合并成 1 个视觉 Token)
- 大语言模型中使用M-ROPE


能力：
- 多语言
- 动态分辨率
- 长视频

2d rope 和 M-rope
- 2d rope 只用在vit中
  - 实现方式：对一个表示空间位置的向量，假设为（1，3，dim)拆分为上半部分（代表h方向），和下半部分（代表w方向)（1，3，0:dim//2) (1,3,dim//2:) 分别应用1d rope,然后合并
- M rope 用在llm中
  <img width="880" height="287" alt="image" src="https://github.com/user-attachments/assets/100fff10-e6c2-4ead-8881-7f798399bd3f" />


视频处理
- ViT：帧内固定只用 2D-RoPE，无时间；
- LLM M-RoPE：T 只按帧的序列序号简单自增（第 1 帧 T=1，第 2 帧 T=2，不考虑真实 FPS 时间间隔）。

论文中的一些结论：
- 不能一味增加image_size，过大了之后，效果不一定好
- 验证scaling law ,发布2B、8B、72B等系列模型， 实验结果表明随着模型参数增加，数据增加，模型性能还在持续提升。
