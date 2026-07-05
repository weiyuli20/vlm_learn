# qwen2.5vl

模型架构
<img width="803" height="672" alt="image" src="https://github.com/user-attachments/assets/53afccf6-afe5-44b9-9cff-0236714d6bae" />
- vision encoder : 从头训练，另外相比qwen2vl(修改了vit的结构，使用ffn with swiglu, rmsnorm)，还使用了window attention减少计算量
- llm: qwen2.5
- adapter:2层的mlp 和patchmerger(和qwen2vl一致）

模型训练：

预训练有3个阶段：
<img width="693" height="268" alt="image" src="https://github.com/user-attachments/assets/0c730579-a2cf-4b72-a40d-91bafab1c607" />

后训练：
- sft
- dpo

MRoPE改进：对齐了现实video的帧率，即对齐了绝对时间
