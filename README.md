        针对大语言模型在低显存设备上的高吞吐推理需求，将FlexLLMGen（Stanford 高吞吐量 LLM 推理引擎，支持 CPU/GPU/Disk 混合 Offloading + 4-bit 压缩）
    从仅支持OPT系列模型扩展适配支持通义千问 Qwen3系列模型。完成了pytorch后端注意力算子适配及数值对齐验证工作。针对 Prefill（一次处理完整 prompt）与 Decode
    （逐 token 自回归生成）两种推理阶段，分别适配了 Grouped Query Attention 的 KV 头扩展机制，保证KV Cache 以 n_kv_head 格式压缩存储以节省显存占用；针对
    Qwen3 无偏置归一化结构，基于 RMSNorm 重构了 Attention 层、MLP 层及 Output Embedding 的前向计算路径；以逐位置旋转矩阵乘法实现了 Rotary Position Embedding，
    适配 FlexLLMGen block-wise offloading 调度模式下的 Prefill 全序列与 Decode 单 token 两种计算场景；针对 MLP 的 SwiGLU 激活结构，完成了 gate/up/down 三矩阵的权重排布适配；通过 HuggingFace 逐层 dump 验证机制，
    以 Cosine Similarity 与 Top-K Token 重叠率为指标，完成从 Embedding 层到 Logits 层的全链路数值对齐确认。最后，FlexLLMGen 在 Qwen3-0.6B 上实现端到端文本生成，成功将模型权重和KV cache卸载到cpu及ssd上峰值显存
    仅 1.5GB，覆盖多种 Offloading 策略与序列长度的 Benchmark 验证通过。
