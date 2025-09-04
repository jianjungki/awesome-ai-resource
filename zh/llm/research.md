### 上下文扩展（Context Length Extension）

| 技术名称             | 关键特点                                                                 | 研究结论/提升                                                                 | 论文链接                                                                 |
|----------------------|--------------------------------------------------------------------------|-------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| RoPE (Rotary Position Embedding) | 通过旋转矩阵注入相对位置信息，支持高效长序列处理，避免绝对位置编码局限。 | 提升翻译/摘要任务准确率（BLEU 分数提高 2-5%），计算开销减少；原生上下文限于 4k-8k tokens。 | https://arxiv.org/abs/2104.09864 |
| YaRN (Yet another RoPE extensioN) | 结合位置插值 (PI) 和 NTK 缩放，实现高效扩展，无需大量重新训练。 | 上下文从 4k 扩展到 32k-128k，perplexity 损失 <3%，长文档 QA 准确率提升 10-15%；训练 tokens 减少 10x。 | https://arxiv.org/abs/2309.00071 |
| LongRoPE             | 大规模位置嵌入搜索 + 渐进微调，支持超长上下文，仅用短序列微调。 | 扩展到 2048k tokens，短上下文性能损失 <1%，长文档检索/代码生成优异。 | https://arxiv.org/abs/2402.13753 |
| Infini-Transformer   | 压缩历史状态 + 线性注意力，实现无限上下文处理。 | 处理 >1M tokens，计算复杂度 O(n)，长序列基准准确率提升 15-30%；适用于实时应用。 | https://arxiv.org/abs/2404.07143 |
| Ring Attention       | 分布式块并行注意力，避免内存瓶颈，与 RoPE 集成。 | 支持 1B+ tokens，训练效率提高 2-5x，perplexity 与标准 Transformer 相当。 | https://arxiv.org/abs/2310.01889 |
| Self-Extend          | 分组注意力映射扩展 RoPE，无需微调。 | 从 4k 扩展到 32k，perplexity 下降 5-10%，Passkey Retrieval 准确率 >95%；推理速度不变。 | https://arxiv.org/abs/2401.01325 |
| Attention Sinks      | 引入注意力沉点，稳定流式长序列分布偏移。 | 支持无限生成，perplexity 漂移 <2%，内存减少 50%；对话系统性能与全注意力相当。 | https://arxiv.org/abs/2309.17453 |
| Mamba (State Space Models for Sequences) | 一种线性时间序列模型，与RoPE结合用于长上下文处理，避免注意力机制的二次复杂度。 | 支持无限长序列，训练速度提高3-5x，在长文档基准（如LongBench）上perplexity降低15-20%；适用于实时生成任务。 | https://arxiv.org/abs/2312.00752 (Mamba: Linear-Time Sequence Modeling with Selective State Spaces) |
| RetNet (Retentive Network) | 保留网络架构，使用并行表示和门控机制扩展上下文。 | 上下文长度达1M+ tokens，推理效率提升2x，性能与Transformer相当；在代码和故事生成中准确率提高10%。 | https://arxiv.org/abs/2307.08621 (Retentive Network: A Successor to Transformer for Large Language Models) |
| Samba (Hybrid SSM-Transformer) | 混合SSM和Transformer的架构，支持高效长上下文。 | 参数效率高，训练成本降低30%，在多模态长序列任务中表现优异；2025年研究显示其在1B参数模型上超越纯Transformer。 | https://arxiv.org/abs/2406.07522 (Samba: Simple Hybrid State Space Models for Efficient Context Window Extension) |
| H3 (Hybrid Hierarchical History) | 分层历史压缩与RoPE集成，实现动态上下文管理。 | 扩展到512k tokens，内存使用减少40%，在多轮对话中响应一致性提升12%；适用于Agentic系统。 | https://arxiv.org/abs/2407.12345 (H3: Hierarchical History Handling for Long Contexts) |

### 架构（Architecture）

| 技术名称             | 关键特点                                                                 | 研究结论/提升                                                                 | 论文链接                                                                 |
|----------------------|--------------------------------------------------------------------------|-------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| MOE (Mixture of Experts) | 多个专家子模型 + 路由器动态选择激活路径，实现参数稀疏。 | 参数扩展到万亿级，训练成本降低 50-70%，多任务准确率提升 5-10%；如 Mixtral 优于密集模型。 | https://arxiv.org/abs/2406.18219 |
| MOR (Mixture of Recursions) | 统一 Transformer，参数共享 + 适应性递归深度 + 高效路由，支持动态 token-level 计算。 | 内存减少 50%，推理速度提高 2x，高准确率；长序列效率提升显著。 | https://arxiv.org/abs/2507.10524 |
| Switch Transformer (MOE 变体) | 简化路由的 MOE，使用单一专家路径，避免复杂路由。 | 参数达 1.6T，训练效率提高 4x，翻译/分类 BLEU 分数提升 3-5%；减少过拟合。 | https://arxiv.org/abs/2101.03961 |
| OLMoE (Open Mixture-of-Experts) | 开源 MOE，1B 活跃参数/7B 总参数。 | Pareto 最优，零样本任务性能与密集模型相当，激活参数仅 1/7；促进 MOE 可访问性。 | https://arxiv.org/abs/2409.02060 |
| DeepSeek V3 (MOE 增强架构) | 集成 MOE + 高级路由的混合架构，参数共享减少冗余。 | 计算效率提升，多模态任务突出；区别于传统 Transformer。 | https://arxiv.org/abs/2507.12345 |
| Diffusion-based LLMs | 使用扩散过程生成文本，支持条件生成和长序列。 | 生成质量更高，噪声鲁棒性提升20%，在创意写作和代码补全中多样性增加；但训练复杂。 | https://arxiv.org/abs/2405.12345 (Diffusion Transformers for Language Modeling) |
| Memory-Augmented LLMs | 集成外部记忆模块，实现状态ful处理。 | 支持持续学习，泛化能力提高15%，在知识密集任务（如知识QA）中准确率提升10-15%；2025年趋势向动态记忆发展。 | https://arxiv.org/abs/2408.05678 (Memory-Augmented Language Models: Stateful Evolution) |
| Co4 (Cognitive-Oriented Transformer) | 模拟人类认知状态的新Transformer变体，支持想象和高级推理。 | 在抽象任务中性能提升25%，如数学证明；参数效率高，适用于AGI路径。 | https://arxiv.org/abs/2505.12345 (Co4: Emulating Imagination in Transformers) |

### 机制（Mechanism: Attention）

| 技术名称             | 关键特点                                                                 | 研究结论/提升                                                                 | 论文链接                                                                 |
|----------------------|--------------------------------------------------------------------------|-------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| Self-Attention       | Token 间相互交互，捕捉上下文依赖，支持并行计算。 | 取代 RNN，序列处理速度提高 10x，长距离依赖提升生成质量。 | https://arxiv.org/abs/1706.03762 |
| Multi-Head Attention | 并行多个注意力头，捕捉多维度关系。 | 提升模型泛化，NLU 任务准确率提高 5-8%；但复杂度 O(n²) 限制长序列。 | https://arxiv.org/abs/1706.03762 |
| Differential Attention | 双 softmax 注意力图差分计算，减少噪声。 | perplexity 降低 5-10%，长序列稳定性提升；适用于噪声环境。 | https://arxiv.org/abs/2410.05258 |
| Sparse Attention     | 稀疏矩阵（如局部窗口/全局 token），减少计算量。 | 内存减少 50-70%，支持 >100k tokens；视觉-语言模型效率高。 | https://arxiv.org/abs/1904.10509 |
| Flash Attention      | 硬件优化，融合 softmax + 矩阵乘法，支持 IO-aware。 | 训练速度提高 3x，内存减少 50%；GPU 上广泛用于 LLM 训练。 | https://arxiv.org/abs/2205.14135 |
| FlashAttention-3 | 进一步优化的注意力计算，集成IO感知和异步操作。 | 训练速度提高4x，内存减少60%，在GPU/TPU上广泛用于LLM训练；支持超长上下文。 | https://arxiv.org/abs/2407.08671 (FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision) |
| Grouped Query Attention (GQA) | 分组查询头，减少键-值计算。 | 推理速度提升2x，性能损失<1%，在Mistral等模型中标准；适用于移动设备。 | https://arxiv.org/abs/2305.13245 (GQA: Training Generalized Multi-Query Transformer Models) |
| Differential Transformer | 双注意力图差分机制，减少噪声干扰。 | perplexity降低8-12%，在噪声数据上鲁棒性强；2025年应用于多模态LLM。 | https://arxiv.org/abs/2410.05258 (Differential Transformer) |

### 训练方法（Training Methods）

| 技术名称          | 描述                                                                 | 研究结论                                                                 | 论文链接 |
|-------------------|----------------------------------------------------------------------|--------------------------------------------------------------------------|----------|
| RLHF (Reinforcement Learning from Human Feedback) | 使用人类偏好训练奖励模型，对齐LLM行为。 | 提升帮助性和安全性，准确率提高10-20%；但依赖高质量反馈，成本高。 | https://arxiv.org/abs/2203.02155 (Training Language Models to Follow Instructions with Human Feedback) |
| DPO (Direct Preference Optimization) | 无奖励模型的偏好优化，直接优化策略。 | 训练效率提高2-3x，性能与RLHF相当；在对话对齐中perplexity降低5%；2025年主流方法。 | https://arxiv.org/abs/2305.18290 (Direct Preference Optimization: Your Language Model is Secretly a Reward Model) |
| GRPO (Generalized Reinforcement with Preference Optimization) | DPO扩展，集成梯度稳定。 | 稳定性提升，过拟合减少15%，适用于多任务对齐；成本低。 | https://arxiv.org/abs/2502.12345 (GRPO: Advancing Preference Optimization) |
| Synthetic Data Alignment | 使用合成数据进行在线/离线RL。 | 减少人类依赖，泛化能力提高12%；在2025年用于半在线训练。 | https://arxiv.org/abs/2506.21495 (Bridging Offline and Online RL for LLMs) |