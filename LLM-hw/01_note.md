
## 第一课. 浦语大模型全链路开源开放体系

---
1. 了解浦语大模型
   internLM2-Base:
   internLM
   internLM-chat
   internLM2:
- 超长上下文
- 对话、
- 工具调用升级
- 数据处理和分析能力
  数据-预训练-微调XTuner-部署LMDeploy-评测OpenCompass-应用Lagent AgentLego
- 微调 大语言模型的下游应用中，增量续训和有监督微调是经常会用到两种方式， 増量续训: 使用场景:让基座模型学习到一些新知识，如某个垂类领域知识文章、书籍、代码等训练数据: 全量参数 有监督微调: 使用场景:让模型学会理解各种指令进行对话，或者注入少量领域知识训练数据:高质量的对话、问答数据

- InternEvo是一个高效的预训练框架，它支持在数千个GPU上扩展模型训练，通过数据、张量、序列和流水线并行性，以及Zero Redundancy Optimizer (ZeRO)策略，显著降低了训练所需的内存占用。
- 续训和微调
    1. 续训，学到新知识
    2. 微调，理解指令
- 对齐
    - 对齐阶段包括监督式微调（SFT）和人类反馈强化学习（RLHF），以确保模型遵循人类指令并符合人类价值观。
    - 引入了一种新的条件在线RLHF（COOL RLHF）策略，通过条件奖励模型解决冲突的人类偏好，并执行多轮RLHF以减少奖励黑客攻击。

技术报告：
介绍gpt（2023）的大模型发展，包括预训练、推理、部署等，浦语的优势，对其进行了解

英特尔开源模型及智能体框架
模型推理和部署：Mdepot提供全链条的部署解决方案，支持模型轻量化和推理引擎，显示了开源社区的发展趋势。
智能体框架和工具箱：Legend框架支持多种智能体能力，AgentLego工具箱和多媒体算法功能提供了丰富的AI应用支持。