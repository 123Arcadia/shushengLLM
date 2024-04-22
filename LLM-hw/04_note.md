## Huggingface与TurboMind

### HuggingFace

HuggingFace是一个高速发展的社区，包括Meta、Google、Microsoft、Amazon在内的超过5000家组织机构在为HuggingFace开源社区贡献代码、数据集和模型。可以认为是一个针对深度学习模型和数据集的在线托管社区，如果你有数据集或者模型想对外分享，网盘又不太方便，就不妨托管在HuggingFace。
托管在HuggingFace社区的模型通常采用HuggingFace格式存储，简写为HF格式。

但是HuggingFace社区的服务器在国外，国内访问不太方便。国内可以使用阿里巴巴的MindScope社区，或者上海AI Lab搭建的OpenXLab社区，上面托管的模型也通常采用HF格式。

### TurboMind

TurboMind是LMDeploy团队开发的一款关于LLM推理的高效推理引擎，它的主要功能包括：LLaMa 结构模型的支持，continuous batch 推理模式和可扩展的 KV 缓存管理器。

TurboMind推理引擎仅支持推理TurboMind格式的模型。因此，TurboMind在推理HF格式的模型时，会首先自动将HF格式模型转换为TurboMind格式的模型。该过程在新版本的LMDeploy中是自动进行的，无需用户操作。

几个容易迷惑的点：

- TurboMind与LMDeploy的关系：LMDeploy是涵盖了LLM 任务全套轻量化、部署和服务解决方案的集成功能包，TurboMind是LMDeploy的一个推理引擎，是一个子模块。LMDeploy也可以使用pytorch作为推理引擎。

- TurboMind与TurboMind模型的关系：TurboMind是推理引擎的名字，TurboMind模型是一种模型存储格式，TurboMind引擎只能推理TurboMind格式的模型。

**常用预训练框架：**

`ls /root/share/new_models/Shanghai_AI_Laboratory/`

![img.png](../images/04_01.png)