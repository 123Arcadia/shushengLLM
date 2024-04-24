## XTuner 微调 LLM


---

### XTuner 微调多模态LLM

---


![img.png](../images/04_note_01.png)

安装一个 XTuner 的源码:

`studio-conda xtuner0.1.1`

- 数据集准备


- 模型准备

```shell
|-- model/
    |-- tokenizer.model
    |-- config.json
    |-- tokenization_internlm2.py
    |-- model-00002-of-00002.safetensors
    |-- tokenizer_config.json
    |-- model-00001-of-00002.safetensors
    |-- model.safetensors.index.json
    |-- configuration.json
    |-- special_tokens_map.json
    |-- modeling_internlm2.py
    |-- README.md
    |-- configuration_internlm2.py
    |-- generation_config.json
    |-- tokenization_internlm2_fast.py
```

- 配置文件选择

  用于定义和控制模型训练和测试过程中各个方面的参数和设置的工具

```shell
# 列出所有内置配置文件
# xtuner list-cfg

# 假如我们想找到 internlm2-1.8b 模型里支持的配置文件
xtuner list-cfg -p internlm2_1_8b
```

    - list-cfg: 配置文件搜索：可以加上一个参数 -p 或 --pattern ，后面输入的内容将会在所有的 config 文件里进行模糊匹配搜索
    
    - copy-cfg： {CONFIG_NAME} 对应的是上面搜索到的 internlm2_1_8b_qlora_alpaca_e3 ,而 {SAVE_PATH} 则对应的是刚刚新建的 /root/ft/config

        `xtuner copy-cfg {CONFIG_NAME} {SAVE_PATH}




- 模型训练

  模型过拟合：

  措施：减少保存权重文件的间隔并增加权重文件保存的上限、增加常规的对话数据集从而稀释原本数据的占比



LoRA 或者 QLoRA ： 微调得到的是一个层

