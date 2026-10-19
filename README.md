# 宏观权重

基于 **Qwen/Qwen3.5-4B**，使用数学、经济学与中国政策问答及知识卡进行 QLoRA 微调的 LoRA 适配器。

> 适配器按 Apache-2.0 发布，保留基础模型许可及来源，并明确标注微调变更。

## 发布内容

- `adapter_model.safetensors`：网站实际使用的 higher 第 341 步适配器，55,888,736 字节。
- `adapter_config.json`：将机器本地基础模型路径改为公开模型标识，权重内容不变。
- 权重参数说明、Apache-2.0 许可、来源及变更说明和 SHA-256 校验文件。分词器从基础模型取得。

这是增量适配器，不能独立运行，需要另行取得基础模型。教材 SQL 库、训练题答、验证/留出题、运行日志、账号信息均不在本目录中。

## 训练记录

训练集有 1,865 道整题与 186 张知识卡，共 2,051 条独立样本。整题按数学、经济学、政策 30:50:20 抽样，知识卡每个采样轮另加入一次。

LoRA r=8、alpha=16、dropout=0，学习率 1e-4，单批次 1、梯度累积 8，固定种子 20261017。验证每 50 次更新和轮末执行，至少一轮后使用 patience=3、min_delta=0.005 的早停规则，上限 30 轮。该组实际运行 491 次更新，约 1.4402 个采样轮；发布权重来自第 341 步，而非最后一步。

最佳合格验证损失为 1.7094660086（238 道验证题的回答 token 加权交叉熵）。它不是答题正确率。本发布不宣称专家级能力、完整教材掌握程度或未知题正确率。模型可能出现事实混淆和推理错误。

## 简介与适用范围

宏观权重是面向中文经济分析的实验性适配器，可用于经济概念解释、模型与机制分析、政策传导讨论，以及辅助理解宏观数据。训练包含数学、经济学、中国政策问答与知识卡；本仓库只公开权重及其说明，不公开训练材料。

权重不自带实时数据、新闻检索或网站接口。分析当前经济形势时，应在提问中提供有日期、单位和口径的数据。生成内容可能存在事实或推理错误，不能把验证损失视为正确率，也不能据此保证投资判断。

## 下载与使用

1. 在本仓库 [Releases](https://github.com/DeanZhangzhengwei/hongguan-quanzhong/releases) 下载 `hongguan-quanzhong-v1.zip` 并解压。
2. 将 `adapter_model.safetensors` 与 `adapter_config.json` 放在同一目录，例如 `./hongguan-adapter`。这是 LoRA 增量权重，不是 GGUF，也不是能单独启动的完整模型。
3. 安装支持 Qwen3.5 的 PyTorch、Transformers、PEFT、Accelerate；下例另需 bitsandbytes，并面向支持 BF16 的 NVIDIA CUDA 环境。PyTorch 请按自己的 CUDA 环境安装。

实际训练与生成环境的依赖版本见 `model_card.json`。其他硬件或版本组合未逐一验证，以下四位量化示例的显存需求会随输入及输出长度变化，不保证在仅剩 3 GB 显存时可用。

```bash
python -m pip install transformers==5.20.1 peft==0.21.2 accelerate==1.15.0 bitsandbytes==0.50.2
```

加载与提问示例（第一次运行需要下载基础模型及分词器）：

```python
import torch
from peft import PeftModel
from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

base_id = "Qwen/Qwen3.5-4B"
adapter_dir = "./hongguan-adapter"  # 改为解压后的目录
tokenizer = AutoTokenizer.from_pretrained(base_id)
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base = Qwen3_5ForConditionalGeneration.from_pretrained(
    base_id, quantization_config=quant, device_map={"": 0},
    dtype=torch.bfloat16, attn_implementation="sdpa",
)
model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
model.eval()
messages = [
    {"role": "system", "content": "你是经济分析助手。依据给定数据解释机制，明确假设和不确定性。"},
    {"role": "user", "content": "请解释降息通过哪些渠道影响消费、投资与物价，并说明传导受阻的条件。"},
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda:0")
with torch.inference_mode():
    output = model.generate(
        **inputs, max_new_tokens=1024, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
answer = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(answer)
```

`enable_thinking=False` 用于请求直接回答。此示例使用实际生成器中的模型类与加载方式，未针对所有设备环境运行验证。若达到输出长度上限，回答可能不完整。

## 使用前提

基础模型：<https://huggingface.co/Qwen/Qwen3.5-4B>。加载基础模型后，通过 PEFT 加载本目录的适配器。模型结构与较新版本的 Transformers/PEFT 有关；实际训练环境版本见 `model_card.json`。原始下载的基础模型提交版本未在所检查的训练清单中记录。

权重 SHA-256：

```text
a070dec4e052d60b0587f34994151de306b1de0596573cd369bafa2125493c00
```

## 许可与来源

基础模型 Qwen/Qwen3.5-4B 来自 Qwen 团队，采用 Apache-2.0，允许在遵守条款的前提下修改和分发。适配器与本仓库自有说明按 Apache-2.0 发布，见 `LICENSE`；上游许可原文另保存在 `BASE_MODEL_LICENSE.txt`。来源和微调变更见 `NOTICE`。这不是 Qwen 官方微调版本，不表示获得其背书。

本发布包不包含教材和训练题库，也不授予这些资料的再分发权。训练材料的来源记录与权重的发布许可分别管理；本地训练确认未被记录为出版社授权。上述元数据情况本身并不构成禁止发布适配器的依据，Apache-2.0 也不代替独立第三方材料的权利。

## GitHub 发布

目标仓库：`DeanZhangzhengwei/hongguan-quanzhong`。README 标题为“宏观权重”。适配器约 53.3 MiB，可将权重作为 GitHub Release 附件发布，仓库保留说明、配置和校验文件。权重下载见仓库 Releases。
