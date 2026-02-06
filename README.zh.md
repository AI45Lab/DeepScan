<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="logo-light.svg">
    <img alt="DeepScan Logo" src="logo-dark.svg" width="200">
  </picture>

  <br>
  
  <h1>DeepScan: Diagnostic Framework for LLMs</h1>

  <!-- <p><b>DeepScan: Diagnostic Framework for LLMs</b></p> -->

  <p>
    <a href="https://ai45.shlab.org.cn/safety-entry"><img alt="AI45"
      src="https://img.shields.io/badge/AI45-Homepage-0066CC?style=flat-square"></a>
    <!--    <a href="https://github.com/your-org/DeepScan"><img alt="GitHub" -->
      <!-- src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white"></a> -->
    <a href="https://huggingface.co/your-org/DeepScan"><img alt="Hugging Face"
      src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black"></a>
    <a href="https://arxiv.org/abs/XXXX.XXXXX"><img alt="arXiv"
      src="https://img.shields.io/badge/arXiv-b31b1b?style=flat-square&logo=arxiv&logoColor=white"></a>
    <a href="https://deepscan.readthedocs.io"><img alt="Documentation"
      src="https://img.shields.io/badge/Documentation-8CA1AF?style=flat-square&logo=readthedocs&logoColor=white"></a>
  </p>
</div>


面向大语言模型与多模态大模型的可扩展诊断框架，围绕"注册 → 配置 → 执行 → 汇总"设计，提供统一的 Runner、评估器与汇总器抽象，便于快速搭建或定制诊断流水线。

> **🛡️ 安全评测：** 配套项目 **[DeepSafe](https://github.com/AI45Lab/DeepSafe)** 专注于安全评测，与 DeepScan 组合可形成 **评测-诊断的工业闭环**。

## ✨ 特性

- **📦 模型注册表**：注册和管理模型实例，支持 Qwen、Llama、Mistral、Gemma、GLM、InternLM、InternVL 等
- **🚀 统一模型接口**：一致的 `generate`/`chat` 抽象，跨模型族统一使用
- **📊 数据集注册表**：注册和管理数据集实例，支持多种数据格式
- **⚙️ 配置管理**：从 YAML/JSON 文件加载和管理配置
- **🔍 可扩展Evaluator**：内置多种诊断Evaluator：
  - **TELLME**：用指标量化表征中不同概念的解耦程度
  - **X-Boundary**：诊断隐表示空间：安全/有害/边界几何关系
  - **MI-Peaks**：基于互信息追踪生成中推理表征信息演化
  - **SPIN**：分析公平、隐私等安全目标间的潜在冲突
- **📝 可定制Summarizer**：聚合和格式化不同基准的评估结果
- **🔌 插件架构**：易于扩展自定义Evaluator和Summarizer
- **💻 CLI 支持**：无需编写代码即可从命令行运行评估

## 📥 安装

**最小安装**（仅核心依赖）：
```bash
pip install -e .
```

**推荐安装**（包含大多数用例所需的常用依赖）：
```bash
pip install -e ".[default]"
```

**开发模式安装**：
```bash
pip install -e ".[dev]"
```

**其他可选依赖**：

```bash
# 🤖 模型运行器依赖
pip install -e ".[qwen]"          # Qwen 模型
pip install -e ".[glm]"           # GLM 模型
pip install -e ".[ministral3]"    # Ministral 3 (多模态) 模型

# 🔬 评估器依赖
pip install -e ".[tellme]"        # TELLME 评估器 + 指标栈
pip install -e ".[xboundary]"     # X-Boundary 评估器 + 可视化栈
pip install -e ".[mi_peaks]"     # MI-Peaks 评估器

# 🎁 便捷扩展
pip install -e ".[all]"           # 所有评估器依赖 (tellme + xboundary + mi_peaks)

```

## 🚀 快速上手

### 🎯 从配置文件端到端运行（任何评估器）

**Python API：**
```python
from deepscan import run_from_config

# YAML/JSON 或包含 model/dataset/evaluator 部分的字典
results = run_from_config("examples/config.tellme.yaml")

# 指定输出目录和运行 ID
results = run_from_config(
    "examples/config.tellme.yaml",
    output_dir="results",
    run_id="my_experiment",
)
```

**CLI**（无需编写 Python 代码）：
```bash
# ✅ 基本用法
python -m deepscan.run --config examples/config.tellme.yaml --output-dir runs

# 🏷️ 使用自定义运行 ID（可选；默认为 run_<时间戳>）
python -m deepscan.run --config examples/config.tellme.yaml --output-dir runs --run-id experiment_001

# 🔍 干运行（验证配置而不加载模型/数据集）
python -m deepscan.run --config examples/config.tellme.yaml --dry-run

# 💾 可选：同时将单个合并的 JSON 写入指定位置
python -m deepscan.run --config examples/config.tellme.yaml --output results.json
```

### 1. 注册模型和数据集（全局注册表）📝

#### 🔧 注册单个模型

```python
from deepscan.registry.model_registry import get_model_registry
from deepscan.registry.dataset_registry import get_dataset_registry

# 重要：使用全局注册表，以便 `run_from_config()` 可以找到您的条目
model_registry = get_model_registry()
dataset_registry = get_dataset_registry()

@model_registry.register_model("gpt2")
def create_gpt2():
    from transformers import GPT2LMHeadModel
    return GPT2LMHeadModel.from_pretrained("gpt2")

@dataset_registry.register_dataset("glue_sst2")
def create_sst2():
    from datasets import load_dataset
    return load_dataset("glue", "sst2", split="test")
```

#### 注册模型系列（例如 Qwen）🏗️


**选项 1：按世代注册（推荐）** ⭐

```python
from deepscan.registry.model_registry import get_model_registry

registry = get_model_registry()

@registry.register_model(
    "qwen3",
    model_family="qwen",
    model_generation="qwen3",
)
def create_qwen3(model_name: str = "Qwen3-8B", device: str = "cuda", **kwargs):
    """创建指定名称的 Qwen3 模型。"""
    from transformers import AutoModelForCausalLM
    
    model_paths = {
        "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
        "Qwen3-1.5B": "Qwen/Qwen3-1.5B",
        "Qwen3-2B": "Qwen/Qwen3-2B",
        "Qwen3-8B": "Qwen/Qwen3-8B",
        "Qwen3-14B": "Qwen/Qwen3-14B",
        "Qwen3-32B": "Qwen/Qwen3-32B",
    }
    
    if model_name not in model_paths:
        raise ValueError(f"不支持的模型: {model_name}")
    
    return AutoModelForCausalLM.from_pretrained(
        model_paths[model_name],
        device_map=device,
        **kwargs
    )

# 用法：
runner = registry.get_model("qwen3", model_name="Qwen3-8B", device="cuda")
```

**选项 2：使用世代前缀注册单个模型** 🔑

```python
@registry.register_model(
    "qwen3/Qwen3-8B",
    model_family="qwen",
    model_generation="qwen3",
    model_name="Qwen3-8B",
)
def create_qwen3_8b(device: str = "cuda", **kwargs):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        device_map=device,
        **kwargs
    )

# 用法：
runner = registry.get_model("qwen3/Qwen3-8B", device="cuda")
```

**💡 为什么按世代组织？** 不同的 Qwen 世代（qwen、qwen2、qwen3）即使在相同的参数数量下也可能具有不同的架构、分词器和配置。

**选项 3：使用预注册模型（推荐）** ⭐

Qwen 模型在导入框架时会自动注册：

```python
from deepscan.registry.model_registry import get_model_registry

# 模型已注册 - 直接使用！
registry = get_model_registry()
runner = registry.get_model("qwen3", model_name="Qwen3-8B", device="cuda")
```

或者简单地导入框架，模型即可使用：

```python
import deepscan  # Qwen 模型自动注册
from deepscan.registry.model_registry import get_model_registry

registry = get_model_registry()
runner = registry.get_model("qwen3", model_name="Qwen3-8B", device="cuda")
```

查看 `deepscan/models/` 了解模型实现，查看 `examples/` 了解端到端评估流水线。

#### 预注册资源（模型 + 数据集）🎁

框架附带开箱即用的注册项，在您 `import deepscan` 时会自动加载：

**🤖 模型**（查看 `deepscan/models/` 了解实现）：
- **Qwen**: qwen / qwen2 / qwen2.5 / qwen3 变体
- **Llama**: Llama 2/3 变体
- **Mistral**: Mistral 和 Ministral3（多模态）
- **Gemma**: Gemma 和 Gemma3（多模态）
- **GLM**: GLM-4 系列
- **InternLM**: InternLM2/3 变体
- **InternVL**: InternVL3.5（多模态）

**📊 数据集**：
- BeaverTails（HF 数据集）
- `tellme/beaver_tails_filtered`（CSV 加载器）
- `xboundary/diagnostic`（X-Boundary 诊断数据集）

```python
import deepscan
from deepscan.registry.model_registry import get_model_registry
from deepscan.registry.dataset_registry import get_dataset_registry

model_registry = get_model_registry()
dataset_registry = get_dataset_registry()

# 使用任何已注册的模型
model = model_registry.get_model("qwen3", model_name="Qwen3-8B", device="cuda")
# 或 Llama、Mistral 等

# 使用已注册的数据集
dataset = dataset_registry.get_dataset("tellme/beaver_tails_filtered", test_path="/path/to/test.csv")
```

> 💡 内置数据集加载器依赖于 Hugging Face `datasets`（包含在核心依赖中）。

要使用通过 `datasets.save_to_disk` 保存的副本，传递本地路径：

```python
dataset = dataset_registry.get_dataset(
    "beaver_tails",
    split="330k_train",
    path="/path/to/BeaverTails",
)
```

### 🏃 模型运行器

模型注册表查找现在返回一个**模型运行器**——一个暴露统一 `generate()` 接口并保留底层 Hugging Face 模型/分词器的对象。

```python
from deepscan.models.base_runner import GenerationRequest, PromptMessage, PromptContent
from deepscan.registry.model_registry import get_model_registry

runner = get_model_registry().get_model(
    "qwen3",
    model_name="Qwen3-8B",
    device="cuda",
)

# 快速文本生成
response = runner.generate("用两句话解释什么是注册表模式。")
print(response.text)

# 带结构化消息的聊天式提示
chat_request = GenerationRequest.from_messages(
    [
        PromptMessage(role="system", content=[PromptContent(text="你是一位数学导师。")]),
        PromptMessage(role="user", content=[PromptContent(text="帮我因式分解 x^2 + 5x + 6。")]),
    ],
    temperature=0.1,
    max_new_tokens=128,
)
chat_response = runner.generate(chat_request)
print(chat_response.text)
```

运行器通过 `runner.model` / `runner.tokenizer` 保留原始模型/分词器的访问，以便现有诊断代码在需要时仍可访问底层 API。

### 2. 加载配置 ⚙️

```python
from deepscan import ConfigLoader

# 从文件加载
config = ConfigLoader.from_file("config.yaml")

# 或从字典创建
config = ConfigLoader.from_dict({
    "model": {"generation": "qwen3", "model_name": "Qwen3-8B", "device": "cuda"},
    "dataset": {"name": "beaver_tails", "split": "330k_train"},
    "evaluator": {"type": "tellme", "batch_size": 4},
})
```

### 3. 创建和使用评估器 🔍

评估器通常通过 `run_from_config` 使用，但也可以以编程方式使用：

```python
from deepscan.evaluators.registry import get_evaluator_registry
from deepscan.registry.model_registry import get_model_registry
from deepscan.registry.dataset_registry import get_dataset_registry

# 获取注册表
evaluator_registry = get_evaluator_registry()
model_registry = get_model_registry()
dataset_registry = get_dataset_registry()

# 从注册表创建评估器（通过 config= 传递参数）
evaluator = evaluator_registry.create_evaluator(
    "tellme",
    config=dict(batch_size=4, layer_ratio=0.6666, token_position=-1),
)

# 获取模型和数据集
model = model_registry.get_model("qwen3", model_name="Qwen3-8B", device="cuda")
dataset = dataset_registry.get_dataset("tellme/beaver_tails_filtered", test_path="/path/to/test.csv")

# 运行评估（通常通过 run_from_config 完成）
# results = evaluator.evaluate(model, dataset, ...)
```

### 4. 创建自定义评估器 🛠️

```python
from deepscan.evaluators.base import BaseEvaluator
from deepscan.evaluators.registry import get_evaluator_registry

class CustomEvaluator(BaseEvaluator):
    def evaluate(self, model, dataset, **kwargs):
        # 您的自定义评估逻辑
        results = {}
        # ... 实现 ...
        return results

# 注册评估器
registry = get_evaluator_registry()
registry.register_evaluator("custom_eval")(CustomEvaluator)

# 在配置中或以编程方式使用（通过 config= 传递参数）
evaluator = registry.create_evaluator("custom_eval", config=dict(param1=value1))
```

### 5. 汇总结果 📊

```python
from deepscan.summarizers.base import BaseSummarizer

class SimpleSummarizer(BaseSummarizer):
    def summarize(self, results, benchmark=None, **kwargs):
        # 最小示例：保留一小部分键
        return {
            "benchmark": benchmark,
            "keys": sorted(results.keys()),
        }

summarizer = SimpleSummarizer(name="simple")

summary = summarizer.summarize(results, benchmark="beaver_tails")

# 格式化为 markdown
report = summarizer.format_report(summary, format="markdown")
print(report)
```

## 🏗️ 架构

### 核心组件

1. **📦 注册表系统** (`deepscan/registry/`)
   - `BaseRegistry`: 通用注册表模式
   - `ModelRegistry`: 模型注册和检索
   - `DatasetRegistry`: 数据集注册和检索

2. **⚙️ 配置** (`deepscan/config/`)
   - `ConfigLoader`: 加载和管理 YAML/JSON 配置
   - 支持点号访问嵌套配置
   - 合并多个配置

3. **🔍 评估器** (`deepscan/evaluators/`)
   - `BaseEvaluator`: 所有评估器的抽象基类
   - `TellMeEvaluator`: BeaverTails 上的解耦度指标
   - `XBoundaryEvaluator`: 安全边界分析
   - `MiPeaksEvaluator`: 模型内省和峰值分析
   - `SpinEvaluator`: 自对弈微调评估
   - `EvaluatorRegistry`: 评估器类的注册表

4. **📝 汇总器** (`deepscan/summarizers/`)
   - `BaseSummarizer`: 所有汇总器的抽象基类
   - `SummarizerRegistry`: 汇总器类的注册表
   - 多种输出格式（dict、JSON、Markdown、文本）

## 🔧 扩展框架

### ➕ 添加新评估器

1. 继承 `BaseEvaluator`
2. 实现 `evaluate` 方法
3. 使用评估器注册表注册

```python
from deepscan.evaluators.base import BaseEvaluator
from deepscan.evaluators.registry import get_evaluator_registry

class MyEvaluator(BaseEvaluator):
    def evaluate(self, model, dataset, **kwargs):
        # 您的评估逻辑
        results = {}
        # ... 实现 ...
        return results

# 注册
registry = get_evaluator_registry()
registry.register_evaluator("my_evaluator")(MyEvaluator)
```

### ➕ 添加新汇总器

1. 继承 `BaseSummarizer`
2. 实现 `summarize` 方法
3. 使用汇总器注册表注册

```python
from deepscan.summarizers.base import BaseSummarizer
from deepscan.summarizers.registry import get_summarizer_registry

class MySummarizer(BaseSummarizer):
    def summarize(self, results, benchmark=None, **kwargs):
        # 您的汇总逻辑
        return {"summary": "..."}

# 注册
registry = get_summarizer_registry()
registry.register_summarizer("my_summarizer")(MySummarizer)
```

## 📋 配置示例文件

```yaml
# 📝 最小 TELLME 风格配置（见 `examples/config.tellme.yaml`；多评估器配置见 `examples/`）
model:
  generation: qwen3
  model_name: Qwen3-8B
  device: cuda
  dtype: float16
  # 可选：指向本地检查点目录以避免下载
  # path: /path/to/models--Qwen--Qwen3-8B

dataset:
  name: tellme/beaver_tails_filtered
  test_path: /path/to/test.csv
  # train_path: /path/to/train.csv
  # max_rows: 400

evaluator:
  type: tellme
  batch_size: 4
  layer_ratio: 0.6666
  token_position: -1
```

## 📚 示例

查看 `examples/` 目录了解完整使用示例：

**🔍 单评估器：**
- `config.tellme.yaml`: 最小 TELLME 解耦度指标配置

**🔗 多评估器（同一模型、多基准）：**
- `config.x-boundary.tellme.spin.mi-peaks.qwen2.5-7b-instruct.yaml`: TELLME、X-Boundary、SPIN、MI-Peaks 与 Qwen2.5-7B-Instruct
- `config.xboundary-llama3.3-70b-instruct.yaml`: 同上评估套件，使用 Llama 3.3 70B Instruct

## 💻 开发

```bash
# 📦 以开发模式安装
pip install -e ".[dev]"

# ✅ 运行测试
pytest

```

## 📄 许可证

MIT License

## 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何贡献的指南。

## 📧 联系我们

如有问题或建议，请通过以下方式联系我们：

  📧 邮箱：[shaojing@pjlab.org.cn](mailto:shaojing@pjlab.org.cn)