<div align="center">
  <img src="logo.svg" alt="DeepScan Framework" width="1000px"/>
</div>
<div style="height: 20px;"></div>


# DeepScan: 面向大语言模型的诊断框架 🔬

面向大语言模型与多模态大模型的可扩展诊断框架，围绕"注册 → 配置 → 执行 → 汇总"设计，提供统一的 Runner、评估器与汇总器抽象，便于快速搭建或定制诊断流水线。

## 🎨 设计与核心组件

- **📦 核心解耦与注册表**
    - 统一注册：ModelRegistry、DatasetRegistry、EvaluatorRegistry、SummarizerRegistry 通过名称查找和参数化创建，避免硬编码依赖。
    - 基础抽象：BaseRegistry 负责注册/获取，BaseEvaluator 负责评估生命周期，BaseSummarizer 负责结果协议与格式化。
    - Runner 层：BaseRunner 提供统一 generate()/chat()，屏蔽底层 HF/自研推理实现，同时保留 model/tokenizer 直达接口。
- **🔌 模型与数据源的可插拔工厂**
    - 模型：支持按"世代"或具体型号注册（如 Qwen3 各尺寸）；工厂函数接收 model_name、device、path 等参数，返回 Runner。
    - 数据集：支持 HF datasets、CSV、本地 save_to_disk；加载函数按需接受 split/path/max_rows 等参数。
    - 新增只需"写工厂 + 注册"，无需修改管线或入口脚本。
- **⚙️ 配置驱动的端到端执行**
    - ConfigLoader 支持 YAML/JSON/字典，具备多配置合并与点号访问。
    - run_from_config / CLI：解析配置 → 通过注册表实例化组件 → 推理与评估 → 输出结构化结果或单文件汇总。
    - 支持 --output-dir（分布式结果）与 --output（单 JSON 汇总）。
- **🔍 评估器的模板方法模式**
    - BaseEvaluator 规定评估流程：准备 → 数据迭代 → 指标/归因计算 → 聚合结果。
    - 两大分支：NeuronAttributionEvaluator（层/神经元选择、梯度/自定义方法），RepresentationEngineeringEvaluator（表示分析与干预）。
    - 通过 EvaluatorRegistry 注册自定义 evaluator；内置 TELLME 评估器示例化完整解耦度计算链路。
- **📝 Summarizer 的结果协议**
    - BaseSummarizer.summarize 返回轻量 dict，format_report 输出 JSON/Markdown/纯文本。
    - SummarizerRegistry 允许为特定 benchmark/任务定制关键字段、指标重命名或分层汇总。
- **🔧 扩展路径（最小改动）**
    - 统一 Runner 接口减少条件分支，便于替换推理后端或支持多模态输入。
    - 注册表集中命名空间与生命周期管理，降低模块耦合；默认资源（Qwen、BeaverTails）可在导入时自动挂载。
    - 配置与实现解耦，便于 A/B 测试、批量实验、环境切换。
    - 结果协议与可视化分离，评估算法升级不影响上层呈现。
- **🔄 典型开发工作流**
    - 导入框架，获取全局注册表（可自动加载默认模型/数据集）。
    - 用配置声明模型/数据/评估器/汇总器及参数。
    - run_from_config 或 CLI 直接执行，自动完成加载、推理、评估、汇总。
    - 用 Summarizer 生成 JSON/Markdown 报告，或自定义后处理与落盘结构。

## 📥 安装

**最小安装**（仅核心依赖）：
```bash
pip install -e .
```

**推荐安装**（包含大多数用例所需的常用依赖）：
```bash
pip install -e ".[default]"
```

**其他可选依赖**：

```bash
# 🤖 [可选] 模型运行器依赖
pip install -e ".[qwen]"          # Qwen 模型
pip install -e ".[glm]"            # GLM 模型
pip install -e ".[ministral3]"     # Ministral 3 (多模态) 模型

# 🔬 [可选] 评估器依赖
pip install -e ".[tellme]"         # TELLME 评估器
pip install -e ".[xboundary]"      # X-Boundary 评估器
pip install -e ".[mi_peaks]"       # MI-Peaks 评估器

# 🛠️ [可选] 开发依赖
pip install -e ".[dev]"            # 测试和代码质量工具

# 🎁 [可选] 所有评估器依赖
pip install -e ".[all]"            # 包含所有评估器依赖

# 🌐 [可选] API 服务器（内部使用，开源版本不包含）
pip install -e ".[api]"            # FastAPI + Uvicorn
```

## 🚀 快速上手

### 💻 CLI

```bash
# ✅ 运行 TELLME 诊断
python -m deepscan.run --config examples/config.tellme.yaml --output-dir runs

# ✅ 运行 X-Boundary 诊断
python -m deepscan.run --config examples/config.xboundary.yaml --output-dir runs

# 🔍 Dry Run: 仅验证配置与注册项（不加载模型/数据）
python -m deepscan.run --config examples/config.tellme.yaml --dry-run
```

### 🐍 Python

```python
from deepscan import run_from_config

# ✅ 运行 TELLME 诊断
results = run_from_config("examples/config.tellme.yaml")

# ✅ 运行 X-Boundary 诊断
results = run_from_config("examples/config.xboundary.yaml")

# 🔍 Dry Run: 仅验证配置与注册项（不加载模型/数据）
results = run_from_config("examples/config.tellme.yaml", dry_run=True)
results = run_from_config("examples/config.xboundary.yaml", dry_run=True)
```



## 📋 配置示例（YAML）

```yaml
model:
  generation: qwen3        # 注册表键
  model_name: Qwen3-8B
  device: cuda

dataset:
  name: tellme/beaver_tails_filtered
  test_path: /path/to/test.csv

evaluator:
  type: tellme
  batch_size: 4
  layer_ratio: 0.6666
  token_position: -1

# 可选：对运行结果进行汇总
summarizer:
  type: simple
```

## 🔄 典型工作流

1. 📥 导入框架，自动加载内置注册项（Qwen 模型族、BeaverTails 数据等）。
2. ⚙️ 在配置中声明模型/数据集/评估器/汇总器及其参数。
3. 🚀 通过 `run_from_config` 或 CLI 运行：解析配置 → 通过注册表实例化组件 → 推理与评估 → 结果落盘。
4. 📊 在 `results/<run_id>/` 下获取每个模型的 `results.json`，若指定汇总器则额外生成 `summary.json` / `summary.md`。

## 🎯 进阶用法

- **🔄 多模型批量任务**：`model` 字段可传列表，同一数据集与评估器下批量对比多个模型，结果按模型独立目录写入。
- **📈 进度回调**：`run_from_config` 接受 `on_progress_start`、`on_progress_update`、`on_progress_done` 回调，按状态上报进度百分比。也可使用 `progress_sink` 参数传入自定义进度跟踪对象。
- **📁 输出控制**：`--output-dir` 控制基目录；`--run-id` 自定义运行标识；`--output` 可将汇总直接写为单一 JSON。

## 🔍 已支持的诊断方法

- **🔬 TELLME**
  - 📊 指标：R_diff、R_same、R_gap（coding rate），erank（有效秩），cos_sim/pcc/L1/L2/hausdorff 等距离度量。
  - 📁 数据：`tellme/beaver_tails_filtered`（CSV/HF 均可，需标签列）。
  - ⚙️ 关键参数：`batch_size`、`layer` 或 `layer_ratio`、`token_position`、`max_rows`、`prompt_suffix_{train,test}`。
  - 📦 依赖：安装 `.[tellme]`（torch、transformers、pandas、opt_einsum、tqdm 等）。
  - 💡 配置示例：
    ```yaml
    evaluator:
      type: tellme
      batch_size: 4
      layer_ratio: 0.6666   # 或显式 layer: -1
      token_position: -1
    ```

- **🛡️ X-Boundary**
  - 📊 指标：基于类中心的分离度 `separation_score`、边界比 `boundary_ratio`；可选 t-SNE 可视化。
  - 📁 数据：带安全/不安全标签的对话消息（消息列表 + label），会按原论文流程做 chat template + 截断/填充。
  - ⚙️ 关键参数：`batch_size`、`max_length`、`target_layers`/`target_layers_csv`（默认取 1/3、2/3 深度）、`save_metrics_json`、`save_tsne`、t-SNE 超参（perplexity/random_state/dpi）。
  - 📦 依赖：安装 `.[xboundary]`（torch、transformers、numpy、sklearn、matplotlib、seaborn 等）。
  - 💡 配置示例：
    ```yaml
    evaluator:
      type: xboundary
      batch_size: 8
      max_length: 1024
      target_layers: [8, 24]   # 可省略，默认按深度分位
      save_tsne: true
    ```

- **🎮 SPIN**
  - 📊 指标：自对弈微调评估指标。
  - 📁 数据：需要特定的训练/测试数据集。
  - ⚙️ 关键参数：`batch_size`、`max_length` 等。
  - 📦 依赖：安装 `.[tellme]` 或 `.[all]`。
  - 💡 配置示例：
    ```yaml
    evaluator:
      type: spin
      batch_size: 4
      max_length: 512
    ```

- **📈 MI-Peaks**
  - 📊 指标：模型内省与峰值分析。
  - 📁 数据：支持多种数据集格式。
  - ⚙️ 关键参数：`batch_size`、`target_layers` 等。
  - 📦 依赖：安装 `.[mi_peaks]` 或 `.[all]`。
  - 💡 配置示例：
    ```yaml
    evaluator:
      type: mi_peaks
      batch_size: 4
      target_layers: [8, 16, 24]
    ```
## 🔧 扩展指引

- **➕ 新增模型** 🏗️
  ```python
  from deepscan.registry.model_registry import get_model_registry
  registry = get_model_registry()

  @registry.register_model("my_llm", model_family="custom")
  def create_my_llm(device="cuda", **kwargs):
      ...  # 返回 BaseRunner 实例
  ```
- **➕ 新增数据集** 📊
  ```python
  from deepscan.registry.dataset_registry import get_dataset_registry
  registry = get_dataset_registry()

  @registry.register_dataset("my_dataset")
  def load_my_dataset(path: str, split: str = "train"):
      ...  # 返回可迭代/索引的数据对象
  ```
- **➕ 新增评估器** 🔍
  ```python
  from deepscan.evaluators import NeuronAttributionEvaluator
  from deepscan.evaluators.registry import get_evaluator_registry

  class MyEvaluator(NeuronAttributionEvaluator):
      def _compute_attributions(self, model, dataset, target_layers, target_neurons=None):
          ...  # 返回指标/归因结果

  get_evaluator_registry().register_evaluator("my_eval")(MyEvaluator)
  ```
- **➕ 新增汇总器** 📝
  ```python
  from deepscan.summarizers import BaseSummarizer
  from deepscan.summarizers.registry import get_summarizer_registry

  class MySummarizer(BaseSummarizer):
      def summarize(self, results, benchmark=None, **kwargs):
          return {"keys": sorted(results.keys())}

  get_summarizer_registry().register_summarizer("my_sum")(MySummarizer)
  ```

## 🎁 内置资源与示例

- **🤖 模型**：自动注册多个模型系列
  - Qwen 系列：qwen / qwen2 / qwen2.5 / qwen3
  - Llama 系列
  - Mistral 系列（包括 Ministral3 多模态）
  - Gemma 系列（包括 Gemma3 多模态）
  - GLM 系列
  - InternLM 系列
  - InternVL 系列（多模态）
- **📊 数据集**：BeaverTails（HF）及过滤 CSV 版本，支持 `datasets.load_dataset` 或 `save_to_disk` 本地加载。
- **🔍 评估器**：
  - 🔬 TELLME：解耦度指标
  - 🛡️ X-Boundary：可视化与边界分析
  - 📈 MI-Peaks：模型内省与峰值分析
  - 🎮 SPIN：自对弈微调评估
- **📚 示例**：
  - `examples/config.tellme.yaml`、`examples/tellme_evaluation.py`：TELLME 评估
  - `examples/config.xboundary.yaml`、`examples/xboundary_evaluation.py`：X-Boundary 评估
  - `examples/ministral3_multimodal_demo.py`：多模态模型示例
  - `examples/gemma3_multimodal_demo.py`：Gemma3 多模态示例