# IFlow Agent

IFlow Agent 是一个面向真实任务的通用多模态 Multi-Agent 执行与评测框架。系统通过 MainAgent 进行任务规划和结果汇总，将子任务并行委派给 ReAct SubAgent，并通过统一工具协议处理联网搜索、网页读取、图片、音频、视频和代码执行任务。

项目同时提供独立评测系统，用于区分“Agent 已完成任务”和“任务结果正确”，支持答案质量、工具路由、执行成本、延迟和稳定性等多维指标。

## 核心能力

- **多 Agent 协作**：MainAgent 负责规划与汇总，SubAgent 负责工具执行和证据收集，支持并行子任务。
- **统一工具协议**：工具通过 `BaseAction` 注册，自动生成参数描述、执行环境和调用轨迹。
- **上下文与状态隔离**：每个并行 SubAgent 使用独立 Environment clone，避免步骤、完成状态和工具权限相互污染。
- **模型能力路由**：支持 OpenAI 兼容接口，可为规划、文本、视觉和 ASR 分配不同模型。
- **多模态处理**：支持图片、音频和视频；视频具备直接识别、FFmpeg 抽帧、OpenCV 回退和音轨转写链路。
- **可观测执行**：记录答案、Main/SubAgent trace、工具调用、token、成本、耗时和任务状态。
- **通用评测**：支持 JSONL/JSON/YAML 用例、7 类评分器、并发、重复运行、超时隔离及 JSON/CSV 报告。

## 运行架构

```text
User Task + Attachments
  -> GeneralAgent
    -> MainAgent（任务规划、并行委派、答案汇总）
      -> DelegateTaskTool
        -> ToolEnvironment.clone()
          -> Runner
            -> ReAct SubAgent
              -> BaseAction Tools
                -> Search / Web / Image / Audio / Video / Code
    -> GeneralAgentResult
      -> Answer + Trace + Tokens + Cost + Latency
```

## 项目结构

| 目录 | 职责 |
|---|---|
| `base/agent/` | Agent、工具协议和 Memory 上下文管理 |
| `base/engine/` | OpenAI 兼容模型客户端、重试、token 与成本记录 |
| `master/` | MainAgent、ReAct SubAgent、任务委派和提示词 |
| `runtime/` | Environment 协议、工具环境和 Agent 执行循环 |
| `tools/` | 搜索、网页、图片、音频、视频和代码工具 |
| `evaluation/` | 用例加载、评分器、Judge、批量评测和报告 |
| `general_agent.py` | 通用 Agent 公共入口与结果模型 |
| `config/` | 模型和工具 API 配置 |

## API 配置

模型配置链路：

```text
model_config.yaml -> LLMsConfig -> AsyncLLM -> AsyncOpenAI
```

IFlow Agent 直接使用 OpenAI 兼容协议连接模型服务。MiMo 提供兼容端点，因此配置 `base_url`、`api_key` 和实际模型名称即可，不依赖额外的模型聚合网关。

实际配置位于 `iflow_agent2/config/model_config.yaml`，MiMo 和 Tavily 密钥直接明文保存在这一个本地文件中。该文件已被 `iflow_agent2/.gitignore` 忽略；仓库只应保留不含密钥的 `model_config.example.yaml`。

默认模型别名和实际请求模型如下：

| 配置别名 | API 模型 | 用途 |
|---|---|---|
| `mimo-pro` | `mimo-v2.5-pro` | MainAgent、复杂推理、结果汇总 |
| `mimo` | `mimo-v2.5` | SubAgent、普通文本、图片与视频理解 |
| `mimo-asr` | `mimo-v2.5-asr` | 音频和视频音轨转写 |
| `vision` | `mimo-v2.5` | 图片、直接视频和视频抽帧理解 |

`mimo-v2.5-pro` 会消耗一部分输出预算进行内部推理。不要把 `max_tokens` 设得过小，否则可能出现 `finish_reason=length` 且正文为空；live test 使用 100，Agent 默认不主动压缩该预算。

也可以用 `IFLOW_AGENT_MODEL_CONFIG=/path/to/models.yaml` 指定另一份 YAML。环境变量占位符解析仅作为兼容能力保留，当前默认配置不需要环境变量。

`from_model_names()` 根据模型别名创建 MainAgent 和 SubAgent：

```python
import asyncio

from iflow_agent2 import BaseAction, GeneralAgent


class MyTool(BaseAction):
    name: str = "MyTool"
    description: str = "Return data for one query."
    parameters: dict = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    async def __call__(self, query: str = "", **kwargs):
        return {"success": True, "output": f"result for {query}"}


async def main():
    agent = GeneralAgent.from_model_names(
        main_model="mimo-pro",
        sub_models=["mimo"],
        tools=[MyTool()],
        max_attempts=5,
        subagent_max_steps=12,
    )
    result = await agent.run("Use MyTool to answer the request.")
    print(result.answer)


asyncio.run(main())
```

真实密钥只写在已忽略的 `model_config.yaml`，不要写入示例文件或测试文件。如果该文件曾经被 Git 跟踪过，仅添加 `.gitignore` 不会自动取消跟踪，提交前应检查 Git 状态。

## 搜索、图片和视频

联网搜索使用 `TavilySearchAction`，直接读取同一份 YAML 中的 `tools.tavily.api_key`：

```python
from iflow_agent2.tools import TavilySearchAction

agent = GeneralAgent.from_model_names(
    main_model="mimo-pro",
    sub_models=["mimo"],
    tools=[TavilySearchAction()],
)
```

实际 API 探测确认：`mimo-v2.5` 支持 base64 图片和 `video_url` 视频输入，`mimo-v2.5-pro` 会拒绝图片输入。因此 `vision` 别名映射到 base 版，Pro 只承担文本推理。

`VideoAnalysisAction` 对 20 MB 以内的完整视频优先使用 MiMo `video_url`；直接输入失败、文件较大或分析指定时间段时，自动用 FFmpeg 抽取 JPEG 帧，并以 OpenCV 作为最终回退。音轨通过 FFmpeg 转成 16 kHz 单声道 WAV，再交给 `mimo-asr`，最后由 `mimo-pro` 汇总画面与音频结果。

MiMo ASR 的网关要求请求只有 `input_audio`，不能附带文本 part。音频工具已针对 `mimo-v2.5-asr` 做此适配；其他 OpenAI 兼容音频模型仍使用“文本提示 + 音频”的通用格式。

## 注入其他 LLM

为了测试或使用自定义客户端，可以直接传入对象。Main/SubAgent 保持原接口要求：

```python
class CustomLLM:
    async def __call__(self, prompt: str, max_tokens=None) -> str:
        return '{"action":"complete","params":{"answer":"done"}}'

    def get_usage_summary(self) -> dict:
        return {
            "model": "custom",
            "total_cost": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }


agent = GeneralAgent(
    main_llm=CustomLLM(),
    sub_models=["worker"],
    tools=[MyTool()],
    llm_factory=lambda model_name: CustomLLM(),
)
```

生产环境建议让 `llm_factory` 为每个并行 SubAgent 创建独立客户端实例。

## 多模态附件

附件使用通用路径、URL 或字典格式：

```python
result = await agent.run(
    "分析这些附件",
    attachments=[
        {"type": "image", "path": "photo.png"},
        {"type": "audio", "path": "meeting.mp3"},
        {"type": "video", "url": "https://example.com/demo.mp4"},
    ],
)
```

Environment 会把附件写入 MainAgent/SubAgent 的原始 instruction。实际读取和分析由注册的图片、音频、视频工具完成。

可选内置工具包括：

- `GoogleSearchAction`
- `TavilySearchAction`
- `ExtractUrlContentAction`
- `ImageAnalysisAction`
- `ParseAudioAction`
- `VideoAnalysisAction`
- `ExecuteCodeAction`

这些工具不会自动注册，应由调用方明确选择。特别是原 `ExecuteCodeAction` 运行的是本地 subprocess，并不是真正的容器沙箱，不应在不可信请求中启用。

## 工具限制

MainAgent 在委派任务时可以指定：

```json
{"tools": ["MyTool"]}
```

系统除了过滤 SubAgent prompt，还会在 clone 出来的 Environment 中限制可执行工具。SubAgent 即使输出未授权工具名，也不会被执行。

以下名称为框架保留名称，不能注册成普通工具：

- `finish`
- `delegate_task`
- `complete`

## 通用 Agent 评测

通用任务不一定具有唯一标准答案，因此 IFlow Agent 将评测层与 Agent 执行层分离：

```text
Evaluation Suite
  -> EvaluationRunner
    -> GeneralAgent.run()
      -> GeneralAgentResult（答案、轨迹、token、成本、耗时）
    -> ScorerRegistry（答案评分 + 工具轨迹评分）
  -> EvaluationReport（JSON / JSONL / CSV）
```

`GeneralAgentResult.status == "done"` 只表示 MainAgent 调用了 `complete`，不代表答案正确。真正的正确性由评测用例中的 scorer 决定。评测代码位于 `iflow_agent2/evaluation/`，不会侵入 MainAgent、SubAgent 或工具执行逻辑。

### 用例格式

评测套件支持 JSONL、JSON 和 YAML。推荐使用一行一个任务的 JSONL：

```json
{
  "id": "video-001",
  "category": "video",
  "instruction": "判断视频画面的主要颜色，只回答颜色。",
  "attachments": [{"type": "video", "path": "fixtures/blue.mp4"}],
  "expected": "蓝色",
  "scorers": [
    {"type": "contains", "weight": 0.8},
    {
      "type": "tool_usage",
      "weight": 0.2,
      "required_tools": ["VideoAnalysisAction"],
      "forbidden_tools": ["ExecuteCodeAction"]
    }
  ],
  "allowed_tools": ["VideoAnalysisAction"],
  "pass_threshold": 0.8,
  "timeout_seconds": 300,
  "tags": ["multimodal", "smoke"]
}
```

相对附件路径以评测套件所在目录为基准解析。`allowed_tools` 会在 CLI 创建 Agent 时限制该 case 可见的工具；`expected_tools` 和 `forbidden_tools` 也可作为 `tool_usage` 的简写字段。

项目提供了可直接修改的示例套件：`iflow_agent2/evaluation/suites/smoke.example.jsonl`。

### 评分器

| scorer | 用途 | 主要参数 |
|---|---|---|
| `completion` | 检查 Agent 是否完成且答案非空 | 无 |
| `exact` | 忽略首尾空白和大小写后的精确匹配 | `case_sensitive` |
| `contains` | 检查答案包含一个或多个片段 | `match: all/any` |
| `numeric` | 数值及允许误差 | `tolerance`、`absolute_tolerance`、`relative_tolerance` |
| `json` | JSON 完全匹配或子集匹配 | `mode: exact/subset`、`required_keys` |
| `tool_usage` | 根据真实 SubAgent trace 检查工具调用 | `required_tools`、`forbidden_tools` |
| `semantic` | 按 rubric 进行开放式语义评分 | `rubric`，必须显式配置 Judge |

多个 scorer 按 `weight` 加权得到 case 分数，再与 `pass_threshold` 比较。答案评分和工具评分因此可以同时存在，例如答案正确占 80%，工具路由正确占 20%。

### 命令行运行

使用本地 `model_config.yaml` 中的 MiMo 和 Tavily 配置运行示例套件：

```bash
conda run -n py310 python -m iflow_agent2.evaluation \
  iflow_agent2/evaluation/suites/smoke.example.jsonl \
  --output iflow_agent2/evaluation/results/smoke \
  --main-model mimo-pro \
  --sub-models mimo \
  --repeats 3 \
  --max-concurrency 2
```

默认启用 Tavily、网页提取、图片、音频和视频工具，不启用本地代码执行。只有明确需要时才加入：

```bash
--tools TavilySearchAction,ImageAnalysisAction,ExecuteCodeAction
```

包含 `semantic` scorer 时必须指定 Judge 模型，否则该 scorer 会明确返回 0 分并说明未配置：

```bash
--judge-model mimo-pro
```

Judge 调用会产生额外 token，目前不会计入被测 Agent 的 `total_cost`，应作为独立评测成本管理。使用与被测 Agent 相同的模型做 Judge 可能产生自评偏差，正式评测应结合另一模型或人工抽检。

CLI 会生成：

- `summary.json`：总分、通过率、Agent 完成率、分类指标、各 scorer 指标、延迟、token、成本和重复运行波动。
- `results.jsonl`：每次运行的答案、评分原因和完整 Agent 轨迹。
- `results.csv`：便于筛选、对比和制作报表的扁平结果。

命令在所有 run 都通过时返回退出码 0，否则返回 1，适合接入 CI。

### Python 接口

自定义工具或 Agent 构造逻辑时直接使用 Python API：

```python
import asyncio

from iflow_agent2 import GeneralAgent
from iflow_agent2.evaluation import EvaluationRunner, load_cases
from iflow_agent2.tools import ImageAnalysisAction, TavilySearchAction


cases = load_cases("eval_cases.jsonl")


def create_agent(case):
    available = {
        "TavilySearchAction": TavilySearchAction,
        "ImageAnalysisAction": ImageAnalysisAction,
    }
    names = case.allowed_tools or list(available)
    return GeneralAgent.from_model_names(
        main_model="mimo-pro",
        sub_models=["mimo"],
        tools=[available[name]() for name in names],
    )


runner = EvaluationRunner(
    create_agent,
    max_concurrency=2,
    suite_name="product-regression",
)
report = asyncio.run(runner.run(cases, repeats=3))
report.save("evaluation-results/product-regression")
print(report.summary())
```

每次 repeat 都通过 `agent_factory` 创建新的 Agent，避免不同任务共享 MainAgent 上下文或 token 计数。重复运行的分数标准差会写入 `repeat_score_stddev`，用于观察模型输出稳定性。

### 建议测试分层

1. 每次提交运行离线单元测试，验证工具协议、环境 clone、评分器和报告。
2. 每日运行少量 live smoke case，检测 MiMo、Tavily、FFmpeg 等外部能力。
3. 发版前运行完整回归套件，每题至少重复 3 次，并按 text/web/image/audio/video 分类查看。
4. 开放式任务使用语义 Judge，同时抽样人工复核。

当前 MiMo 的价格没有配置在 `ModelPricing` 中，因此报告中的 token 数准确，但 `total_cost` 可能为 0。只有配置了模型单价后，成本指标才可用于正式比较。

## 测试

```bash
conda run -n py310 env PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest iflow_agent2/tests -q -p no:cacheprovider
```

普通测试完全离线，不调用真实 API。真实接口测试需显式开启：

```bash
IFLOW_RUN_LIVE_TESTS=1 conda run -n py310 python -m pytest \
  iflow_agent2/tests/test_mimo.py iflow_agent2/tests/test_websearh.py -q
```
