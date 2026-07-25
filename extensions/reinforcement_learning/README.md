# Reinforcement Learning Extension

该目录提供一个可选的 GRPO 训练扩展，用于把符合 IFlow `MainAgent` 决策协议的开源模型训练成任务协调模型。它与 Agent 运行时、工具系统和评测系统隔离，导入 `iflow_agent2` 或运行普通任务时不会加载训练框架，也不会自动开始训练。

## 包含内容

| 路径 | 作用 |
|---|---|
| `grpo/reward.py` | verl 自定义奖励回调，导出 `compute_score` |
| `grpo/train_grpo.sh` | 通用 GRPO 启动模板，需要显式执行 |
| `grpo/config/deepspeed_zero3.json` | 供需要 DeepSpeed 的外部训练器选用的 ZeRO-3 配置 |

训练脚本默认使用 verl 的 FSDP2 + vLLM 链路，因此不会自动读取 ZeRO-3 配置。两种配置并列保留，便于后续接入不同训练器。

## 在完整流程中的位置

该扩展只处理“评测已经确认属于 MainAgent 策略问题”之后的训练阶段：

```text
通用任务执行
  -> 固定套件评测
    -> 失败归因
      -> 审核并构造 Train 数据
        -> GRPO 训练
          -> 部署候选模型
            -> 使用冻结 Dev/Test 套件回归评测
```

工具错误、API 故障、缺少多模态能力等问题不应通过奖励训练修复。评测用的冻结 Test 数据也不能放入 `TRAIN_FILE`。端到端说明见 [完整工作流](../../docs/workflow.md)。

## 决策协议

训练样本中的模型输出必须是以下两种 JSON 之一：

```json
{
  "action": "delegate_task",
  "params": {
    "tasks": [
      {
        "task_instruction": "独立、可执行的子任务",
        "model": "worker-model",
        "tools": ["RegisteredTool"]
      }
    ]
  }
}
```

```json
{
  "action": "complete",
  "params": {
    "answer": "最终答案"
  }
}
```

奖励由四个归一化维度组成：格式正确性 10%、动作有效性 10%、工具与子任务合理性 20%、决策质量 60%。前两项由确定性校验计算，后两项由 OpenAI 兼容的 Judge 模型计算。

## 数据契约

`TRAIN_FILE` 和 `VAL_FILE` 应是 verl 可读取的 Parquet 文件。每条样本除了 verl 版本要求的 `prompt`、`data_source` 和 `reward_model` 字段外，可以在 `extra_info` 中提供：

```json
{
  "question": "用户任务",
  "answer": "预期答案或成功标准",
  "subtask_history": "当前步骤之前的执行历史",
  "reference_decision": {
    "action": "delegate_task",
    "params": {"tasks": []}
  }
}
```

`reference_decision` 是可选字段，奖励函数不会要求候选输出复制参考路径。具体 Parquet 基础字段应以你安装的 verl 版本为准。

## Judge 配置

默认从项目的 `config/model_config.yaml` 读取 `mimo-pro`。可以使用以下变量覆盖：

| 变量 | 作用 |
|---|---|
| `IFLOW_RL_JUDGE_MODEL` | `model_config.yaml` 中的模型别名，默认 `mimo-pro` |
| `IFLOW_RL_JUDGE_REQUEST_MODEL` | 实际发送给服务端的模型名 |
| `IFLOW_RL_JUDGE_BASE_URL` | OpenAI 兼容服务地址 |
| `IFLOW_RL_JUDGE_API_KEY` | Judge API 密钥 |
| `IFLOW_RL_JUDGE_TIMEOUT` | 单次请求超时秒数，默认 60 |
| `IFLOW_RL_JUDGE_MAX_RETRIES` | 最大重试次数，默认 3 |

## 可选训练入口

训练框架和 GPU 依赖不会加入核心 `requirements.txt`。需要实验时，先单独准备与硬件匹配的 PyTorch、vLLM、Ray 和 verl 环境，再显式执行：

```bash
VERL_DIR=/path/to/verl \
BASE_MODEL_PATH=/path/to/base-model \
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/validation.parquet \
bash extensions/reinforcement_learning/grpo/train_grpo.sh
```

常用超参数可以通过 `N_GPUS_PER_NODE`、`TRAIN_BATCH_SIZE`、`ROLLOUT_GROUP_SIZE`、`LEARNING_RATE`、`TOTAL_EPOCHS` 和 `TENSOR_PARALLEL_SIZE` 覆盖。额外的 Hydra 参数可以直接追加到命令末尾。

当前扩展只提供接口、奖励和启动模板，不包含训练数据、模型权重、训练结果，也不承诺任何未经实际实验验证的指标。
