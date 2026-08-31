# Feedback-RDP Agent：基于反馈驱动的持续命名实体识别项目说明与实施计划

## 1. 项目定位

本项目以原始 `INER_RDP-main` 为代码基线，目标是构建一个以 RDP 为主要学习方法、以本地 Qwen-1.5B 为辅助分析器的持续命名实体识别系统。

项目不让大模型承担实体识别、逐 token 伪标签或最终训练决策。RDP 仍然负责模型训练，Python 监控器负责计算训练状态，Python 控制器负责执行经过约束的动作，Qwen 只负责低频的语义分析、异常解释和任务反思。

推荐的方法名称：

```text
Feedback-driven RDP Agent
```

中文可称为：

```text
反馈驱动的持续命名实体识别 Agent
```

## 2. 要解决的问题

在 class-incremental NER 中，模型按任务逐步学习新实体类型：

```text
任务 0：学习初始实体
任务 1：加入新实体
任务 2：继续加入新实体
...
```

每次学习新实体时，旧实体可能发生：

- old-class F1 下降；
- 新旧类别边界混淆；
- encoder 表示空间漂移；
- prototype 与当前特征空间不匹配；
- teacher 伪标签不可靠；
- 新类监督过强、旧类监督不足。

本项目不预先假设“语义相似度等于遗忘风险”，而是通过训练反馈判断当前真正发生了什么，再针对性地调整 RDP 的旧类保护机制。

## 3. 总体思想

系统闭环如下：

```text
RDP 训练
    ↓
Python Monitor 收集状态
    ↓
判断旧类遗忘、表示漂移、teacher 混淆、伪标签不确定性
    ↓
必要时调用 Qwen-1.5B 辅助解释和提出有限候选动作
    ↓
Python Controller 校验动作、限幅、执行和冷却
    ↓
继续 RDP 训练
    ↓
任务结束评估 old/new/seen F1
    ↓
保存 Reflection Memory
    ↓
下一任务使用历史反馈
```

核心分工：

```text
RDP：主要学习器
Monitor：测量训练状态
Python Controller：确定性决策和执行
Qwen-1.5B：辅助解释、候选建议和总结
Reflection：长期记忆
```

原则是：

```text
Qwen 可以建议，但不能裁决。
```

## 4. 原始 INER-RDP 基线

原始项目的主要流程是：

```text
读取 YAML 配置
    ↓
加载 NER 数据和 BERT tokenizer
    ↓
第一个任务创建 BertTagger
    ↓
后续任务复制 refer_model 作为 teacher
    ↓
使用 SplitCosineLinear 增加新类别分类器
    ↓
当前任务之外的实体在训练标签中映射为 O
    ↓
teacher + prototype 生成旧类伪标签
    ↓
CE + soft-label distillation + logits KD
    ↓
验证和累计测试
```

原始 RDP 的关键能力：

1. teacher 模型保留旧类别输出；
2. prototype 保存旧类别表示中心；
3. teacher 和 prototype 共同生成旧类伪标签；
4. soft-label 和 logits distillation 约束旧知识；
5. 分类器逐任务扩展；
6. 通过累计 dev/test 观察持续学习性能。

新版不替换这些机制，而是在它们外面增加反馈控制层。

## 5. Agent 的职责边界

### 5.1 Qwen-1.5B 可以做的事情

#### 任务级语义分析

输入当前新旧实体类型、定义、标注规则和统计信息，输出有限标签：

```json
{
  "reason_tags": ["boundary_overlap", "context_overlap"],
  "focus_old_types": ["location"],
  "confidence": "medium"
}
```

#### 异常原因解释

输入结构化指标：

```text
old F1 drop = 3.1
feature drift = 0.28
teacher confusion = 0.32
pseudo confidence = 0.51
```

输出有限原因标签：

```text
feature_drift
teacher_confusion
pseudo_uncertainty
class_imbalance
```

#### 任务结束反思

输入动作、动作前后指标和最终结果，输出：

```json
{
  "summary_tags": ["distillation_helped"],
  "next_task_focus": ["monitor_location_drift"]
}
```

### 5.2 Qwen 不负责的事情

Qwen 不应该：

- 直接执行模型 forward；
- 逐 token 生成训练伪标签；
- 直接输出任意 loss 权重；
- 直接修改学习率；
- 每个 batch 被调用；
- 直接决定 checkpoint；
- 读取当前任务 test 结果后再改变当前任务训练。

数值计算、阈值、动作合法性和最终执行全部由 Python 完成。

## 6. 系统架构

建议新增模块：

```text
src/feedback_agent/
    __init__.py
    state.py
    monitor.py
    policy.py
    qwen_advisor.py
    reflection.py
    storage.py
```

### `state.py`

定义结构化状态：

```python
TrainingState
OldClassState
ActionRequest
ActionRecord
TaskReflection
```

每个状态必须包含：

```text
task_id
step
entity_type
metric value
coverage
timestamp/observation index
```

### `monitor.py`

只负责计算和记录，不负责改变训练：

```python
observe_batch(...)
measure_old_class_f1(...)
measure_teacher_confusion(...)
measure_pseudo_confidence(...)
measure_feature_drift(...)
measure_gradient_conflict(...)
```

### `policy.py`

Python 确定性控制器：

```python
diagnose(state)
validate_action(action, state)
apply_action(action, trainer)
rollback_action(action, trainer)
```

### `qwen_advisor.py`

本地 Qwen-1.5B 接口：

```python
analyze_task(task_context)
explain_anomaly(state)
reflect_task(reflection_context)
```

必须支持：

- temperature 0；
- 固定 prompt version；
- 严格 JSON 解析；
- 输出动作白名单；
- 缓存；
- 失败回退到 Python 策略；
- 每任务调用次数上限。

### `reflection.py`

保存任务结束后的真实结果和动作效果，不直接参与当前任务的即时决策。

## 7. 训练状态

每个旧实体类型维护：

```python
old_class_state = {
    "f1_before": None,
    "f1_current": None,
    "f1_drop": 0.0,
    "feature_drift": 0.0,
    "prototype_similarity": 0.0,
    "teacher_confusion": 0.0,
    "pseudo_confidence": 0.0,
    "pseudo_entropy": 0.0,
    "gradient_conflict": 0.0,
    "coverage": {},
    "active_actions": []
}
```

状态分成三类：

```text
task_start_state：任务开始的基线
online_state：训练中的滚动观测
reflection_state：任务结束后的真实结果
```

不能用当前任务结束后的 test/F1 去修改当前任务已经开始的训练策略。

## 8. 反馈信号

### 8.1 Old-class F1 drop

```text
f1_drop(old) = f1_before(old) - f1_current(old)
```

这是最直接的遗忘反馈，应按类别统计，而不是只看整体 F1。

### 8.2 Feature drift

任务开始时保存 teacher feature/prototype，在训练中计算：

```text
drift(old) = 1 - cosine(
    feature_teacher(old),
    feature_current(old)
)
```

需要固定 probe 来源、样本数量和有效 token 数。

### 8.3 Teacher confusion

对当前任务中的 gold 新类 token 统计 teacher 预测为旧类的比例：

```text
confusion(new, old)
```

样本数不足时标记为低置信度，不直接触发强动作。

### 8.4 Pseudo-label uncertainty

统计：

- teacher 最大概率；
- teacher entropy；
- prototype 修正前后标签不一致率；
- 每个旧类的伪标签覆盖率。

低置信度伪标签应降低监督权重，而不是全部硬改成 `O`。

### 8.5 Gradient conflict

在低频诊断 batch 上计算：

```text
g_new = grad(new CE)
g_old = grad(old KD + anchor)
conflict = max(0, -cos(g_new, g_old))
```

第一版只观测，不做梯度投影。

## 9. Python 策略和动作

第一版动作空间保持很小：

```text
no_action
increase_old_distillation
increase_prototype_anchor
reduce_pseudo_label_weight
increase_observation_frequency
```

第二阶段再考虑：

```text
freeze_last_encoder_layer
reduce_encoder_learning_rate
increase_old_class_sampling
gradient_clip
```

每个动作都必须具备：

```text
trigger condition
target classes
delta
maximum value
duration
cooldown
rollback rule
```

示例：

```json
{
  "action": "increase_old_distillation",
  "targets": ["location"],
  "delta": 0.05,
  "max_weight": 1.5,
  "duration_steps": 300,
  "cooldown_steps": 200
}
```

所有动作必须有限幅度，不允许单个 batch 直接改变全局训练行为。

## 10. 动作归因设计

这是项目最重要的实验约束。

### 单动作原则

一次只执行一个主要动作：

```text
只提高 distillation
只提高 prototype anchor
只降低 pseudo-label weight
```

组合动作必须在单动作有效后测试。

### 动作记录

```json
{
  "task_id": 2,
  "step": 1200,
  "action": "increase_old_distillation",
  "trigger_state": {},
  "metrics_before": {},
  "metrics_after": {},
  "effect_window_steps": 300,
  "accepted": true,
  "rollback": false
}
```

动作效果至少比较：

- 目标旧类 F1；
- 非目标旧类 F1；
- 新类 F1；
- 累计 macro-F1；
- feature drift；
- training loss。

## 11. 阶段实施计划

### P0：冻结并复现 baseline

目标：确认原始 RDP 能稳定运行。

工作：

- 固定代码版本、配置、seed、任务顺序；
- 保存每任务 old/new/seen F1；
- 保存训练样本统计和 checkpoint；
- 不引入 Agent。

验收：相同条件下可复现 baseline，且结果进入 baseline 表格。

### P1：增加 observe-only Monitor

目标：只记录状态，不改变训练。

工作：

- 实现 `TrainingState`；
- 记录 per-class F1、teacher confusion、pseudo confidence、prototype count；
- 任务结束保存 reflection；
- `agent_mode=observe_only`。

验收：开启 Monitor 后 model state、loss、指标与 baseline 一致。

### P2：加入 feature drift

目标：验证 encoder 漂移是否与旧类遗忘相关。

工作：

- 保存任务开始的 teacher/probe 特征；
- 按固定间隔计算 old-class drift；
- 输出 drift 曲线和 F1 曲线；
- 暂不改变 loss。

验收：可以回答 drift 是否提前于 F1 下降出现。

### P3：加入 Python 自适应蒸馏

目标：只根据确定性指标调整 per-class old KD 权重。

工作：

- 增加 per-class distillation weight；
- 使用 EMA 和触发/解除双阈值；
- 单次最大权重变化受限；
- 加入 cooldown 和 rollback。

验收：与 baseline 比较 old-class average F1、average forgetting 和 new-class F1。

### P4：加入软伪标签

目标：降低错误硬伪标签对 RDP 的污染。

工作：

- 使用 teacher/prototype 概率而不是单纯 argmax；
- 按置信度加权旧类监督；
- 低置信度 token 不产生强旧类 CE；
- 保存伪标签覆盖率和 entropy。

验收：旧类 recall 不出现大面积下降，且新类 F1 不明显恶化。

### P5：加入 prototype mean/variance

目标：使旧类别记忆不只保存单一均值。

工作：

- 保存 prototype mean、variance、count；
- 对低 count/空 prototype 禁止强 anchor；
- 记录 prototype drift 和 margin。

验收：prototype anchor 不产生 NaN，且 prototype drift 可复查。

### P6：接入 Qwen-1.5B

目标：让 Qwen 参与任务分析，但不改变核心训练。

工作：

- 任务开始调用一次 `analyze_task()`；
- 任务结束调用一次 `reflect_task()`；
- 异常时最多调用一次 `explain_anomaly()`；
- 固定 JSON schema、缓存和失败回退；
- Qwen 建议只映射到动作白名单。

验收：Qwen 关闭时流程完全可运行；Qwen 输出错误时训练不失败。

### P7：动态动作执行

目标：让 Agent 根据训练反馈执行有限干预。

工作：

- 先只开放 `increase_old_distillation`；
- 再开放 `increase_prototype_anchor`；
- 最后开放 `reduce_pseudo_label_weight`；
- 每次只执行一个动作；
- 保存动作前后效果和回退记录。

验收：动作有效性可以通过独立对照实验验证。

### P8：梯度冲突和更强动作

目标：在已有证据后处理优化层面的冲突。

工作：

- 先 observe-only 计算 gradient conflict；
- 验证 conflict 与 F1 drop 的关系；
- 只有在关系稳定后才尝试 gradient clip/projection；
- 动态学习率和冻结层最后实现。

验收：额外干预不引起训练不稳定，并且在多个 seed 上有效。

## 12. 实验矩阵

所有实验固定数据、任务顺序、seed、epoch、学习率和 checkpoint 规则。

```text
A  Original RDP
B  RDP + observe-only Monitor
C  RDP + Python adaptive distillation
D  RDP + soft pseudo labels
E  RDP + prototype mean/variance
F  RDP + Qwen analysis only
G  RDP + Python actions
H  RDP + Python actions + Qwen suggestions
I  RDP + feature drift intervention
J  RDP + gradient conflict intervention
```

解释：

- `B-A`：监控是否改变 baseline；
- `C-A`：自适应蒸馏是否有效；
- `F-A`：Qwen 只做分析是否影响结果；
- `H-G`：Qwen 建议是否额外帮助；
- `J-I`：优化层干预是否值得使用。

## 13. 评价指标

### 模型性能

- cumulative micro-F1；
- cumulative macro-F1；
- new-class F1；
- old-class average F1；
- seen-class average F1；
- average forgetting；
- per-class F1 drop；
- old-to-new/new-to-old confusion。

### Agent 性能

- 触发次数；
- 动作接受/拒绝/回退次数；
- 动作后目标类 F1 变化；
- Qwen 建议与 Python 最终动作一致率；
- Qwen 输出解析失败率；
- 监控开销和 Qwen 调用次数；
- 风险/状态信号与实际遗忘的相关性。

风险预测和防遗忘效果必须分别报告，不能用一个指标代替另一个指标。

## 14. 关键失败模式和防护

### 动作归因困难

解决：单动作、独立对照、动作前后窗口和完整日志。

### 阈值脆弱

解决：EMA、相对阈值、双阈值、冷却期和最大变化限制。

### Qwen 输出不稳定

解决：固定 prompt、temperature 0、JSON schema、动作白名单和 Python 回退。

### Reflection 样本太少

解决：前几个任务只记录，不自动校准；多个 seed/任务顺序后再使用。

### Dev 不代表真实遗忘

解决：分别统计 old/new/seen，使用固定 probe，测试集只做最终评估。

### 动态策略震荡

解决：连续多次触发、动作持续窗口、cooldown、滞回阈值和 rollback。

## 15. 最终完成标准

```text
[ ] 原始 RDP baseline 可复现
[ ] observe-only 不改变模型更新
[ ] 状态按 old class 记录并带 coverage
[ ] feature drift 可复查
[ ] Python 自适应蒸馏可独立消融
[ ] 软伪标签不会导致旧类召回崩溃
[ ] prototype 统计支持空值和低 count
[ ] Qwen 失败不会中断训练
[ ] Qwen 不直接决定数值权重
[ ] 每个动作可限幅、冷却和回退
[ ] Reflection 不泄漏当前任务测试结果
[ ] 多 seed、多任务顺序下结果可复查
[ ] 能区分 RDP 改进效果和 Qwen Agent 效果
```

## 16. 最终方法表述

推荐表述：

> We propose a feedback-driven continual NER framework built on RDP. A Python training monitor tracks class-wise forgetting, representation drift, teacher confusion, and pseudo-label uncertainty. A lightweight local Qwen-1.5B model provides task-level semantic analysis and reflection, while a deterministic controller applies bounded interventions to RDP distillation, prototype anchoring, and pseudo-label weighting.

中文表述：

> 本项目提出一种基于 RDP 的反馈驱动持续命名实体识别框架。系统通过 Python 监控器跟踪类别级遗忘、表示漂移、教师模型混淆和伪标签不确定性，由本地 Qwen-1.5B 提供任务级语义分析与反思，再由确定性控制器对 RDP 的蒸馏、原型约束和伪标签权重实施有限幅度的自适应调整。

该方法的核心不是让大模型替代 NER 训练，而是：

```text
RDP 学习
→ 监控反馈
→ 有限干预
→ 任务反思
→ 下一任务改进
```
