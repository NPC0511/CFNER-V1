# 计划修订与可行性评估

本文件补充 `PROJECT_PLAN_FEEDBACK_RDP_AGENT.md`。

## 因果归因

- 为每个动作设置固定效果窗口，记录动作前、短期、长期指标。
- 使用相同 seed、数据顺序和训练预算的 action/no-action 配对实验；跨 seed 只用于稳定性验证。
- 同时观察目标旧类、非目标旧类、新类和累计 macro-F1，保存动作触发、持续、冷却和回退记录。
- 论文中将其描述为受约束在线干预证据，而非严格因果证明。

## 阈值敏感

- 第一版在开发任务上调参，后续任务和 seed 验证；不使用最终 test 调参。
- 保存 threshold version 和修改记录。
- 使用 EMA/历史分位数的相对阈值，配合最小样本数、连续 K 次触发、触发/解除双阈值和 cooldown。
- 第一版不使用数据集专属阈值，但使用归一化指标降低跨数据集差异。

## 可行性

高可行：observe-only Monitor、per-class F1/confusion/pseudo/prototype 统计、JSON/CSV 记录、Python 自适应蒸馏、低频 Qwen 分析与反思。

中等可行：feature drift、prototype mean/variance、软伪标签、动作回退和历史校准。

后置高风险：每 batch Qwen、Qwen 直接输出数值权重、同时执行多个动作、全量梯度冲突、动态学习率和梯度投影。

## 潜在问题

1. 监控信号可能与真实遗忘相关性很弱；无效信号应停留在 observe-only。
2. Dev/probe 可能不代表旧类真实分布，需报告覆盖量和限制。
3. 旧类保护可能损害新类，动作必须检查 new-class F1 和累计 macro-F1。
4. Reflection 样本少时不得自动校准。
5. Qwen 与 Python 策略结果接近时，Qwen 的价值主要是解释和反思，不应声称其带来性能提升。

## 修订后的执行顺序

```text
P0 baseline
P1 observe-only
P2 状态统计
P3 paired action/no-action 框架
P4 Python per-class adaptive distillation
P5 软伪标签
P6 prototype mean/variance
P7 Qwen 分析与反思（默认不执行建议）
P8 有限动作与回退
P9 feature drift 干预
P10 gradient conflict/动态优化
```
