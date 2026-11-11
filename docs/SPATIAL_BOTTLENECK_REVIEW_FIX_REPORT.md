# SPATIAL_BOTTLENECK 审查修复报告

日期：2026-11-08。修改范围为 `visualnav/`。按本轮确认，**继续从零训练，不改为预训练微调**。

本轮完成代码修改、静态审阅和文本差异检查；没有运行测试、导入/语法编译、训练或仿真。下面的“已实现”指代码已接通，不代表实验效果已经验证。

## 1. 审查问题与处理

| 审查项 | 代码处理 | 仍需实验回答的问题 |
|---|---|---|
| waypoint 坐标和尺度 | 保留正确的位置恢复路径；训练标签和先验共用米制直方图函数；拒绝非法尺度、非零数据集索引；补坐标链路检查 | Ubuntu 上执行真实数据与边界检查 |
| 统计先验可能解释收益 | 增加训练集常量先验、原全局条件辅助头、固定空间池化对照，以及 Token 置换诊断 | 模型是否依赖图像相关的 Token，导航是否有额外收益 |
| 辅助梯度和策略收益混淆 | 分支记录 NoMaD、访问 CE、GOAL_RELATION 的梯度连通性、范数和余弦；保持 auxiliary 对共享编码器的梯度 | CE 改善与闭环 SR/SPL/碰撞改善是否同时成立 |
| 初始化一致性 | 配置明确 scratch；每次启动构建同种子 CPU baseline 参考，校验所有共同初始张量哈希；记录 SPATIAL_BOTTLENECK/GOAL_RELATION 初始哈希 | 多种子结果与方差 |
| SPATIAL_BOTTLENECK/GOAL_RELATION 消融公平性 | 噪声、时间步、mask 和动作/GOAL_RELATION 随机前向分别隔离；记录消费样本、成功更新样本、额外 GOAL_RELATION 标签及时间/参数量；增加运行比较工具 | 实际训练预算、AMP 跳过次数和计算成本是否可比 |
| GOAL_RELATION 校准 | 新校准文件同时绑定完整权重和模型配置，区分同形状参数下不同 SPATIAL_BOTTLENECK 模式；增加拒绝旧校准的检查 | 对每个 GOAL_RELATION / SPATIAL_BOTTLENECK+GOAL_RELATION checkpoint 在固定校准划分独立拟合 |

SPATIAL_BOTTLENECK 的访问头仍读取 `(Z, c)`。没有擅自删除目标条件或切断共享编码器梯度；“绕过 Token”的可能性通过对照与诊断检验，不能仅靠加一个损失就宣称已消除。

## 2. 坐标与标签协议

输入仍是 `ViNT_Dataset` 的七项 batch。`actions` 是归一化的**机器人局部 waypoint 位置**，乘 `action_scale_m` 后直接划分区域，没有 `cumsum`。扩散分支另做位置差分；扩散输出须经 `decode_actions` 恢复位置后才可解释为轨迹。

`metric_visitation_histograms()` 统一训练目标和先验拟合的分区规则。当前训练入口是单个 Habitat manifest，`metric_waypoint_spacing × waypoint_spacing = action_scale_m`。未增加异构多数据集支持；错误数据集索引现在直接报错。

新增检查使用真实 `_compute_actions()`：非零 yaw、非单位尺度、stride=2、多个方向及距离段，并检查扩散编解码往返。另覆盖 NaN 无效样本、非法尺度及不支持的数据集索引。

## 3. 对照模式与正确比较方式

所有模式使用同一套距离边界、访问目标和动作训练流。`--ablation` 仍独立控制 GOAL_RELATION。

| `--spatial-bottleneck-mode` | 表征来源 | 反馈给策略条件 | 访问 CE |
|---|---|---|---|
| off | 原 NoMaD | 无 | 无 |
| tokens_only | learned query Token | 有 | 无 |
| auxiliary | learned query Token | 无 | 有 |
| bottleneck | learned query Token | 有 | 有 |
| global_auxiliary | 原始 256 维条件接 MLP | 无 | 有 |
| pooled_auxiliary | 当前帧空间特征固定池化 | 无 | 有 |
| pooled_bottleneck | 当前帧空间特征固定池化 | 有 | 有 |

固定池化对照在默认维度下形成 2×3 个图像网格槽位，共 6×64 维。它保留空间投影、位置编码、FFN、访问头和条件读取接口，用固定池化替换 learned query 提取注意力。反馈模式仍使用共同的条件注意力读取器；这里移除的是**提取空间表征的可学习查询**。网格槽位不等于机器人极坐标区域。

建议先关闭 GOAL_RELATION 做机制实验：

- `global_auxiliary`、`pooled_auxiliary` 与 `auxiliary` 比较：同样无直接反馈、同样有 CE，检查表征提取方式。
- `pooled_bottleneck` 与 `bottleneck` 比较：同样有反馈、同样有 CE，检查查询提取的贡献。
- `off`、`tokens_only`、`auxiliary`、`bottleneck` 比较：检查反馈与辅助监督。

这些网络参数量不完全相等，不能声称“等参数/等计算”。`provenance.json` 记录总参数、各优化器组参数以及排除已知未使用参数后的数量；训练日志记录耗时和显存。全局辅助头没有空间 Token，诊断接口的 `spatial_tokens` 返回 `None`。

## 4. 常量先验与 Token 置换

`prepare_trajectory_regions` 新生成 **schema 2** 统计文件，同时保存距离边界和 train-only 常量先验。旧 schema 1 需要重跑该命令，不需要重新采集数据。

先验是训练有效动作样本直方图的期望均值：每个观测先归一化自己的访问直方图，再按该观测随机目标成为有效动作样本的概率加权。这样不会把含更多有效 waypoint 的样本额外放大，也考虑了轨迹尾部可采样目标数不同。固定混入 `1e-4` 的均匀概率质量，防止未见区域产生 `log(0)`；该值在看验证结果前固定。训练签名、拟合设置、先验与边界一起写入指纹。

验证的 `visitation.train_prior` 与模型指标使用相同标签和有效样本规则。train/validation/test 不混合拟合。

验证还在一个 batch 的有效样本间循环交换**整组 Token**，固定原条件 c 与标签；原模型和置换模型的 CE 使用相同子集。记录 `cross_entropy_increase`，反馈模式还记录条件向量变化。有效样本不足 2 时跳过并计数，不伪造零差值。

置换几乎不影响 CE，提示模型可能不依赖样本对应的 Token；出现明显变化，只能说明依赖这些 Token，仍不能证明几何理解或闭环收益。它也不能区分 Token 中的空间结构与其他图像信息。

## 5. 梯度、初始化和训练公平性


auxiliary 仍通过共享编码器影响策略。访问 CE 使用融合前 c，不直接监督残差读取层；零初始化残差也会使第一个策略反向中的部分 Token 梯度为零。这两种现象不应误判为接线失败。

初始化校验不加载任何预训练权重、不冻结参数、不改变 baseline 公共参数的随机序列。`provenance.json` 保存共同参数与 CPU 参考的哈希，以及新增模块的初始哈希。

新随机协议 `epoch_streams_and_per_microbatch_branch_rng_v2`：

- diffusion 噪声和时间步各用独立 generator，按 seed/epoch 重建。
- 动作与 GOAL_RELATION 前向的 dropout/stochastic depth 按 seed/epoch/microbatch 分别隔离。
- 验证 mask、噪声和采样随机数独立；验证不消耗训练随机状态。
- 仍只支持 epoch 边界恢复，checkpoint 的训练规格记录新协议。旧协议 checkpoint 可以按原方式推理，**不能被当作新协议的完全一致断点继续训练**。

预算记录同时包含尝试消费的 action/GOAL_RELATION 样本与成功 optimizer update 覆盖的样本，保留 AMP skips。不会为了凑齐更新次数而悄悄重复数据。比较工具将不同成功更新预算明确标为不一致，研究报告需要说明或重新组织实验。

## 6. GOAL_RELATION 校准约束

新 `calibration.json` 为 schema 2，绑定全模型权重 SHA、raw/EMA 来源、model_config SHA、标签定义与选择器配置。后者涵盖 SPATIAL_BOTTLENECK 的模式和表征类型，防止相同参数形状但不同前向语义的校准混用。离线评测还检查校准数据划分指纹。

旧 schema 1 仅继续用于不含 SPATIAL_BOTTLENECK 的旧 GOAL_RELATION 模型；任何 SPATIAL_BOTTLENECK+GOAL_RELATION 模型必须用自己的 schema 2 校准。校准仍使用原来的两个独立校准集，不从测试集拟合温度或阈值。

## 7. Ubuntu 执行入口

以下命令从 `visualnav/train/` 执行。本轮没有执行它们。

```bash
# 修改 YAML 中三个数据路径后，重建含先验的新统计文件。
python -m vint_train.data.prepare_trajectory_regions --config config/nomad.yaml

# 新增机制对照，先关闭 GOAL_RELATION。
python train.py -c config/nomad.yaml --ablation baseline --spatial-bottleneck-mode global_auxiliary
python train.py -c config/nomad.yaml --ablation baseline --spatial-bottleneck-mode pooled_auxiliary
python train.py -c config/nomad.yaml --ablation baseline --spatial-bottleneck-mode pooled_bottleneck

# 同一 seed 的完整运行记录对比；实际运行目录替换示例路径。
python -m vint_train.training.compare_ablation_runs \
  /path/to/baseline /path/to/spatial_bottleneck /path/to/goal_relation /path/to/spatial_bottleneck_goal_relation \
  --output /path/to/ablation_audit.json
```

四组主实验继续使用 [实现说明](SPATIAL_BOTTLENECK_IMPLEMENTATION_PLAN.md) 中的命令。GOAL_RELATION 校准继续使用 [GOAL_RELATION 文档](GOAL_RELATION_HABITAT.md) 的独立校准命令。

比较工具核对共同训练规格、数据指纹、模型配置和共同初始哈希；核对同结构新增模块初始哈希、SPATIAL_BOTTLENECK 共同维度/目标/监督权重和 GOAL_RELATION 设置。完整日志才可确认整个训练预算；只有恢复后片段或尚未完成的日志，预算结果为 `null`。不一致时退出码为 1。它不证明闭环效果，也不把额外 GOAL_RELATION 标签或额外计算视为相同成本。

## 8. 主要改动文件与验证边界

新增 `training/experiment_protocol.py`、`training/compare_ablation_runs.py`；更新模型工厂/编码器/空间模块、轨迹统计与目标、训练循环、CLI/YAML、GOAL_RELATION 校准及说明文档。代码路径均位于 `train/vint_train/`。


尚未验证：测试实际通过、CUDA/AMP、activation checkpoint、完整 optimizer/EMA 恢复、显存/耗时、访问预测与闭环导航结果。本轮完成的是代码层面的防错与实验对照条件；科学结论仍须由 Ubuntu 上的实验给出。
