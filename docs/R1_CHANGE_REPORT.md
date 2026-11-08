# R1 代码修改报告

本文保留首次实现记录。随后审查修复已加入全局/固定池化对照、训练先验、随机流隔离、初始化校验和新校准绑定；当前代码与升级步骤以 [R1 审查修复报告](R1_REVIEW_FIX_REPORT.md) 为准。

日期：2026-11-08

目录：`visualnav/`

交付状态：模型、训练、验证、恢复配置及消融入口已在代码中接通；本轮只在 Mac 编辑和静态审阅，未运行单元测试、导入/语法检查、训练或仿真。

## 1. 最终实现的 R1

依据最终确认，R1 已从“仿真几何监督的占据瓶颈”改为 **轨迹引导的空间瓶颈**：只使用现有 RGB 和动作轨迹，不新增深度、地图、碰撞或占据标签。

保留 6 个 64 维空间 token。辅助输出为 6×4 个区域的未来 waypoint 访问分布，不再是 free/occupied/unknown。融合条件仍为 256 维，由原距离头、R3 关系头与扩散网络共用。R3 的已有监督协议及选择器保持原设计。

这里的分布表示示范轨迹经过哪些区域，**不表示全部可行路线或障碍概率**。没有被示范访问的区域不能当作不可通行。这一变化也意味着论文中的几何 IoU、unknown→free 和占据图解释需要替换。

## 2. 修改文件与作用

下列代码路径相对 `visualnav/train/`。

| 文件 | 类型 | 作用 |
|---|---|---|
| `vint_train/models/nomad/spatial_bottleneck.py` | 新增 | 空间特征投影、位置编码、6 个查询 token、目标条件访问分布头、零初始化残差读取 |
| `vint_train/models/nomad/nomad_vint.py` | 修改 | 从 frame-major 编码结果读取当前帧池化前特征；提供条件融合、辅助输出与独立 token 提取 |
| `vint_train/models/nomad/nomad.py` | 修改 | 转发可选 `return_r1`，新增 `spatial_tokens` 调用分支 |
| `vint_train/models/factory.py` | 修改 | R1 构建、架构/分段元数据校验；隔离新增模块的初始化 RNG |
| `vint_train/data/trajectory_regions.py` | 新增 | 从训练轨迹拟合距离边界；检查标签语义、训练签名、动作资格和采样设置 |
| `vint_train/data/prepare_r1_data.py` | 新增 | 生成 R1 统计 JSON 的独立命令 |
| `vint_train/training/trajectory_visitation.py` | 新增 | waypoint 转访问直方图、软目标 CE、有效样本处理、分布指标 |
| `vint_train/training/goal_relation_train.py` | 修改 | 同一次动作流前向计算 R1 输出；将辅助项接入累计窗口分母和损失 |
| `vint_train/training/simulation_train.py` | 修改 | spatial 优化器组、AMP/EMA 训练接入、三种 goal-mask 验证、梯度诊断、统计和恢复配置记录 |
| `vint_train/inference/rgb_policy.py` | 修改 | `predict_rgb_visitation` 可选诊断；拒绝把未受监督的访问头当作有效预测 |
| `train.py` | 修改 | 新增独立 `--r1-mode`，旧训练入口明确拒绝不支持的 R1 配置 |
| `config/nomad_r1_r3_habitat.yaml` | 新增 | R1/R3 共同配置及初始训练参数 |
| `tests/test_r1_core.py` | 新增 | 6 项回归检查，留到 Ubuntu 执行，本轮未运行 |

文档方面更新了 `train/README.md`、`docs/R3_HABITAT.md`，重写了 [修改计划与使用说明](R1_IMPLEMENTATION_PLAN.md)。研究目录的 `12_r1_r3_paper_figures.md` 已增加旧 R1 图稿过期提示；本报告为新增交付文档。

## 3. 数据、模型与损失如何接起来

**数据侧：**复用原 ViNT 七项动作 batch，不增加数据加载流。将累计局部 waypoint 乘回 `action_scale_m` 得米制位置，计算极坐标区域直方图。只对原 action_mask 有效且至少含一个非原点 waypoint 的样本监督，负目标和全原点样本不贡献该项损失。

**分段侧：**6 个扇区覆盖整圈，正前方为第 0 扇区中心；4 个距离段的内部边界由 train 半径分位数拟合。默认最多采样 200000 点、固定拟合种子，记录拟合数量与签名；验证和测试不拟合边界。最后一段无有限上界。这里不需要新增相机内外参、深度格式或机器人足迹配置。

**模型侧：**当前帧空间特征经投影与查询注意力生成 token，token 提取不接目标。辅助头接收 token 和原 goal-mask 条件，24 格统一 softmax。残差读取另将 token 融入条件；真实未来轨迹只进入损失 target，不进入网络输入。

**损失侧：**新增 `lambda_visitation × CE`，默认权重 0.1。先对整个梯度累积窗口统计有效样本数，再归一化各 microbatch 的损失和；尾窗使用相同规则。R3 关系流不重复计算访问监督，避免组合模型额外增加 R1 标签预算。

**训练基础设施：**沿用随机初始化与全参数训练，接入现有优化器更新、AMP、EMA 和 epoch checkpoint。R1 配置、分段定义、统计内容指纹被写入训练记录，并参与严格恢复检查。仍使用 v2 checkpoint，不做旧 R3 权重的新增模块 warm start。

## 4. 消融与推理行为

| `--r1-mode` | 空间反馈给策略 | 访问分布辅助损失 |
|---|---|---|
| `off` | 无 | 无 |
| `tokens_only` | 有 | 无 |
| `auxiliary` | 无 | 有 |
| `bottleneck` | 有 | 有 |

`--ablation baseline/r_only/r_d` 仍控制 R3，两个开关组合得到 baseline、R1、R3、R1+R3。auxiliary 保留同一空间 token 提取器，用于隔离反馈作用，并非全局池化 MLP 基线。

正常导航继续使用原 RGB 评分和扩散采样接口，不额外调用访问分布头，也不增加轨迹重排序或停车规则。R1+R3 需要按新 checkpoint 重新校准 R3，不能复用旧 R3 权重的校准产物。R1 关闭时，工厂不增加 R1 参数/元数据字段；实际旧 checkpoint 兼容性尚未做运行验收。

诊断接口返回 `spatial_tokens[B,6,64]`、访问 logits/probability `[B,6,4]` 和共享条件。tokens_only 的访问头未训练，指标写为 null，诊断接口拒绝输出受监督预测。消融中不受任务梯度更新的参数另行记录在 provenance。

## 5. 记录与验证情况

新增日志包括有效样本/waypoint 数、CE、KL、L1、预测在示范访问格上的概率质量、最高概率格是否被访问、各格平均分布与辅助梯度诊断。非有限预测数量单列；指标只在有限预测上统计并报告分母，训练 loss 仍由原 AMP/梯度机制处理。

| 项目 | 本轮状态 |
|---|---|
| 数据流、当前帧索引、标签/输入隔离的静态阅读 | 已完成 |
| 模型工厂、R3 校准/推理调用与恢复配置的静态阅读 | 已完成 |
| `git diff --check` | 通过；仅为差异格式检查 |
| 新增 6 项 R1 单元回归检查 | 已编写，未执行 |
| Python 导入、语法编译、模型实际前向/反向 | 未执行 |
| CUDA AMP、activation checkpoint、恢复等价性 | 未执行 |
| Habitat 闭环、训练效果、4070 Ti Super 显存与速度 | 未执行 |

6 项检查针对米制缩放与边界、无效 logits 排除、不均匀 microbatch 梯度、未训练头的指标、初始恒等融合与辅助梯度、train-only 拟合及过期统计拒绝。它们不能代替完整训练与闭环验收。

## 6. Ubuntu 上的第一步

先修改 `config/nomad_r1_r3_habitat.yaml` 的 manifest、action_statistics、r1_statistics 路径。原动作统计仍匹配数据时可以直接复用。以下从 `visualnav/train` 执行，本轮没有执行这些命令：

```bash
python -m vint_train.data.prepare_r1_data --config config/nomad_r1_r3_habitat.yaml

# 先验证 R1 单独训练入口
python train.py -c config/nomad_r1_r3_habitat.yaml --ablation baseline --r1-mode bottleneck

# 组合入口，另一次实验使用
python train.py -c config/nomad_r1_r3_habitat.yaml --ablation r_d --r1-mode bottleneck
```

需要运行新增回归检查时：

```bash
python -m unittest discover -s tests -p 'test_r1_*.py' -v
```

完整四组命令、损失公式、坐标约定和恢复方式见 [实现说明](R1_IMPLEMENTATION_PLAN.md)。默认 batch=4、累积=4、辅助权重=0.1 均是待验证初值，没有声称已满足实际显存或时间预算。后续需先通过 Ubuntu 运行验证，再以主实验和机制对照判断 R1 的实际收益。
