# R3：Habitat 从零联合训练

已完成 R3 单测；此前合成数据的训练、恢复、AMP、校准和推理 smoke 已通过。部署入口现使用 ROS 2 Humble / Python 3.10，环境和启动方法见 [部署说明](../deployment/README.md)。这些检查不代表真实 Habitat 闭环或实机验收通过；模板的 batch、学习率、100 epochs 仍是初始配置，不代表已测得的论文效果。

## 1. 入口与范围

- 配置：`train/config/nomad_r3_habitat.yaml`；`train.py` 的 `habitat_scratch` 分支。
- 数据：`data/simulation.py`、`prepare_sim_data.py`、`goal_relation_dataset.py`；动作流复用 `ViNT_Dataset` 和 FP32 图像缓存。
- 模型：`models/nomad/goal_relation_head.py`；共享 MLP 为 256→128→64，输出 R logit 与 `H * Softplus(z_D)`。
- 训练：`training/goal_relation_train.py`、`simulation_train.py`、`simulation_checkpoint.py`。
- 校准：`training/calibrate_r3.py`；通用推理：`inference/`；ROS 2 接口：`deployment/src/r3_navigate.py`。

整个有效网络随机初始化、参与优化，包括两路 EfficientNet、Transformer、原距离头、扩散 U-Net 和 R3。原实现注册但前向未使用的 EfficientNet `_fc` 和 Transformer 模板 `sa_layer` 为兼容旧权重保留；其名称写入 `provenance.json`，不虚报梯度。R-only 消融关闭 D 损失和 D 排序，其 D 输出层不构成有效任务。验证和校准报告保留 distance_valid_count，但将 distance_MAE_m 设为 null，并用 distance_head_trained=false 明确标记。

R3 关闭、且未配置仿真尺度时，原模型结构、原训练入口和旧推理权重路径保留。新实验不调用旧权重迁移。`resume_from` / `--resume` 只接受同配置的完整 v2 checkpoint。

本轮没有采集器、路径标签生成器、CARLA 适配或 Habitat 控制驱动。现有 Habitat 实验环境需要按以下协议导出数据，并调用通用 RGB 推理/选择接口。ROS 2 节点提供消息接入，不要求实机实验。

## 2. 数据准备

复制 `train/examples/habitat_r3/manifest.template.json` 为自己的 `manifest.json`。其中 `null` 故意不可训练：必须填写实际 `action_scale_m`、`action_time_step_s`、H 和 NavMesh 投影误差容限。示例关系文件只是格式示意，里面的路径和标签不是可用于实验的数据。

```text
habitat/
  manifest.json
  relations/train.jsonl
  relations/validation.jsonl
  relations/calibration_temperature.jsonl
  relations/calibration_threshold.jsonl
  relations/test.jsonl
  rgb_trajectories/
    train_scene/episode_0001/
      0.jpg, 1.jpg, ...
      traj_data.pkl
    ...
```

`data_root` 相对 manifest 所在目录；关系 JSONL 路径也相对 manifest；每行 RGB 路径相对 `data_root`。所有 RGB 必须是三通道。动作轨迹的 `traj_data.pkl` 为可信本地 pickle，包含 `position[T,2]`（米）、`yaw[T]`（弧度），逐帧编号从 0 连续增长。只读取需要的 position/yaw；其他字段可保留。

坐标必须已从 Habitat 转为 `nomad_xy_ccw`：世界平面使用一致的 X/Y，yaw 从 +X 朝 +Y 逆时针增加；机器人局部 +x 向前、+y 向左。局部坐标计算为 `(p - p_t) @ [[cos(yaw), -sin(yaw)], [sin(yaw), cos(yaw)]]`。不要直接把 Habitat 三维位置/四元数填入平面轨迹；本实现不猜测你的坐标转换。

`waypoint_spacing=s` 表示每隔 s 个导出帧取一个上下文帧/预测点。`action_time_step_s` 是相邻预测 waypoint 的实际时间间隔，原始导出帧间隔因此为 `action_time_step_s/s`。关系流的历史 RGB 也必须遵守这一采样协议，顺序为最旧到当前，共 `context_size+1` 张。推理 RGB 流必须采用同样的原始帧节奏；ROS 2 接口按 s 抽帧。

`action_scale_m` 是局部米制 waypoint 除以的尺度。统计工具在除以该尺度后计算 delta 范围，扩散输出解码时乘回同一尺度。它不从机器人 `MAX_V / RATE` 推断。`resolved_config.yaml` 和 v2 checkpoint 保存尺度、时间间隔、stride、统计范围及尺度指纹。

五个划分为 `train`、`validation`、`calibration_temperature`、`calibration_threshold`、`test`。程序对**两条数据流合并后的 scene_id** 检查划分互斥。两个校准集也互斥。`episode_id` 与地图版本必须存在。场景标识、不可变的 `data_revision` 以及导出来源由数据导出方保证真实；不能靠重命名场景把同一场景放进多个划分。manifest、轨迹内容和关系 JSONL 参与数据指纹；RGB 内容由不可变 revision 约束，没有逐像素重算全库哈希。

### R / D 标签

每行必填字段见 `relation_cases.jsonl`；`reachable=null` 表示未知，不能省略该字段。`path_length_m` 是未截断的原始路径长度，不能给不可达样本填 H。`distance_valid` 与 `is_shortest` 都是布尔值。

| 查询证据 | R | D 有效 |
|---|---:|---:|
| 满足足迹约束的已确认最短路径 ≤ H | 1 | 是 |
| 已找到路径 ≤ H，但未确认最短 | 1 | 否 |
| 已确认最短路径 > H | 0 | 否 |
| 完整 NavMesh 上确认不连通 | 0 | 否 |
| 仅找到路径 > H，未确认最短 | 未知 | 否 |
| 超时、地图不足、投影不合格、不确定 | 未知 | 否 |

`query_status` 使用 `found`、`no_path_certified`、`timeout`、`incomplete_map`、`invalid_projection`、`indeterminate`。已知标签要求两个端点的投影误差都在容限内。无效投影应为未知；不知道投影误差可填 null，但不能生成已知标签。

label_definition 统一记录足迹配置 ID、规划器配置 ID、`static_footprint_holonomic`、`full_scene_navmesh`、未知区域策略和 H；各行记录 scene、episode、geometry_map_id、起终点引用和查询证据。实际足迹参数、NavMesh 构建参数、规划器配置与起终点记录应随数据归档，ID 必须能够追溯到它们。这些审计字段不进入网络输入。

H 必填，没有 5 米默认值，也不等于 8 个 waypoint 的运动范围。导出时可给默认子目标标 `is_base_subgoal=true`。统计命令附带输出真实最短路长度分位数和 H 内比例，供确定最终 H 使用。最终 H 改变后需要重导标签并固定数据版本；不能在测试集上调 H。

### 校准/测试的完整候选组

温度集可以是单独 RGB 对。阈值集和测试集还必须有完整决策组，才能得到实际选择器的离线决策误选率。把 `decision_context.example.json` 的字段合并进每条关系记录：同一 `(scene_id, episode_id, decision_id)` 的上下文 RGB 和 `route_context` 完全相同，每个候选一行，各自有目标 RGB、R/D 标签及 `candidate_node_index`。

`previous_localized_node` 和 `completed_through` 是**本次定位之前**的状态。`candidate_nodes` 必须覆盖当前定位窗口，以及本次定位产生的所有合法 `base`、`base+1`、`base+2` 候选；可以预先导出定位窗口再向前扩三个节点的超集（窗口末节点 close 时，默认目标还会前移一个）。文件中声明的候选必须一行不缺。校准程序先用原距离头在同一定位窗口定位，再按与运行时完全相同的逻辑更新前沿和选择子目标。没有候选组、缺合法候选或元数据不一致会报错。

`required_nodes` 表示必须实际到达的节点，`broken_after` 表示该节点与下一个节点之间不连续。不会自动跨断点重启路线；切换路线/episode 必须由外部任务管理器明确处理。

## 3. Ubuntu 命令

使用已有 NoMaD 训练环境及其 `diffusion_policy` 依赖，保留 `diffusers==0.11.1`。新增代码使用 `LinearLR`、`SequentialLR` 和可选的非重入 activation checkpoint；需要支持这些接口的 PyTorch。没有在这台 Mac 安装或升级环境。以下从仓库 `visualnav/train` 目录执行，先把配置中的两个数据路径改为自己的路径。

```bash
python -m vint_train.data.prepare_sim_data \
  --manifest /data/habitat/manifest.json \
  --output /data/habitat/action_stats.json --context-size 3 --horizon 8

python train.py -c config/nomad_r3_habitat.yaml --ablation baseline
python train.py -c config/nomad_r3_habitat.yaml --ablation r_only
python train.py -c config/nomad_r3_habitat.yaml --ablation r_d
```

这三个训练命令应分别运行，不是在单张 GPU 上同时启动。`baseline` 没有 R3 头/关系训练流；`r_only` 有共享 R3 MLP 和 R 监督，关闭 D 损失/排序；`r_d` 启用全部。公共模块始终先初始化，再构建 R3 头，因此相同 seed 的公共初始参数相同。保持相同数据划分、动作 batch、累积、epoch 数和公共超参数。AMP 溢出可能使实际成功更新数不同，比较时报告实际动作样本数、成功更新、关系额外样本/重复次数和训练时间/峰值显存。

统计工具只用 train 动作轨迹拟合 min/max；验证/校准/测试不会拟合。`action_stats.audit.json` 记录标签数量与默认子目标路径长度审计。共享 FP32 图像缓存默认放在 manifest 旁 `.nomad_cache/<data_fingerprint>/<split>`，可通过 `simulation.cache_dir` 指定磁盘；不缓存任何视觉条件特征。关系流直接解码 RGB。每个 epoch 遍历一次动作训练集，关系流耗尽后按新 cycle seed 洗牌。

输出目录包含 `resolved_config.yaml`、`dataset_manifest.json`、`provenance.json`、逐窗口 `steps.jsonl`、逐 epoch `metrics.jsonl` 和 `checkpoint_latest.pth`。provenance 记录公共模块的初始权重 SHA256，可直接核对三组的初始化；v2 checkpoint 内也保存数据清单。默认每 10 epoch 额外存档，最终 epoch 也存档。旧的文本日志绘图脚本不负责读取这份新的嵌套 JSONL。

```bash
python train.py -c config/nomad_r3_habitat.yaml --ablation r_d \
  --resume /path/to/previous_run/checkpoint_latest.pth
```

恢复仍用相同消融选项、seed、epochs、优化器/采样/尺度/标签配置和数据指纹，会创建一个新的日志目录延续原进度。只支持完整 epoch 边界；中途终止的 epoch 从上一个完整 checkpoint 重做。不能把旧 NoMaD 权重、EMA-only 文件或其他实验 checkpoint 当初始化。修改总 epochs 会改变 cosine 计划，因此也被严格恢复检查拒绝。

### 损失与更新细节

每个累积窗口先收齐 CPU microbatch、预生成 goal mask 和四个分母。动作/temporal 的分母分别是各自有效数 `+0.01*B`，其中 B 是整个窗口动作样本数；这是保留原 NoMaD reduction 的实现，**不是严格的有效样本均值**。R/D 分母分别是各自的有效标签数。空项不回传；尾窗使用真实大小，不强行除以 4。

R 使用 BCEWithLogits，D 使用 `SmoothL1(D_hat/H, D/H, beta=0.1)`。总损失默认 `(1-1e-4)*L_A + 1e-4*L_T + lambda_R3*(L_R+L_D)`；`lambda_R3=0.1+0.9*min(s/500,1)`，s 是窗口开始前的成功更新数。R-only 关闭 D 项。

动作与关系逐 microbatch 分别前向/反向；共享参数累计梯度，窗口末统一 unscale、全局 clip norm 1、step。标准 GradScaler 的 scale 下降才记溢出跳步；不检查 `step()` 的返回值。仅成功更新推进 s 和 EMA。完全无监督窗口不调用优化器。学习率按有效训练 epoch 推进，零更新 epoch 不推进。初始 lr 是峰值的 0.1，4 个有效 epoch 后到峰值，再 cosine。两种预热各自记录。

checkpoint v2 保存模型、原格式 EMA 权重/衰减字段、优化器、调度器、GradScaler、成功更新数、参数组名称/顺序、RNG、完整推理配置、数据与尺度指纹。每个 epoch/cycle 重建有明确 seed 的 DataLoader worker。验证隔离 Python/NumPy/PyTorch RNG；CUDA 算法和跨版本可复现性仍需在实际环境验收，不承诺跨机器逐 bit 相同。

梯度诊断包括 unscale 后裁剪前全局范数，以及定期采样的视觉压缩层/最后 Transformer 层两任务梯度范数和夹角。后者只针对所记录的共享参数与首个有效 microbatch，不代表整个模型的完整窗口梯度冲突测量。

## 4. 校准、离线评价与推理

```bash
python -m vint_train.training.calibrate_r3 \
  --checkpoint /path/to/run/checkpoint_latest.pth \
  --manifest /data/habitat/manifest.json --weights ema --device cuda:0 \
  --output /path/to/run/calibration.json

python -m vint_train.training.calibrate_r3 \
  --checkpoint /path/to/run/checkpoint_latest.pth \
  --manifest /data/habitat/manifest.json --weights ema --device cuda:0 \
  --calibration /path/to/run/calibration.json --split test \
  --output /path/to/run/test_report.json
```

温度拟合只用 `calibration_temperature` 的已知 R；正温度以 `exp(log_T)` 参数化。阈值只用 `calibration_threshold`，在经验 FDR≤0.05、已知标签的接受数≥100 的阈值中最大化召回，再按更低 FDR、更高阈值打破平局。未知标签不冒充负例或计入这 100 个，另报接受的未知候选数。无合格阈值输出 `status: failed` 并非零退出，不能用于正式导航。CLI 可改规则做额外研究，论文主设置应保留默认并记录实际规则。

校准产物绑定精确的模型 tensor SHA256、raw/EMA 来源、标签定义和选择器设置。改变权重、H、路线窗口或排序规则后，旧产物不能复用。训练前应冻结完整划分清单；校准和测试命令也检查训练时的数据指纹，不能悄悄替换测试集。候选级 AP/Brier/FDR/召回、接受场景/episode 数、D 的有效正例 MAE、决策级误选率分别报告。候选 FDR 只是该独立样本上的经验量，不是决策或闭环风险保证。

`inference/rgb_policy.py` 接收未 ImageNet 归一化的 `[B,3*(context+1),H,W]` RGB 与 `[B,3,H,W]` 目标 RGB；共享 helper 完成归一化。`score_rgb_pairs` 输出原 temporal distance、R 概率、D 和条件；先完成定位/选择，仅在 selected 时把对应条件传给 `sample_conditioned_actions`，返回 `[B,N,horizon,2]` 米制 waypoint。

通用路线接口顺序：`localization_window` → `prepare_route_decision` → 补齐 `legal_candidate_nodes` 的评分 → `GoalRelationSelector.select` → 仅 selected 时扩散采样。`tick()` 必须由独立的单调墙钟周期调用；一段没有图像的时间也必须能停止。每个真实新 RGB 用递增 frame_id，重复帧不产生新决定。新 episode 调 `reset_episode()` 并重置外部定位/前沿/到达状态。

默认目标通过 R 时保留；否则只在合法的后两个节点中按 D、路线索引排序。不会越过尚未确认到达的必经节点、断点或回到已完成节点。选中更远目标不推进完成前沿。D>H 仅统计不一致；非有限候选作为异常拒绝。全拒绝立即 abstain 且不扩散；连续 3 个新帧拒绝，或首次拒绝 2 秒未恢复，或图像流中断 2 秒，进入保持停止的 failed。只有显式 reset 能清除 failed。

`base_accept_rate`、`fallback_rate`、`distance_changed_selection_rate` 的分母为全部决策；fallback_rate 包含尝试替代后仍全部拒绝的情况。`distance_changed_selection_rate_on_fallback` 的分母为实际选中替代目标的决策。D 改选用同模型、同候选集合的路线顺序选择作对照；它不等价于另训的 R-only 闭环消融。拒绝、异常、超时和失败独立计数。

### ROS 2 接入

把 checkpoint 与同次实验的 `resolved_config.yaml` 放到 `deployment/config/models.yaml` 的 `nomad_r3` 所指位置；不能用尚未填数据参数的训练模板作部署配置。在 `deployment/src` 下启动：

```bash
python pd_controller.py --control-mode metric
python navigate.py --model nomad_r3 --dir YOUR_TOPOMAP \
  --calibration /path/to/calibration.json --decision-log /path/to/decisions.jsonl
```

topomap RGB 文件为 `0.jpg` 起的连续数字索引，旁边可放 `route.json`（格式见 route 示例）。`--radius` 与 `--close-threshold` 必须和校准设置一致。`--allow-uncalibrated` 仅供显式调试，日志记录 calibrated=false。

导航和 exploration 均在 `/topoplan/metric_waypoint` 发布一条 `Float32MultiArray [prediction_time_s, x_m, y_m]`；预测时间为 `(waypoint_index+1)*action_time_step_s`。PD 控制器使用同一条消息中的坐标和时间计算速度，再应用机器人速度上限。`/waypoint` 和 `/topoplan/waypoint_time_s` 继续提供观测接口。启动 controller 时明确使用 `--control-mode metric`，切回 legacy 时重启并使用 `--control-mode legacy`；包装脚本会自动根据模型配置选择模式。

`/topoplan/r3_stop` 为带心跳的停止信号，控制器在停止、心跳中断、waypoint 过期或缺预测时间时输出零速度，并清空旧 waypoint。独立墙钟 watchdog 不依赖 ROS 模拟时间推进。ROS 2 图像必须有正数且递增的 header stamp.sec/nanosec（没有 header.seq），零时间戳或重传同一标识不计新帧；不同 episode 时间戳重置时先发 reset。`/topoplan/reset_episode`（std_msgs/Empty）清空拒绝计数、图像历史、定位/前沿和失败状态。`/topoplan/r3_status` 与 JSONL 记录状态转移，图像中断即使没有后续决策也会留记录。拒绝/失败不发布到达成功，且不发布旧 waypoint。

到达状态独立于选择状态：最终节点需原距离头判断 close，且没有跳过必经节点/断点、R3 没有拒绝，才发布 reached_goal。Habitat 的真正成功、碰撞、SPL 等仍由仿真评估器根据真值判断；视觉定位得到的 reached_goal 不能代替论文真值成功率。

## 5. 测试与真实环境验收

在已有训练环境、`visualnav/train` 目录运行新增的轻量测试：

```bash
python -m unittest discover -s tests -p 'test_r3_*.py' -v
```

2026-11-08 验证：21 个 R3 训练侧测试、10 个 ROS 2 部署侧测试通过；部署测试包含真实 DDS、合成 RGB、小模型 checkpoint 实际推理，以及图像中断和模拟时钟暂停时的停车检查。以下列表保留为真实数据、真实闭环和实机的验收要求，不因单测或合成数据 smoke 通过而视为全部完成：

1. 用小份真实 Habitat 导出验证坐标、相机帧序、scene 划分和最短路标签；注入超时、错误投影、跨流 scene 泄漏应报错/标未知。
2. 检查训练启动没有下载/读取预训练权重；相同 seed 的公共初始化相同。动作、temporal、R、D 分别有有效监督，各有效模块梯度有限且非空；兼容性未使用参数按 provenance 排除。
3. 对不均匀有效标签数、全未知关系 batch、全无监督窗口、尾窗验证归一化。baseline 必须使用同一动作/temporal reduction。
4. CUDA FP16 注入梯度溢出：AdamW 参数、EMA、successful_steps、R3 预热都不能误推进；检查 unscale 后全局裁剪。确认关 AMP、开 activation checkpoint 的实际结果。
5. 中断在 epoch 边界，对比连续运行与恢复后的下一更新、优化器状态、LR、EMA、Scaler 和 RNG；数据/尺度/参数组/训练设置不匹配必须拒绝恢复。
6. 对照一条已知米制轨迹确认训练目标→归一化→反归一化→部署 waypoint 和预测时间一致；验证/校准/测试不重拟合范围。
7. 合成路线检查窗口边缘、默认目标优先、有限前跳、必经节点、断点、D 改选/平局、D>H、NaN；补齐候选组后离线选择应与在线一致。
8. 控制链验证重复图像不加拒绝次数、3 次/2 秒拒绝、图像中断、模拟时钟暂停、节点退出、失败后的新图像、reset、最终到达；abstain 时扩散调用数应为零，控制输出为零，成功消息不得误发。
9. 在相同场景、公共初始化、可比动作预算下分别完成 NoMaD / R-only / R+D 的 Habitat 闭环评价。另报标签查询量、关系额外训练量、时间/显存、延迟、拒绝率、超时/失败、候选误选、决策误选与 D 改选率；不要仅凭离线 AP 或 D MAE 宣称导航提升。

参考代码与借用范围见 `R3_REFERENCES.md`。
