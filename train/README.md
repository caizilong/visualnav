### 训练

在 [nomad.yaml](config/nomad.yaml) 中设置两个独立的开关：

```yaml
spatial_bottleneck:
  enabled: false
goal_relation:
  enabled: false
```

直接编辑现有配置块里的 enabled 字段。默认都关闭时，使用原 NoMaD 训练数据和参数。打开任一模块后自动使用 Habitat 数据和 habitat_training 参数。两者都关闭时，可以在 YAML 顶层设置 training_profile: habitat_scratch，或传入 --ablation baseline，以训练 Habitat 上的 NoMaD 对照。

--spatial-bottleneck-mode 和 --ablation 分别覆盖对应模块的 YAML 选择。数据准备和消融命令见[空间瓶颈使用说明](../docs/SPATIAL_BOTTLENECK_IMPLEMENTATION_PLAN.md)及[目标关系训练与校准说明](../docs/GOAL_RELATION_HABITAT.md)。

已有扩展 checkpoint 的配置、训练状态和校准文件含旧模块名。使用新入口前需迁移元数据，并用对应模型重新校准。若要精确续训旧实验，请使用保存该 checkpoint 时的代码版本。外部 ROS 2 节点若订阅停止或状态信号，请同步更新话题名，见[部署说明](../deployment/README.md)。

下方脚本用于基础训练。

```
cd train
./run_train.sh
```

### 绘图

在仓库根目录执行：

```
cd /home/czl/visualnav
conda activate nomad_train

python train/plot_training_log.py train/training.log \
  --output-dir train/logs/paper_figures \
  --formats png svg \
  --figures-only \
  --log-y
```

图片保存在 train/logs/paper_figures/。

- --formats png svg：同时生成 PNG 和 SVG 矢量图；只要 PNG 就改成 --formats png。
- --figures-only：只输出图片。
- --log-y：损失使用对数纵轴，余弦相似度仍使用线性纵轴；去掉它则全部使用线性坐标。

### 部署
