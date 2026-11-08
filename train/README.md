### 训练

R1 按 **RGB + 现有轨迹访问分布监督**接入，无需新增深度或占据标签。
配置为 `config/nomad_r1_r3_habitat.yaml`，数据准备、四组主实验与消融见
[R1 修改计划与使用说明](../docs/R1_IMPLEMENTATION_PLAN.md)。新增对照、初始化/随机流校验、schema 2 统计和校准要求见 [R1 审查修复报告](../docs/R1_REVIEW_FIX_REPORT.md)。本轮尚未运行测试或训练。

Habitat 从零训练的 NoMaD / R-only / R+D 使用独立入口配置，见
[R3 数据、训练、校准和 Ubuntu 验收说明](../docs/R3_HABITAT.md)。
入口配置是 `config/nomad_r3_habitat.yaml`，不要用下面的旧训练脚本启动 R3。

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
