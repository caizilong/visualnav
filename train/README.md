### 训练
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
