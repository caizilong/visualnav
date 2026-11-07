# R3 实现参考与借用范围

以下为接口与设计参考。本轮没有复制/嵌入外部完整文件，也没有新增训练框架依赖；依赖原仓库已有的 PyTorch、Diffusers 和 diffusion_policy。实现效果尚需 Ubuntu 验收。

| 来源 | 核查版本 / 地址 | 本实现使用的范围 |
|---|---|---|
| PyTorch AMP 官方说明 | [2.15 AMP examples](https://docs.pytorch.org/docs/2.15/notes/amp_examples.html) | 累积期间 scale 不变，窗口末 unscale、clip、step/update；遵守有效 batch 的更新时机。PyTorch 2.15 是查阅文档版本，不表示本机已安装该版本。代码沿用兼容现有环境的 `torch.cuda.amp` 入口。 |
| PyTorch LR scheduler | [SequentialLR](https://docs.pytorch.org/docs/2.15/generated/torch.optim.lr_scheduler.SequentialLR.html) | LinearLR 正起点预热后切 cosine；按有效 epoch 调度。 |
| Hugging Face Diffusers | [v0.11.1 training_utils.py](https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/training_utils.py)（Apache-2.0） | 继续使用本项目 pin 的 `EMAModel(model=..., power=.75)`、`ema.step(model)` 和 `averaged_model`；显式保存 decay 等字段，没有假设该旧版本具有新版本的 EMA state_dict API。 |
| diffusion_policy | [5ba07ac6661db573af695b419a7947ecb704690f](https://github.com/real-stanford/diffusion_policy/tree/5ba07ac6661db573af695b419a7947ecb704690f)（MIT） | 参考动作 delta/归一化及条件 U-Net 工作流；U-Net 沿用当前工程依赖，未在此变更其安装版本。累积/EMA 更新时机按本任务规则重新实现，未照搬参考训练循环。 |
| ULVN | [e2f47fe3e2757a1ebdc041f4364dddd26a19f9f5 navigation.py](https://github.com/hren20/ulvn/blob/e2f47fe3e2757a1ebdc041f4364dddd26a19f9f5/ulvn_core/navigation.py)（MIT） | 参考定位、实际导航目标和任务状态分离；本实现另写有限前跳、必经节点/断点约束、abstain/failed 和 reset，不导入其整套导航系统。 |
| TorchUncertainty | [3f82fe5d15a7bf877a821731baadef1e4731c31d temperature_scaler.py](https://github.com/torch-uncertainty/torch-uncertainty/blob/3f82fe5d15a7bf877a821731baadef1e4731c31d/src/torch_uncertainty/post_processing/calibration/temperature_scaler.py)（Apache-2.0） | 参考独立校准数据与温度缩放结构；本实现只用 PyTorch LBFGS 拟合 log-temperature，不引入该框架，也不把温度缩放当作 FDR 保证。 |

候选 FDR 阈值选择、窗口级不同分母、v2 恢复校验、数据合同和路线状态机均为本工程针对既定方案的实现。参考成熟项目的接口不等于已验证 R3 的研究新颖性或实验收益。
