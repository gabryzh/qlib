
# 介绍

什么是 GeneralPtNN
- 修复了先前无法同时支持时间序列和表格数据设计的不足
- 现在您只需替换 Pytorch 模型结构即可运行 NN 模型。

我们提供一个示例来演示当前设计的有效性。
- `workflow_config_gru.yaml` 与之前的结果对齐 [GRU(Kyunghyun Cho, 等)](../README.md#Alpha158-dataset)
  - `workflow_config_gru2mlp.yaml` 用于演示我们可以以最小的改动将配置从时间序列转换为表格数据
    - 您只需更改网络和数据集类即可进行转换。
- `workflow_config_mlp.yaml` 实现了与 [MLP](../README.md#Alpha158-dataset) 类似的功能

# 待办事项

- 我们将使现有模型与当前设计对齐。

- `workflow_config_mlp.yaml` 的结果与 [MLP](../README.md#Alpha158-dataset) 的结果不同，因为 GeneralPtNN 与之前的实现相比，具有不同的停止方法。具体来说，GeneralPtNN 根据周期来控制训练，而之前的方法则由最大步数控制。
