# 投资组合优化策略

## 介绍

在`qlib/examples/benchmarks`中，我们有各种预测股票收益的**alpha**模型。我们还使用一个简单的基于规则的`TopkDropoutStrategy`来评估这些模型的投资表现。然而，这样的策略过于简单，无法控制投资组合的风险，如相关性和波动性。

为此，应使用基于优化的策略来权衡回报和风险。在本文档中，我们将展示如何使用`EnhancedIndexingStrategy`在最小化相对于基准的跟踪误差的同时最大化投资组合的回报。


## 准备

我们的例子使用中国股票市场数据。

1. 准备CSI300权重：

   ```bash
   wget https://github.com/SunsetWolf/qlib_dataset/releases/download/v0/csi300_weight.zip
   unzip -d ~/.qlib/qlib_data/cn_data csi300_weight.zip
   rm -f csi300_weight.zip
   ```
   注意：我们没有找到任何公开的免费资源来获取基准中的权重。为了运行这个例子，我们手动创建了这个权重数据。

2. 准备风险模型数据：

   ```bash
   python prepare_riskdata.py
   ```

这里我们使用在`qlib.model.riskmodel`中实现的**统计风险模型**。
然而，强烈建议用户使用其他风险模型以获得更好的质量：
* **基本面风险模型**，如MSCI BARRA
* [深度风险模型](https://arxiv.org/abs/2107.05201)


## 端到端工作流

您可以通过运行`qrun config_enhanced_indexing.yaml`来完成`EnhancedIndexingStrategy`的工作流。

在这个配置中，与`qlib/examples/benchmarks/workflow_config_lightgbm_Alpha158.yaml`相比，我们主要更改了策略部分。
