# 介绍
这是基于`Qlib`提供的`Meta Controller`组件实现的`DDG-DA`。

更多详情请参阅论文：*DDG-DA: Data Distribution Generation for Predictable Concept Drift Adaptation* [[arXiv](https://arxiv.org/abs/2201.04038)]


# 背景
在许多现实世界的场景中，我们经常处理随时间顺序收集的流数据。由于环境的非平稳性，流数据分布可能会以不可预测的方式发生变化，这被称为概念漂移。为了处理概念漂移，以前的方法首先检测概念漂移发生的时间/地点，然后调整模型以适应最新数据的分布。然而，在许多情况下，环境演变的一些潜在因素是可预测的，这使得对流数据的未来概念漂移趋势进行建模成为可能，而这种情况在以前的工作中没有得到充分的探索。

因此，我们提出了一种新颖的方法`DDG-DA`，它可以有效地预测数据分布的演变并提高模型的性能。具体来说，我们首先训练一个预测器来估计未来的数据分布，然后利用它来生成训练样本，最后在生成的数据上训练模型。

# 数据集
论文中的数据是私有的。因此，我们在Qlib的公共数据集上进行实验。
虽然数据集不同，但结论仍然相同。通过应用`DDG-DA`，用户可以在测试阶段看到代理模型的IC和预测模型的性能都有上升的趋势。

# 运行代码
用户可以通过运行以下命令来尝试`DDG-DA`：
```bash
    python workflow.py run
```

默认的预测模型是`Linear`。用户可以在`DDG-DA`初始化时通过更改`forecast_model`参数来选择其他预测模型。例如，用户可以通过运行以下命令来尝试`LightGBM`预测模型：
```bash
    python workflow.py --conf_path=../workflow_config_lightgbm_Alpha158.yaml run
```

# 结果
Qlib公共数据集中相关方法的结果可以在[这里](../)找到

# 要求
以下是运行DDG-DA的``workflow.py``的最低硬件要求。
* 内存：45G
* 磁盘：4G

对于这个例子，带有CPU和RAM的Pytorch就足够了。
