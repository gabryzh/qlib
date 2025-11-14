# 使用时间路由适配器和最优传输学习多种股票交易模式

时间路由适配器（TRA）旨在捕捉股票市场数据中的多种交易模式。更多详情请参阅[我们的论文](http://arxiv.org/abs/2106.12950)。

如果您在研究中发现我们的工作有用，请引用：
```
@inproceedings{HengxuKDD2021,
 author = {Hengxu Lin and Dong Zhou and Weiqing Liu and Jiang Bian},
 title = {Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport},
 booktitle = {Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
 series = {KDD '21},
 year = {2021},
 publisher = {ACM},
}

@article{yang2020qlib,
  title={Qlib: An AI-oriented Quantitative Investment Platform},
  author={Yang, Xiao and Liu, Weiqing and Zhou, Dong and Bian, Jiang and Liu, Tie-Yan},
  journal={arXiv preprint arXiv:2009.11189},
  year={2020}
}
```

## 用法（推荐）

**更新**：`TRA`已被移至`qlib.contrib.model.pytorch_tra`，以支持其他`Qlib`组件，如`qlib.workflow`和`Alpha158/Alpha360`数据集。

请按照官方[文档](https://qlib.readthedocs.io/en/latest/component/workflow.html)使用`TRA`和`workflow`。这里我们也提供了一些示例配置文件：

- `workflow_config_tra_Alpha360.yaml`: 使用`Alpha360`数据集运行`TRA`
- `workflow_config_tra_Alpha158.yaml`: 使用`Alpha158`数据集运行`TRA`（带特征子采样）
- `workflow_config_tra_Alpha158_full.yaml`: 使用`Alpha158`数据集运行`TRA`（不带特征子采样）

`TRA`的性能报告在[基准测试](https://github.com/microsoft/qlib/tree/main/examples/benchmarks)中。

## 用法（未维护）

本节用于重现论文中的结果。

### 运行

我们在`run.sh`中附加了论文的运行脚本。

这里有两种运行模型的方式：

* 使用默认参数从脚本运行

  您可以直接从Qlib命令`qrun`运行：
  ```
  qrun configs/config_alstm.yaml
  ```

* 使用自定义参数从代码运行

  也允许设置不同的参数。请参阅`example.py`中的代码：
  ```
  python example.py --config_file configs/config_alstm.yaml
  ```

这里我们在预训练的主干模型上训练TRA。因此，我们在TRA的脚本之前运行`*_init.yaml`。

### 结果

运行脚本后，您可以在路径`./output`中找到结果文件：

* `info.json` - 配置设置和结果指标。
* `log.csv` - 运行日志。
* `model.bin` - 模型参数字典。
* `pred.pkl` - 预测分数和用于推断的输出。

论文中报告的评估指标：
此结果由qlib==0.7.1生成。

| 方法 | MSE| MAE| IC | ICIR | AR | AV | SR | MDD |
|---|---|---|---|---|---|---|---|---|
|Linear|0.163|0.327|0.020|0.132|-3.2%|16.8%|-0.191|32.1%|
|LightGBM|0.160(0.000)|0.323(0.000)|0.041|0.292|7.8%|15.5%|0.503|25.7%|
|MLP|0.160(0.002)|0.323(0.003)|0.037|0.273|3.7%|15.3%|0.264|26.2%|
|SFM|0.159(0.001)|0.321(0.001)|0.047|0.381|7.1%|14.3%|0.497|22.9%|
|ALSTM|0.158(0.001)|0.320(0.001)|0.053|0.419|12.3%|13.7%|0.897|20.2%|
|Trans.|0.158(0.001)|0.322(0.001)|0.051|0.400|14.5%|14.2%|1.028|22.5%|
|ALSTM+TS|0.160(0.002)|0.321(0.002)|0.039|0.291|6.7%|14.6%|0.480|22.3%|
|Trans.+TS|0.160(0.004)|0.324(0.005)|0.037|0.278|10.4%|14.7%|0.722|23.7%|
|ALSTM+TRA(Ours)|0.157(0.000)|0.318(0.000)|0.059|0.460|12.4%|14.0%|0.885|20.4%|
|Trans.+TRA(Ours)|0.157(0.000)|0.320(0.000)|0.056|0.442|16.1%|14.2%|1.133|23.1%|

我们在论文中的实验结果的更详细演示可以在`Report.ipynb`中找到。

## 常见问题

有关使用TRA的帮助或问题，请提交GitHub issue。

有时我们可能会遇到损失为`NaN`的情况，请检查sinkhorn算法中的`epsilon`参数，根据输入的尺度调整`epsilon`很重要。
