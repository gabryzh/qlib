# 介绍
这个文件夹包含2个例子
- 一个高频数据集的例子
- 一个在高频数据中预测价格趋势的例子

## 高频数据集

这个数据集是RL高频交易的一个例子。

### 获取高频数据

通过运行以下命令获取高频数据：
```bash
    python workflow.py get_data
```

### 转储、重新加载和重新初始化数据集


高频数据集在`workflow.py`中实现为`qlib.data.dataset.DatasetH`。`DatatsetH`是[`qlib.utils.serial.Serializable`](https://qlib.readthedocs.io/en/latest/advanced/serial.html)的子类，其状态可以以`pickle`格式从磁盘转储或加载。

### 关于重新初始化

从磁盘重新加载`Dataset`后，`Qlib`还支持重新初始化数据集。这意味着用户可以重置`Dataset`或`DataHandler`的某些状态，例如`instruments`、`start_time`、`end_time`和`segments`等，并根据这些状态生成新数据。

`workflow.py`中给出了示例，用户可以按如下方式运行代码。

### 运行代码

通过运行以下命令来运行示例：
```bash
    python workflow.py dump_and_load_dataset
```

## 基准性能（预测高频数据中的价格趋势）

以下是用于预测高频数据价格趋势的模型结果。我们将在未来不断更新基准模型。

| 模型名称 | 数据集 | IC | ICIR | Rank IC | Rank ICIR | 多头精确率| 空头精确率 | 多空平均收益 | 多空平均夏普比率 |
|---|---|---|---|---|---|---|---|---|---|
| LightGBM | Alpha158 | 0.0349±0.00 | 0.3805±0.00| 0.0435±0.00 | 0.4724±0.00 | 0.5111±0.00 | 0.5428±0.00 | 0.000074±0.00 | 0.2677±0.00 |
