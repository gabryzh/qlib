# 介绍
由于金融市场环境的非平稳性，数据分布在不同时期可能会发生变化，这使得在训练数据上构建的模型的性能在未来的测试数据中会下降。
因此，使预测模型/策略适应市场动态对模型/策略的性能非常重要。

下表显示了不同解决方案在不同预测模型上的性能。

## Alpha158 数据集
这是[qlib数据的众包版本](data_collector/crowd_source/README.md)：https://github.com/chenditc/investment_data/releases
```bash
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
mkdir -p ~/.qlib/qlib_data/cn_data
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=2
rm -f qlib_bin.tar.gz
```

| 模型名称 | 数据集 | IC | ICIR | Rank IC | Rank ICIR | 年化收益率 | 信息比率 | 最大回撤 |
|---|---|---|---|---|---|---|---|---|
| RR[Linear] |Alpha158 |0.0945|0.5989|0.1069 |0.6495 |0.0857 |1.3682 |-0.0986 |
| DDG-DA[Linear] |Alpha158 |0.0983|0.6157|0.1108 |0.6646 |0.0764 |1.1904 |-0.0769 |
| RR[LightGBM] |Alpha158 |0.0816|0.5887|0.0912 |0.6263 |0.0771 |1.3196 |-0.0909 |
| DDG-DA[LightGBM] |Alpha158 |0.0878|0.6185|0.0975 |0.6524 |0.1261 |2.0096 |-0.0744 |

- `Alpha158`数据集的标签范围设置为20。
- 滚动时间间隔设置为20个交易日。
- 测试滚动期为2017年1月至2020年8月。
- 结果基于众包版本。qlib数据的雅虎版本不包含`VWAP`，因此所有相关因子都缺失并用0填充，这导致了秩亏矩阵（矩阵没有满秩），并使得DDG-DA的低层优化无法求解。
