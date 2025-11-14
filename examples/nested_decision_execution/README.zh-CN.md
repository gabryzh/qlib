# 嵌套决策执行

此工作流是回测中嵌套决策执行的一个示例。Qlib 在回测中支持嵌套决策执行。这意味着用户可以使用不同的策略在不同的频率下做出交易决策。

## 每周投资组合生成和每日订单执行

此工作流提供了一个示例，该示例在每周频率下使用 DropoutTopkStrategy（一种基于每日频率 Lightgbm 模型的策略）进行投资组合生成，并使用 SBBStrategyEMA（一种使用 EMA 进行决策的基于规则的策略）在每日频率下执行订单。

### 用法

通过运行以下命令开始回测：
```bash
    python workflow.py backtest
```

通过运行以下命令开始收集数据：
```bash
    python workflow.py collect_data
```

## 每日投资组合生成和分钟级订单执行

此工作流还提供了一个高频示例，该示例在每日频率下使用 DropoutTopkStrategy 进行投资组合生成，并在分钟级频率下使用 SBBStrategyEMA 执行订单。

### 用法

通过运行以下命令开始回测：
```bash
    python workflow.py backtest_highfreq
```
