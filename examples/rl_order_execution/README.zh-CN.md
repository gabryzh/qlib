# 订单执行的强化学习示例

这个文件夹包含一个订单执行场景的强化学习（RL）工作流示例，包括训练工作流和回测工作流。

## 数据处理

### 获取数据

```
python -m qlib.cli.data qlib_data --target_dir ./data/bin --region hs300 --interval 5min
```

### 生成 Pickle 格式的数据

要运行此示例中的代码，我们需要 pickle 格式的数据。为此，请运行以下命令（可能需要几分钟才能完成）：

[//]: # (TODO: 我们鼓励实现`Dataset`和`DataHandler`的不同子类，而不是转储不同格式的数据帧（例如`qlib/contrib/data/highfreq_provider.py`中的`_gen_dataset`和`_gen_day_dataset`）。这将保持工作流更清晰，接口更一致，并将所有复杂性移至子类。)

```
python scripts/gen_pickle_data.py -c scripts/pickle_data_config.yml
python scripts/gen_training_orders.py
python scripts/merge_orders.py
```

完成后，`data/`下的结构应为：

```
data
├── bin
├── orders
└── pickle
```

## 训练

每个训练任务由一个配置文件指定。任务`TASKNAME`的配置文件是`exp_configs/train_TASKNAME.yml`。此示例提供了两个训练任务：

- **PPO**：IJCAL 2020论文“[An End-to-End Optimal Trade Execution Framework based on Proximal Policy Optimization](https://www.ijcai.org/proceedings/2020/0627.pdf)”中提出的方法。
- **OPDS**：AAAI 2021论文“[Universal Trading for Order Execution with Oracle Policy Distillation](https://arxiv.org/abs/2103.10860)”中提出的方法。

这两种方法的主要区别在于它们的奖励函数。有关详细信息，请参阅它们的配置文件。

以 OPDS 为例，要运行训练工作流，请运行：

```
python -m qlib.rl.contrib.train_onpolicy --config_path exp_configs/train_opds.yml --run_backtest
```

指标、日志和检查点将存储在`outputs/opds`下（由`exp_configs/train_opds.yml`配置）。

## 回测

训练工作流完成后，训练好的模型可用于回测工作流。仍然以 OPDS 为例，训练完成后，模型的最新检查点可以在`outputs/opds/checkpoints/latest.pth`找到。要运行回测工作流：

1. 在`exp_configs/train_opds.yml`中取消`weight_file`参数的注释（默认情况下是注释掉的）。虽然可以在不设置检查点的情况下运行回测工作流，但这将导致随机初始化的模型结果，从而使它们变得毫无意义。
2. 运行`python -m qlib.rl.contrib.backtest --config_path exp_configs/backtest_opds.yml`。

回测结果存储在`outputs/checkpoints/backtest_result.csv`中。

除了 OPDS 和 PPO，我们还提供了 TWAP（[时间加权平均价格](https://en.wikipedia.org/wiki/Time-weighted_average_price)）作为一个弱基线。TWAP 的配置文件是`exp_configs/backtest_twap.yml`。

### 回测和训练流程测试之间的差距

值得注意的是，回测过程的结果可能与训练期间使用的测试过程的结果不同。
这是因为在训练和回测期间使用不同的模拟器来模拟市场状况。
在训练流程中，为了效率，使用了名为`SingleAssetOrderExecutionSimple`的简化模拟器。
`SingleAssetOrderExecutionSimple`对交易量没有限制。
无论订单金额多少，都可以完全执行。
然而，在回测期间，使用了更真实的模拟器`SingleAssetOrderExecution`。
它考虑了更真实场景中的实际约束（例如，交易量必须是最小交易单位的倍数）。
因此，回测期间实际执行的订单金额可能与预期执行的金额不同。

如果您想获得与训练流程中测试期间获得的结果完全相同的结果，您可以只运行带有回测阶段的训练流程。
为此：
- 修改训练配置。添加要使用的检查点的路径（请参阅以下示例）。
- 运行`python -m qlib.rl.contrib.train_onpolicy --config_path PATH/TO/CONFIG --run_backtest --no_training`

```yaml
...
policy:
  class: PPO  # PPO, DQN
  kwargs:
    lr: 0.0001
    weight_file: PATH/TO/CHECKPOINT
  module_path: qlib.rl.order_execution.policy
...
```

## 基准（待定）

为了准确评估使用强化学习算法的模型的性能，最好多次运行实验并计算所有试验的平均性能。然而，鉴于模型训练的耗时性，这并非总是可行的。另一种方法是每个训练任务只运行一次，选择验证性能最高的 10 个检查点来模拟多次试验。在此示例中，我们使用“价格优势（PA）”作为选择这些检查点的指标。这 10 个检查点在测试集上的平均性能如下：

| **模型** | **PA 均值和标准差** |
|-----------------------------|-----------------------|
| OPDS (使用 PPO 策略) | 0.4785 ± 0.7815 |
| OPDS (使用 DQN 策略) | -0.0114 ± 0.5780 |
| PPO | -1.0935 ± 0.0922 |
| TWAP | ≈ 0.0 ± 0.0 |

上表还包括 TWAP 作为基于规则的基线。TWAP 的理想 PA 应为 0.0，但是，在此示例中，订单执行分为两个步骤：首先，将订单平均分配到每个半小时，然后在每个半小时内每五分钟分配一次。由于在一天中的最后五分钟禁止交易，因此这种方法可能与全天的传统 TWAP 略有不同（因为在最后一个“半小时”中缺少 5 分钟）。因此，TWAP 的 PA 可以被认为是一个接近 0.0 的数字。要验证这一点，您可以运行 TWAP 回测并检查结果。
