#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
Qlib 提供了两种接口:
(1) 用户可以通过简单的配置来定义量化研究的工作流程。
(2) Qlib 采用模块化设计，支持像搭积木一样通过代码创建研究工作流程。

接口 (1) 的使用方式是 `qrun XXX.yaml`。接口 (2) 的使用方式如此脚本所示，它与 `qrun XXX.yaml` 的功能几乎相同。
"""
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_BENCH, CSI300_GBDT_TASK


if __name__ == "__main__":
    # 使用默认数据
    provider_uri = "~/.qlib/qlib_data/cn_data"  # 目标目录
    # 获取数据
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    # 初始化 qlib
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # 从配置初始化模型
    model = init_instance_by_config(CSI300_GBDT_TASK["model"])
    # 从配置初始化数据集
    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])

    # 投资组合分析配置
    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
            "account": 100000000,
            "benchmark": CSI300_BENCH,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }

    # 注意：这行代码是可选的
    # 它演示了数据集可以独立使用。
    example_df = dataset.prepare("train")
    print(example_df.head())

    # 开始实验
    with R.start(experiment_name="workflow"):
        # 记录参数
        R.log_params(**flatten_dict(CSI300_GBDT_TASK))
        # 训练模型
        model.fit(dataset)
        # 保存模型
        R.save_objects(**{"params.pkl": model})

        # 预测
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # 信号分析
        sar = SigAnaRecord(recorder)
        sar.generate()

        # 回测。如果用户想根据自己的预测进行回测，
        # 请参考 https://qlib.readthedocs.io/en/latest/component/recorder.html#record-template。
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
