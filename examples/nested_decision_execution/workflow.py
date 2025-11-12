#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
当前版本中 `backtest` 的预期结果如下
'The following are analysis results of benchmark return(1day).'
                       risk
mean               0.000651
std                0.012472
annualized_return  0.154967
information_ratio  0.805422
max_drawdown      -0.160445
'The following are analysis results of the excess return without cost(1day).'
                       risk
mean               0.001258
std                0.007575
annualized_return  0.299303
information_ratio  2.561219
max_drawdown      -0.068386
'The following are analysis results of the excess return with cost(1day).'
                       risk
mean               0.001110
std                0.007575
annualized_return  0.264280
information_ratio  2.261392
max_drawdown      -0.071842
[1706497:MainThread](2021-12-07 14:08:30,263) INFO - qlib.workflow - [record_temp.py:441] - 投资组合分析记录 'port_analysis_30minute.
pkl' 已作为实验 2 的工件保存
'The following are analysis results of benchmark return(30minute).'
                       risk
mean               0.000078
std                0.003646
annualized_return  0.148787
information_ratio  0.935252
max_drawdown      -0.142830
('The following are analysis results of the excess return without '
 'cost(30minute).')
                       risk
mean               0.000174
std                0.003343
annualized_return  0.331867
information_ratio  2.275019
max_drawdown      -0.074752
'The following are analysis results of the excess return with cost(30minute).'
                       risk
mean               0.000155
std                0.003343
annualized_return  0.294536
information_ratio  2.018860
max_drawdown      -0.075579
[1706497:MainThread](2021-12-07 14:08:30,277) INFO - qlib.workflow - [record_temp.py:441] - 投资组合分析记录 'port_analysis_5minute.p
kl' 已作为实验 2 的工件保存
'The following are analysis results of benchmark return(5minute).'
                       risk
mean               0.000015
std                0.001460
annualized_return  0.172170
information_ratio  1.103439
max_drawdown      -0.144807
'The following are analysis results of the excess return without cost(5minute).'
                       risk
mean               0.000028
std                0.001412
annualized_return  0.319771
information_ratio  2.119563
max_drawdown      -0.077426
'The following are analysis results of the excess return with cost(5minute).'
                       risk
mean               0.000025
std                0.001412
annualized_return  0.281536
information_ratio  1.866091
max_drawdown      -0.078194
[1706497:MainThread](2021-12-07 14:08:30,287) INFO - qlib.workflow - [record_temp.py:466] - 指标分析记录 'indicator_analysis_1day
.pkl' 已作为实验 2 的工件保存
'The following are analysis results of indicators(1day).'
        value
ffr  0.945821
pa   0.000324
pos  0.542882
[1706497:MainThread](2021-12-07 14:08:30,293) INFO - qlib.workflow - [record_temp.py:466] - 指标分析记录 'indicator_analysis_30mi
nute.pkl' 已作为实验 2 的工件保存
'The following are analysis results of indicators(30minute).'
        value
ffr  0.982910
pa   0.000037
pos  0.500806
[1706497:MainThread](2021-12-07 14:08:30,302) INFO - qlib.workflow - [record_temp.py:466] - 指标分析记录 'indicator_analysis_5min
ute.pkl' 已作为实验 2 的工件保存
'The following are analysis results of indicators(5minute).'
        value
ffr  0.991017
pa   0.000000
pos  0.000000
[1706497:MainThread](2021-12-07 14:08:30,627) INFO - qlib.timer - [log.py:113] - 时间成本: 0.014s | 等待 `async_log` 完成
"""
from copy import deepcopy
import qlib
import fire
import pandas as pd
from qlib.constant import REG_CN
from qlib.config import HIGH_FREQ_CONFIG
from qlib.data import D
from qlib.utils import exists_qlib_data, init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.tests.data import GetData
from qlib.backtest import collect_data


class NestedDecisionExecutionWorkflow:
    """嵌套决策执行工作流"""
    market = "csi300"
    benchmark = "SH000300"
    data_handler_config = {
        "start_time": "2008-01-01",
        "end_time": "2021-05-31",
        "fit_start_time": "2008-01-01",
        "fit_end_time": "2014-12-31",
        "instruments": market,
    }

    task = {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.0421,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 20,
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": data_handler_config,
                },
                "segments": {
                    "train": ("2007-01-01", "2014-12-31"),
                    "valid": ("2015-01-01", "2016-12-31"),
                    "test": ("2020-01-01", "2021-05-31"),
                },
            },
        },
    }

    exp_name = "nested"

    port_analysis_config = {
        "executor": {
            "class": "NestedExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "inner_executor": {
                    "class": "NestedExecutor",
                    "module_path": "qlib.backtest.executor",
                    "kwargs": {
                        "time_per_step": "30min",
                        "inner_executor": {
                            "class": "SimulatorExecutor",
                            "module_path": "qlib.backtest.executor",
                            "kwargs": {
                                "time_per_step": "5min",
                                "generate_portfolio_metrics": True,
                                "verbose": True,
                                "indicator_config": {
                                    "show_indicator": True,
                                },
                            },
                        },
                        "inner_strategy": {
                            "class": "TWAPStrategy",
                            "module_path": "qlib.contrib.strategy.rule_strategy",
                        },
                        "generate_portfolio_metrics": True,
                        "indicator_config": {
                            "show_indicator": True,
                        },
                    },
                },
                "inner_strategy": {
                    "class": "SBBStrategyEMA",
                    "module_path": "qlib.contrib.strategy.rule_strategy",
                    "kwargs": {
                        "instruments": market,
                        "freq": "1min",
                    },
                },
                "track_data": True,
                "generate_portfolio_metrics": True,
                "indicator_config": {
                    "show_indicator": True,
                },
            },
        },
        "backtest": {
            "start_time": "2020-09-20",
            "end_time": "2021-05-20",
            "account": 100000000,
            "exchange_kwargs": {
                "freq": "1min",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }

    def _init_qlib(self):
        """初始化 qlib"""
        provider_uri_day = "~/.qlib/qlib_data/cn_data"  # 目标目录
        GetData().qlib_data(target_dir=provider_uri_day, region=REG_CN, version="v2", exists_skip=True)
        provider_uri_1min = HIGH_FREQ_CONFIG.get("provider_uri")
        GetData().qlib_data(
            target_dir=provider_uri_1min, interval="1min", region=REG_CN, version="v2", exists_skip=True
        )
        provider_uri_map = {"1min": provider_uri_1min, "day": provider_uri_day}
        qlib.init(provider_uri=provider_uri_map, dataset_cache=None, expression_cache=None)

    def _train_model(self, model, dataset):
        """训练模型"""
        with R.start(experiment_name=self.exp_name):
            R.log_params(**flatten_dict(self.task))
            model.fit(dataset)
            R.save_objects(**{"params.pkl": model})

            # 预测
            recorder = R.get_recorder()
            sr = SignalRecord(model, dataset, recorder)
            sr.generate()

    def backtest(self):
        """回测"""
        self._init_qlib()
        model = init_instance_by_config(self.task["model"])
        dataset = init_instance_by_config(self.task["dataset"])
        self._train_model(model, dataset)
        strategy_config = {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": 50,
                "n_drop": 5,
            },
        }
        self.port_analysis_config["strategy"] = strategy_config
        self.port_analysis_config["backtest"]["benchmark"] = self.benchmark

        with R.start(experiment_name=self.exp_name, resume=True):
            recorder = R.get_recorder()
            par = PortAnaRecord(
                recorder,
                self.port_analysis_config,
                indicator_analysis_method="value_weighted",
            )
            par.generate()

    def collect_data(self):
        """收集数据"""
        self._init_qlib()
        model = init_instance_by_config(self.task["model"])
        dataset = init_instance_by_config(self.task["dataset"])
        self._train_model(model, dataset)
        executor_config = self.port_analysis_config["executor"]
        backtest_config = self.port_analysis_config["backtest"]
        backtest_config["benchmark"] = self.benchmark
        strategy_config = {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": 50,
                "n_drop": 5,
            },
        }
        data_generator = collect_data(executor=executor_config, strategy=strategy_config, **backtest_config)
        for trade_decision in data_generator:
            print(trade_decision)

    def check_diff_freq(self):
        """检查不同频率"""
        self._init_qlib()
        exp = R.get_exp(experiment_name="backtest")
        rec = next(iter(exp.list_recorders().values()))  # 假设这将获取最新的记录器
        for check_key in "account", "total_turnover", "total_cost":
            check_key = "total_cost"

            acc_dict = {}
            for freq in ["30minute", "5minute", "1day"]:
                acc_dict[freq] = rec.load_object(f"portfolio_analysis/report_normal_{freq}.pkl")[check_key]
            acc_df = pd.DataFrame(acc_dict)
            acc_resam = acc_df.resample("1d").last().dropna()
            assert (acc_resam["30minute"] == acc_resam["1day"]).all()

    def backtest_only_daily(self):
        """
        此回测用于比较嵌套执行和单层执行
        由于日级和分钟级数据质量较低，它们很难比较。
        因此，它用于检测导致结果差异很大的严重错误。
        """
        self._init_qlib()
        model = init_instance_by_config(self.task["model"])
        dataset = init_instance_by_config(self.task["dataset"])
        self._train_model(model, dataset)
        strategy_config = {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": 50,
                "n_drop": 5,
            },
        }
        pa_conf = deepcopy(self.port_analysis_config)
        pa_conf["strategy"] = strategy_config
        pa_conf["executor"] = {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
                "verbose": True,
            },
        }
        pa_conf["backtest"]["benchmark"] = self.benchmark

        with R.start(experiment_name=self.exp_name, resume=True):
            recorder = R.get_recorder()
            par = PortAnaRecord(recorder, pa_conf)
            par.generate()


if __name__ == "__main__":
    fire.Fire(NestedDecisionExecutionWorkflow)
