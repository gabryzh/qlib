# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
这个例子是关于如何基于滚动任务模拟 OnlineManager 的。
"""
from pprint import pprint
import fire
import qlib
from qlib.model.trainer import DelayTrainerR, DelayTrainerRM, TrainerR, TrainerRM
from qlib.workflow import R
from qlib.workflow.online.manager import OnlineManager
from qlib.workflow.online.strategy import RollingStrategy
from qlib.workflow.task.gen import RollingGen
from qlib.workflow.task.manage import TaskManager
from qlib.tests.config import CSI100_RECORD_LGB_TASK_CONFIG_ONLINE, CSI100_RECORD_XGBOOST_TASK_CONFIG_ONLINE
import pandas as pd
from qlib.contrib.evaluate import backtest_daily
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy


class OnlineSimulationExample:
    """在线模拟示例"""
    def __init__(
        self,
        provider_uri="~/.qlib/qlib_data/cn_data",
        region="cn",
        exp_name="rolling_exp",
        task_url="mongodb://10.0.0.4:27017/",  # 使用 TrainerR 或 DelayTrainerR 时不是必需的
        task_db_name="rolling_db",  # 使用 TrainerR 或 DelayTrainerR 时不是必需的
        task_pool="rolling_task",
        rolling_step=80,
        start_time="2018-09-10",
        end_time="2018-10-31",
        tasks=None,
        trainer="TrainerR",
    ):
        """
        初始化 OnlineManagerExample。

        参数:
            provider_uri (str, optional): provider uri。默认为 "~/.qlib/qlib_data/cn_data"。
            region (str, optional): 股票区域。默认为 "cn"。
            exp_name (str, optional): 实验名称。默认为 "rolling_exp"。
            task_url (str, optional): 您的 MongoDB url。默认为 "mongodb://10.0.0.4:27017/"。
            task_db_name (str, optional): 数据库名称。默认为 "rolling_db"。
            task_pool (str, optional): 任务池名称 (任务池是 MongoDB 中的一个集合)。默认为 "rolling_task"。
            rolling_step (int, optional): 滚动的步长。默认为 80。
            start_time (str, optional): 模拟的开始时间。默认为 "2018-09-10"。
            end_time (str, optional): 模拟的结束时间。默认为 "2018-10-31"。
            tasks (dict or list[dict]): 一组等待滚动和训练的任务配置
        """
        if tasks is None:
            tasks = [CSI100_RECORD_XGBOOST_TASK_CONFIG_ONLINE, CSI100_RECORD_LGB_TASK_CONFIG_ONLINE]
        self.exp_name = exp_name
        self.task_pool = task_pool
        self.start_time = start_time
        self.end_time = end_time
        mongo_conf = {
            "task_url": task_url,
            "task_db_name": task_db_name,
        }
        qlib.init(provider_uri=provider_uri, region=region, mongo=mongo_conf)
        self.rolling_gen = RollingGen(
            step=rolling_step, rtype=RollingGen.ROLL_SD, ds_extra_mod_func=None
        )  # 滚动任务生成器，ds_extra_mod_func 为 None，因为我们只需要模拟到 2018-10-31，不需要更改处理程序的结束时间。
        if trainer == "TrainerRM":
            self.trainer = TrainerRM(self.exp_name, self.task_pool)
        elif trainer == "TrainerR":
            self.trainer = TrainerR(self.exp_name)
        else:
            raise NotImplementedError(f"不支持此类型的输入")
        self.rolling_online_manager = OnlineManager(
            RollingStrategy(exp_name, task_template=tasks, rolling_gen=self.rolling_gen),
            trainer=self.trainer,
            begin_time=self.start_time,
        )
        self.tasks = tasks

    # 将所有内容重置为初始状态，注意保存重要数据
    def reset(self):
        """重置"""
        if isinstance(self.trainer, TrainerRM):
            TaskManager(self.task_pool).remove()
        exp = R.get_exp(experiment_name=self.exp_name)
        for rid in exp.list_recorders():
            exp.delete_recorder(rid)

    # 运行此项以自动运行所有工作流
    def main(self):
        """主函数"""
        print("========== 重置 ==========")
        self.reset()
        print("========== 模拟 ==========")
        self.rolling_online_manager.simulate(end_time=self.end_time)
        print("========== 收集结果 ==========")
        print(self.rolling_online_manager.get_collector()())
        print("========== 信号 ==========")
        signals = self.rolling_online_manager.get_signals()
        print(signals)
        # 回测
        CSI300_BENCH = "SH000903"
        STRATEGY_CONFIG = {
            "topk": 30,
            "n_drop": 3,
            "signal": signals.to_frame("score"),
        }
        strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
        report_normal, positions_normal = backtest_daily(
            start_time=signals.index.get_level_values("datetime").min(),
            end_time=signals.index.get_level_values("datetime").max(),
            strategy=strategy_obj,
        )
        analysis = dict()
        analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"])
        analysis["excess_return_with_cost"] = risk_analysis(
            report_normal["return"] - report_normal["bench"] - report_normal["cost"]
        )

        analysis_df = pd.concat(analysis)  # type: pd.DataFrame
        pprint(analysis_df)

    def worker(self):
        """工作进程"""
        print("========== 工作进程 ==========")
        if isinstance(self.trainer, TrainerRM):
            self.trainer.worker()
        else:
            print(f"{type(self.trainer)} 不支持工作进程。")


if __name__ == "__main__":
    fire.Fire(OnlineSimulationExample)
