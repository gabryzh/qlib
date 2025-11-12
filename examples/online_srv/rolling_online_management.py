# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
这个例子展示了 OnlineManager 如何处理滚动任务。
共有四个部分，包括第一次训练、例行程序 1、添加策略和例行程序 2。
首先，OnlineManager 将完成第一次训练，并将训练好的模型设置为“在线”模型。
接下来，OnlineManager 将完成一个例行过程，包括更新在线预测 -> 准备任务 -> 准备新模型 -> 准备信号
然后，我们将向 OnlineManager 添加一些新策略。这将完成新策略的第一次训练。
最后，OnlineManager 将完成第二个例行程序并更新所有策略。
"""
import os
import fire
import qlib
from qlib.model.trainer import DelayTrainerR, DelayTrainerRM, TrainerR, TrainerRM, end_task_train, task_train
from qlib.workflow import R
from qlib.workflow.online.strategy import RollingStrategy
from qlib.workflow.task.gen import RollingGen
from qlib.workflow.online.manager import OnlineManager
from qlib.tests.config import CSI100_RECORD_XGBOOST_TASK_CONFIG_ROLLING, CSI100_RECORD_LGB_TASK_CONFIG_ROLLING
from qlib.workflow.task.manage import TaskManager


class RollingOnlineExample:
    """滚动在线示例"""
    def __init__(
        self,
        provider_uri="~/.qlib/qlib_data/cn_data",
        region="cn",
        trainer=DelayTrainerRM(),  # 您可以从 TrainerR、TrainerRM、DelayTrainerR、DelayTrainerRM 中选择
        task_url="mongodb://10.0.0.4:27017/",  # 使用 TrainerR 或 DelayTrainerR 时不是必需的
        task_db_name="rolling_db",  # 使用 TrainerR 或 DelayTrainerR 时不是必需的
        rolling_step=550,
        tasks=None,
        add_tasks=None,
    ):
        if add_tasks is None:
            add_tasks = [CSI100_RECORD_LGB_TASK_CONFIG_ROLLING]
        if tasks is None:
            tasks = [CSI100_RECORD_XGBOOST_TASK_CONFIG_ROLLING]
        mongo_conf = {
            "task_url": task_url,  # 您的 MongoDB url
            "task_db_name": task_db_name,  # 数据库名称
        }
        qlib.init(provider_uri=provider_uri, region=region, mongo=mongo_conf)
        self.tasks = tasks
        self.add_tasks = add_tasks
        self.rolling_step = rolling_step
        strategies = []
        for task in tasks:
            name_id = task["model"]["class"]  # 注意：假设：模型类只能指定一个策略
            strategies.append(
                RollingStrategy(
                    name_id,
                    task,
                    RollingGen(step=rolling_step, rtype=RollingGen.ROLL_SD),
                )
            )
        self.trainer = trainer
        self.rolling_online_manager = OnlineManager(strategies, trainer=self.trainer)

    _ROLLING_MANAGER_PATH = (
        ".RollingOnlineExample"  # OnlineManager 将转储到此文件，以便在调用例行程序时可以加载它。
    )

    def worker(self):
        """工作进程"""
        # 通过其他进程或机器训练任务以实现多进程
        print("========== 工作进程 ==========")
        if isinstance(self.trainer, TrainerRM):
            for task in self.tasks + self.add_tasks:
                name_id = task["model"]["class"]
                self.trainer.worker(experiment_name=name_id)
        else:
            print(f"{type(self.trainer)} 不支持工作进程。")

    # 将所有内容重置为初始状态，注意保存重要数据
    def reset(self):
        """重置"""
        for task in self.tasks + self.add_tasks:
            name_id = task["model"]["class"]
            TaskManager(task_pool=name_id).remove()
            exp = R.get_exp(experiment_name=name_id)
            for rid in exp.list_recorders():
                exp.delete_recorder(rid)

        if os.path.exists(self._ROLLING_MANAGER_PATH):
            os.remove(self._ROLLING_MANAGER_PATH)

    def first_run(self):
        """首次运行"""
        print("========== 重置 ==========")
        self.reset()
        print("========== 首次运行 ==========")
        self.rolling_online_manager.first_train()
        print("========== 收集结果 ==========")
        print(self.rolling_online_manager.get_collector()())
        print("========== 转储 ==========")
        self.rolling_online_manager.to_pickle(self._ROLLING_MANAGER_PATH)

    def routine(self):
        """例行程序"""
        print("========== 加载 ==========")
        self.rolling_online_manager = OnlineManager.load(self._ROLLING_MANAGER_PATH)
        print("========== 例行程序 ==========")
        self.rolling_online_manager.routine()
        print("========== 收集结果 ==========")
        print(self.rolling_online_manager.get_collector()())
        print("========== 信号 ==========")
        print(self.rolling_online_manager.get_signals())
        print("========== 转储 ==========")
        self.rolling_online_manager.to_pickle(self._ROLLING_MANAGER_PATH)

    def add_strategy(self):
        """添加策略"""
        print("========== 加载 ==========")
        self.rolling_online_manager = OnlineManager.load(self._ROLLING_MANAGER_PATH)
        print("========== 添加策略 ==========")
        strategies = []
        for task in self.add_tasks:
            name_id = task["model"]["class"]  # 注意：假设：模型类只能指定一个策略
            strategies.append(
                RollingStrategy(
                    name_id,
                    task,
                    RollingGen(step=self.rolling_step, rtype=RollingGen.ROLL_SD),
                )
            )
        self.rolling_online_manager.add_strategy(strategies=strategies)
        print("========== 转储 ==========")
        self.rolling_online_manager.to_pickle(self._ROLLING_MANAGER_PATH)

    def main(self):
        """主函数"""
        self.first_run()
        self.routine()
        self.add_strategy()
        self.routine()


if __name__ == "__main__":
    fire.Fire(RollingOnlineExample)
