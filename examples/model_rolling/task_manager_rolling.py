# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
这个例子展示了 TrainerRM 如何基于带有滚动任务的 TaskManager 工作。
训练后，如何在 task_collecting 中收集滚动结果将被展示出来。
基于 TaskManager 的能力，`worker` 方法提供了一种简单的多进程方式。
"""

from pprint import pprint

import fire
import qlib
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.workflow.task.gen import RollingGen, task_generator
from qlib.workflow.task.manage import TaskManager, run_task
from qlib.workflow.task.collect import RecorderCollector
from qlib.model.ens.group import RollingGroup
from qlib.model.trainer import TrainerR, TrainerRM, task_train
from qlib.tests.config import CSI100_RECORD_LGB_TASK_CONFIG, CSI100_RECORD_XGBOOST_TASK_CONFIG


class RollingTaskExample:
    """滚动任务示例"""
    def __init__(
        self,
        provider_uri="~/.qlib/qlib_data/cn_data",
        region=REG_CN,
        task_url="mongodb://10.0.0.4:27017/",
        task_db_name="rolling_db",
        experiment_name="rolling_exp",
        task_pool=None,  # 如果用户想要 "rolling_task"
        task_config=None,
        rolling_step=550,
        rolling_type=RollingGen.ROLL_SD,
    ):
        # TaskManager 配置
        if task_config is None:
            task_config = [CSI100_RECORD_XGBOOST_TASK_CONFIG, CSI100_RECORD_LGB_TASK_CONFIG]
        mongo_conf = {
            "task_url": task_url,
            "task_db_name": task_db_name,
        }
        qlib.init(provider_uri=provider_uri, region=region, mongo=mongo_conf)
        self.experiment_name = experiment_name
        if task_pool is None:
            self.trainer = TrainerR(experiment_name=self.experiment_name)
        else:
            self.task_pool = task_pool
            self.trainer = TrainerRM(self.experiment_name, self.task_pool)
        self.task_config = task_config
        self.rolling_gen = RollingGen(step=rolling_step, rtype=rolling_type)

    # 将所有内容重置为初始状态，注意保存重要数据
    def reset(self):
        """重置"""
        print("========== 重置 ==========")
        if isinstance(self.trainer, TrainerRM):
            TaskManager(task_pool=self.task_pool).remove()
        exp = R.get_exp(experiment_name=self.experiment_name)
        for rid in exp.list_recorders():
            exp.delete_recorder(rid)

    def task_generating(self):
        """生成任务"""
        print("========== 生成任务 ==========")
        tasks = task_generator(
            tasks=self.task_config,
            generators=self.rolling_gen,  # 生成不同的日期段
        )
        pprint(tasks)
        return tasks

    def task_training(self, tasks):
        """训练任务"""
        print("========== 训练任务 ==========")
        self.trainer.train(tasks)

    def worker(self):
        """工作进程"""
        # 注意：这仅用于 TrainerRM
        # 通过其他进程或机器训练任务以实现多进程。它与 TrainerRM.worker 相同。
        print("========== 工作进程 ==========")
        run_task(task_train, self.task_pool, experiment_name=self.experiment_name)

    def task_collecting(self):
        """收集任务"""
        print("========== 收集任务 ==========")

        def rec_key(recorder):
            task_config = recorder.load_object("task")
            model_key = task_config["model"]["class"]
            rolling_key = task_config["dataset"]["kwargs"]["segments"]["test"]
            return model_key, rolling_key

        def my_filter(recorder):
            # 只选择 "LGBModel" 的结果
            model_key, rolling_key = rec_key(recorder)
            if model_key == "LGBModel":
                return True
            return False

        collector = RecorderCollector(
            experiment=self.experiment_name,
            process_list=RollingGroup(),
            rec_key_func=rec_key,
            rec_filter_func=my_filter,
        )
        print(collector())

    def main(self):
        """主函数"""
        self.reset()
        tasks = self.task_generating()
        self.task_training(tasks)
        self.task_collecting()


if __name__ == "__main__":
    ## 要使用您自己的参数查看整个过程，请使用以下命令
    # python task_manager_rolling.py main --experiment_name="your_exp_name"
    fire.Fire(RollingTaskExample)
