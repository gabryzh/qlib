#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import fire

import qlib
import pickle
from qlib.constant import REG_CN
from qlib.config import HIGH_FREQ_CONFIG

from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.ops import Operators
from qlib.data.data import Cal
from qlib.tests.data import GetData

from highfreq_ops import get_calendar_day, DayLast, FFillNan, BFillNan, Date, Select, IsNull, Cut


class HighfreqWorkflow:
    """高频数据工作流"""
    SPEC_CONF = {"custom_ops": [DayLast, FFillNan, BFillNan, Date, Select, IsNull, Cut], "expression_cache": None}

    MARKET = "all"

    start_time = "2020-09-15 00:00:00"
    end_time = "2021-01-18 16:00:00"
    train_end_time = "2020-11-30 16:00:00"
    test_start_time = "2020-12-01 00:00:00"

    DATA_HANDLER_CONFIG0 = {
        "start_time": start_time,
        "end_time": end_time,
        "fit_start_time": start_time,
        "fit_end_time": train_end_time,
        "instruments": MARKET,
        "infer_processors": [{"class": "HighFreqNorm", "module_path": "highfreq_processor"}],
    }
    DATA_HANDLER_CONFIG1 = {
        "start_time": start_time,
        "end_time": end_time,
        "instruments": MARKET,
    }

    task = {
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "HighFreqHandler",
                    "module_path": "highfreq_handler",
                    "kwargs": DATA_HANDLER_CONFIG0,
                },
                "segments": {
                    "train": (start_time, train_end_time),
                    "test": (
                        test_start_time,
                        end_time,
                    ),
                },
            },
        },
        "dataset_backtest": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "HighFreqBacktestHandler",
                    "module_path": "highfreq_handler",
                    "kwargs": DATA_HANDLER_CONFIG1,
                },
                "segments": {
                    "train": (start_time, train_end_time),
                    "test": (
                        test_start_time,
                        end_time,
                    ),
                },
            },
        },
    }

    def _init_qlib(self):
        """初始化 qlib"""
        # 使用 cn_data_1min 数据
        QLIB_INIT_CONFIG = {**HIGH_FREQ_CONFIG, **self.SPEC_CONF}
        provider_uri = QLIB_INIT_CONFIG.get("provider_uri")
        GetData().qlib_data(target_dir=provider_uri, interval="1min", region=REG_CN, exists_skip=True)
        qlib.init(**QLIB_INIT_CONFIG)

    def _prepare_calender_cache(self):
        """为缓存预加载日历"""

        # 此代码利用 Linux 的写时复制功能来避免在子进程中多次计算日历
        # 此代码可能会加速，但在 Windows 和 Mac Os 上可能无效
        Cal.calendar(freq="1min")
        get_calendar_day(freq="1min")

    def get_data(self):
        """使用数据集获取高频数据"""
        self._init_qlib()
        self._prepare_calender_cache()

        dataset = init_instance_by_config(self.task["dataset"])
        xtrain, xtest = dataset.prepare(["train", "test"])
        print(xtrain, xtest)

        dataset_backtest = init_instance_by_config(self.task["dataset_backtest"])
        backtest_train, backtest_test = dataset_backtest.prepare(["train", "test"])
        print(backtest_train, backtest_test)

        return

    def dump_and_load_dataset(self):
        """在磁盘上转储和加载数据集状态"""
        self._init_qlib()
        self._prepare_calender_cache()
        dataset = init_instance_by_config(self.task["dataset"])
        dataset_backtest = init_instance_by_config(self.task["dataset_backtest"])

        ##=============转储数据集=============
        dataset.to_pickle(path="dataset.pkl")
        dataset_backtest.to_pickle(path="dataset_backtest.pkl")

        del dataset, dataset_backtest
        ##=============重新加载数据集=============
        with open("dataset.pkl", "rb") as file_dataset:
            dataset = pickle.load(file_dataset)

        with open("dataset_backtest.pkl", "rb") as file_dataset_backtest:
            dataset_backtest = pickle.load(file_dataset_backtest)

        self._prepare_calender_cache()
        ##=============重新初始化数据集=============
        dataset.config(
            handler_kwargs={
                "start_time": "2021-01-19 00:00:00",
                "end_time": "2021-01-25 16:00:00",
            },
            segments={
                "test": (
                    "2021-01-19 00:00:00",
                    "2021-01-25 16:00:00",
                ),
            },
        )
        dataset.setup_data(
            handler_kwargs={
                "init_type": DataHandlerLP.IT_LS,
            },
        )
        dataset_backtest.config(
            handler_kwargs={
                "start_time": "2021-01-19 00:00:00",
                "end_time": "2021-01-25 16:00:00",
            },
            segments={
                "test": (
                    "2021-01-19 00:00:00",
                    "2021-01-25 16:00:00",
                ),
            },
        )
        dataset_backtest.setup_data(handler_kwargs={})

        ##=============获取数据=============
        xtest = dataset.prepare("test")
        backtest_test = dataset_backtest.prepare("test")

        print(xtest, backtest_test)
        return


if __name__ == "__main__":
    fire.Fire(HighfreqWorkflow)
