# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
from pathlib import Path
from typing import Union

import fire

from qlib import auto_init
from qlib.contrib.rolling.ddgda import DDGDA
from qlib.tests.data import GetData

DIRNAME = Path(__file__).absolute().resolve().parent
BENCH_DIR = DIRNAME.parent / "baseline"


class DDGDABench(DDGDA):
    """DDGDA 基准测试"""
    # README.md 中的配置
    CONF_LIST = [
        BENCH_DIR / "workflow_config_linear_Alpha158.yaml",
        BENCH_DIR / "workflow_config_lightgbm_Alpha158.yaml",
    ]

    DEFAULT_CONF = CONF_LIST[0]  # 由于效率，默认为线性

    def __init__(self, conf_path: Union[str, Path] = DEFAULT_CONF, horizon=20, **kwargs) -> None:
        """
        初始化 DDGDABench。

        :param conf_path: 配置文件路径。
        :param horizon: 预测范围。
        """
        # 此代码用于与以前的旧代码兼容
        conf_path = Path(conf_path)
        super().__init__(conf_path=conf_path, horizon=horizon, working_dir=DIRNAME, **kwargs)

        for f in self.CONF_LIST:
            if conf_path.samefile(f):
                break
        else:
            self.logger.warning("模型类型不在基准测试中！")


if __name__ == "__main__":
    kwargs = {}
    if os.environ.get("PROVIDER_URI", "") == "":
        GetData().qlib_data(exists_skip=True)
    else:
        kwargs["provider_uri"] = os.environ["PROVIDER_URI"]
    auto_init(**kwargs)
    fire.Fire(DDGDABench)
