#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.


import qlib
from qlib.constant import REG_CN

from qlib.utils import init_instance_by_config
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_GBDT_TASK


if __name__ == "__main__":
    # 使用默认数据
    provider_uri = "~/.qlib/qlib_data/cn_data"  # 目标目录
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)

    qlib.init(provider_uri=provider_uri, region=REG_CN)

    ###################################
    # 训练模型
    ###################################
    # 模型初始化
    model = init_instance_by_config(CSI300_GBDT_TASK["model"])
    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])
    model.fit(dataset)

    # 获取模型特征重要性
    feature_importance = model.get_feature_importance()
    print("特征重要性:")
    print(feature_importance)
