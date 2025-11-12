#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# 导入fire库，用于创建命令行界面
import fire
# 从qlib的测试数据模块中导入GetData类
# GetData类封装了从网络下载qlib示例数据的功能
from qlib.tests.data import GetData


if __name__ == "__main__":
    # 使用fire库将GetData类暴露为命令行工具。
    # 这样，用户就可以在命令行中运行 `python get_data.py` 加上GetData类中定义的方法（例如 `qlib_data`）
    # 来下载示例数据。
    # 例如：`python get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data`
    fire.Fire(GetData)
