# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# 导入Path类，用于处理文件路径
from pathlib import Path
# 导入ProcessPoolExecutor，用于并行处理
from concurrent.futures import ProcessPoolExecutor

# 导入qlib库
import qlib
# 从qlib.data中导入D，用于数据访问
from qlib.data import D

# 导入fire库，用于创建命令行界面
import fire
# 导入datacompy库，用于比较两个DataFrame
import datacompy
# 导入pandas库，用于数据处理
import pandas as pd
# 导入tqdm库，用于显示进度条
from tqdm import tqdm
# 导入loguru库，用于日志记录
from loguru import logger


class CheckBin:
    """
    CheckBin类用于检查转换后的bin文件是否与原始的csv文件数据一致。
    """
    # 定义比较结果的常量
    NOT_IN_FEATURES = "不在特征中"
    COMPARE_FALSE = "比较失败"
    COMPARE_TRUE = "比较成功"
    COMPARE_ERROR = "比较出错"

    def __init__(
        self,
        qlib_dir: str,
        csv_path: str,
        check_fields: str = None,
        freq: str = "day",
        symbol_field_name: str = "symbol",
        date_field_name: str = "date",
        file_suffix: str = ".csv",
        max_workers: int = 16,
    ):
        """
        初始化CheckBin类。

        Parameters
        ----------
        qlib_dir : str
            qlib数据目录的路径。
        csv_path : str
            原始csv文件或文件夹的路径。
        check_fields : str, optional
            需要检查的字段，以逗号分隔的字符串。默认为None，此时会检查qlib_dir/features/<first_dir>/*.<freq>.bin中的所有字段。
        freq : str, optional
            数据频率，可选值为 "day", "1m"。默认为 "day"。
        symbol_field_name: str, optional
            csv文件中的股票代码字段名。默认为 "symbol"。
        date_field_name: str, optional
            csv文件中的日期字段名。默认为 "date"。
        file_suffix: str, optional
            csv文件的后缀。默认为 ".csv"。
        max_workers: int, optional
            用于并行处理的最大工作进程数。默认为 16。
        """
        # 扩展用户目录（例如'~'）并转换为Path对象
        self.qlib_dir = Path(qlib_dir).expanduser()
        # 获取features目录下的所有子目录（即股票代码）
        bin_path_list = list(self.qlib_dir.joinpath("features").iterdir())
        # 获取所有股票代码并转换为小写，然后排序
        self.qlib_symbols = sorted(map(lambda x: x.name.lower(), bin_path_list))
        # 初始化qlib
        qlib.init(
            provider_uri=str(self.qlib_dir.resolve()),
            mount_path=str(self.qlib_dir.resolve()),
            auto_mount=False,
            redis_port=-1,
        )
        # 扩展用户目录并转换为Path对象
        csv_path = Path(csv_path).expanduser()
        # 获取所有csv文件的路径
        self.csv_files = sorted(csv_path.glob(f"*{file_suffix}") if csv_path.is_dir() else [csv_path])

        if check_fields is None:
            # 如果未指定检查字段，则自动从第一个股票的bin文件中获取
            check_fields = list(map(lambda x: x.name.split(".")[0], bin_path_list[0].glob(f"*.bin")))
        else:
            # 如果指定了检查字段，则按逗号分割
            check_fields = check_fields.split(",") if isinstance(check_fields, str) else check_fields
        # 去除字段名两端的空格
        self.check_fields = list(map(lambda x: x.strip(), check_fields))
        # 为字段名添加qlib所需的前缀"$"
        self.qlib_fields = list(map(lambda x: f"${x}", self.check_fields))
        self.max_workers = max_workers
        self.symbol_field_name = symbol_field_name
        self.date_field_name = date_field_name
        self.freq = freq
        self.file_suffix = file_suffix

    def _compare(self, file_path: Path):
        """
        比较单个csv文件和对应的qlib bin文件。

        Parameters
        ----------
        file_path : Path
            csv文件的路径。

        Returns
        -------
        str
            比较结果，值为 "not in features", "compare True", "compare False", "compare error" 之一。
        """
        # 从文件名中提取股票代码
        symbol = file_path.name.replace(self.file_suffix, "")
        # 检查股票代码是否存在于qlib数据中
        if symbol.lower() not in self.qlib_symbols:
            return self.NOT_IN_FEATURES

        # 加载qlib数据
        qlib_df = D.features([symbol], self.qlib_fields, freq=self.freq)
        # 重命名列，去掉"$"前缀
        qlib_df.rename(columns={_c: _c.strip("$") for _c in qlib_df.columns}, inplace=True)

        # 加载csv数据
        origin_df = pd.read_csv(file_path)
        # 将日期列转换为datetime对象
        origin_df[self.date_field_name] = pd.to_datetime(origin_df[self.date_field_name])
        # 如果csv中没有symbol列，则添加一列
        if self.symbol_field_name not in origin_df.columns:
            origin_df[self.symbol_field_name] = symbol
        # 设置索引为股票代码和日期
        origin_df.set_index([self.symbol_field_name, self.date_field_name], inplace=True)
        # 统一索引名称
        origin_df.index.names = qlib_de.index.names
        # 根据qlib数据的索引重新对齐csv数据，确保两者索引一致
        origin_df = origin_df.reindex(qlib_df.index)

        try:
            # 使用datacompy进行比较
            compare = datacompy.Compare(
                origin_df,
                qlib_df,
                on_index=True,
                abs_tol=1e-08,  # 绝对公差
                rel_tol=1e-05,  # 相对公差
                df1_name="Original",  # df1的名称
                df2_name="New",  # df2的名称
            )
            # 检查是否匹配
            _r = compare.matches(ignore_extra_columns=True)
            return self.COMPARE_TRUE if _r else self.COMPARE_FALSE
        except Exception as e:
            # 如果比较过程中发生异常，则记录警告并返回错误
            logger.warning(f"{symbol} 比较出错: {e}")
            return self.COMPARE_ERROR

    def check(self):
        """
        检查执行 ``dump_bin.py`` 后的bin文件是否与原始csv文件数据一致。
        """
        logger.info("开始检查......")

        # 初始化各种错误列表
        error_list = []
        not_in_features = []
        compare_false = []

        # 使用tqdm显示总进度
        with tqdm(total=len(self.csv_files)) as p_bar:
            # 使用进程池执行并行比较
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # 遍历每个csv文件并获取比较结果
                for file_path, _check_res in zip(self.csv_files, executor.map(self._compare, self.csv_files)):
                    symbol = file_path.name.replace(self.file_suffix, "")
                    # 根据比较结果将股票代码添加到相应的列表中
                    if _check_res == self.NOT_IN_FEATURES:
                        not_in_features.append(symbol)
                    elif _check_res == self.COMPARE_ERROR:
                        error_list.append(symbol)
                    elif _check_res == self.COMPARE_FALSE:
                        compare_false.append(symbol)
                    # 更新进度条
                    p_bar.update()

        logger.info("检查结束......")
        # 打印总结信息
        if error_list:
            logger.warning(f"比较出错的股票: {error_list}")
        if not_in_features:
            logger.warning(f"在qlib features中未找到的股票: {not_in_features}")
        if compare_false:
            logger.warning(f"数据不一致的股票: {compare_false}")
        logger.info(
            f"总计 {len(self.csv_files)} 个文件, {len(error_list)} 个出错, {len(not_in_features)} 个未找到, {len(compare_false)} 个不一致"
        )


if __name__ == "__main__":
    # 使用fire库将CheckBin类暴露为命令行工具
    fire.Fire(CheckBin)
