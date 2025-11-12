# 导入所需的库
from loguru import logger  # 用于日志记录
import os  # 用于操作系统相关的功能，如文件路径
from typing import Optional  # 用于类型提示

import fire  # 用于创建命令行界面
import pandas as pd  # 用于数据处理和分析
import qlib  # qlib库
from tqdm import tqdm  # 用于显示进度条

from qlib.data import D  # 从qlib.data中导入D，用于数据访问


class DataHealthChecker:
    """
    检查数据集的数据完整性和正确性。数据将被转换为pd.DataFrame并检查以下问题：
    - 是否缺少["open", "high", "low", "close", "volume"]中的任何一列
    - 是否有任何数据缺失
    - OHLCV（开盘价、最高价、最低价、收盘价、成交量）列中的任何阶跃变化是否超过阈值（价格默认为0.5，成交量默认为3）
    - 是否缺少任何因子
    """

    def __init__(
        self,
        csv_path=None,  # CSV文件路径
        qlib_dir=None,  # qlib数据目录
        freq="day",  # 数据频率，默认为"day"
        large_step_threshold_price=0.5,  # 价格大幅度变化的阈值，默认为0.5
        large_step_threshold_volume=3,  # 成交量大幅度变化的阈值，默认为3
        missing_data_num=0,  # 允许的缺失数据数量，默认为0
    ):
        # 断言：csv_path或qlib_dir必须提供一个
        assert csv_path or qlib_dir, "必须提供 csv_path 或 qlib_dir 中的一个。"
        # 断言：csv_path和qlib_dir不能同时提供
        assert not (csv_path and qlib_dir), "只能提供 csv_path 或 qlib_dir 中的一个。"

        self.data = {}  # 用于存储数据的字典
        self.problems = {}  # 用于存储问题的字典
        self.freq = freq  # 数据频率
        self.large_step_threshold_price = large_step_threshold_price  # 价格大幅度变化的阈值
        self.large_step_threshold_volume = large_step_threshold_volume  # 成交量大幅度变化的阈值
        self.missing_data_num = missing_data_num  # 允许的缺失数据数量

        if csv_path:
            # 如果提供了csv_path
            assert os.path.isdir(csv_path), f"{csv_path} 应该是一个目录。"
            # 获取目录下所有以.csv结尾的文件
            files = [f for f in os.listdir(csv_path) if f.endswith(".csv")]
            # 遍历文件并加载数据
            for filename in tqdm(files, desc="正在加载数据"):
                df = pd.read_csv(os.path.join(csv_path, filename))
                self.data[filename] = df

        elif qlib_dir:
            # 如果提供了qlib_dir
            # 初始化qlib
            qlib.init(provider_uri=qlib_dir)
            # 加载qlib数据
            self.load_qlib_data()

    def load_qlib_data(self):
        """加载qlib数据"""
        # 获取所有市场的股票代码
        instruments = D.instruments(market="all")
        # 获取所有股票代码的列表
        instrument_list = D.list_instruments(instruments=instruments, as_list=True, freq=self.freq)
        # 需要的字段
        required_fields = ["$open", "$close", "$low", "$high", "$volume", "$factor"]
        # 遍历所有股票代码，获取特征数据
        for instrument in instrument_list:
            df = D.features([instrument], required_fields, freq=self.freq)
            # 重命名字段
            df.rename(
                columns={
                    "$open": "open",
                    "$close": "close",
                    "$low": "low",
                    "$high": "high",
                    "$volume": "volume",
                    "$factor": "factor",
                },
                inplace=True,
            )
            # 存储数据
            self.data[instrument] = df
        print(df)  # 打印最后一个数据帧

    def check_missing_data(self) -> Optional[pd.DataFrame]:
        """检查DataFrame中是否有任何数据缺失。"""
        # 初始化结果字典
        result_dict = {
            "instruments": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }
        # 遍历所有数据
        for filename, df in self.data.items():
            # 找到缺失数据超过阈值的列
            missing_data_columns = df.isnull().sum()[df.isnull().sum() > self.missing_data_num].index.tolist()
            # 如果有缺失数据的列
            if len(missing_data_columns) > 0:
                # 记录文件名和各列的缺失数据数量
                result_dict["instruments"].append(filename)
                result_dict["open"].append(df.isnull().sum()["open"])
                result_dict["high"].append(df.isnull().sum()["high"])
                result_dict["low"].append(df.isnull().sum()["low"])
                result_dict["close"].append(df.isnull().sum()["close"])
                result_dict["volume"].append(df.isnull().sum()["volume"])

        # 将结果转换为DataFrame
        result_df = pd.DataFrame(result_dict).set_index("instruments")
        if not result_df.empty:
            # 如果结果不为空，返回结果
            return result_df
        else:
            # 如果结果为空，记录日志并返回None
            logger.info(f"✅ 没有缺失数据。")
            return None

    def check_large_step_changes(self) -> Optional[pd.DataFrame]:
        """检查OHLCV列中是否有超过阈值的大幅度阶跃变化。"""
        # 初始化结果字典
        result_dict = {
            "instruments": [],
            "col_name": [],
            "date": [],
            "pct_change": [],
        }
        # 遍历所有数据
        for filename, df in self.data.items():
            affected_columns = []  # 受影响的列
            # 遍历OHLCV列
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    # 计算百分比变化
                    pct_change = df[col].pct_change(fill_method=None).abs()
                    # 根据列名设置阈值
                    threshold = self.large_step_threshold_volume if col == "volume" else self.large_step_threshold_price
                    # 如果最大百分比变化超过阈值
                    if pct_change.max() > threshold:
                        # 找到超过阈值的阶跃
                        large_steps = pct_change[pct_change > threshold]
                        # 记录文件名、列名、日期和百分比变化
                        result_dict["instruments"].append(filename)
                        result_dict["col_name"].append(col)
                        result_dict["date"].append(large_steps.index.to_list()[0][1].strftime("%Y-%m-%d"))
                        result_dict["pct_change"].append(pct_change.max())
                        affected_columns.append(col)

        # 将结果转换为DataFrame
        result_df = pd.DataFrame(result_dict).set_index("instruments")
        if not result_df.empty:
            # 如果结果不为空，返回结果
            return result_df
        else:
            # 如果结果为空，记录日志并返回None
            logger.info(f"✅ OHLCV列中没有超过阈值的大幅度阶跃变化。")
            return None

    def check_required_columns(self) -> Optional[pd.DataFrame]:
        """检查DataFrame中是否缺少任何必需的列（OLHCV）。"""
        required_columns = ["open", "high", "low", "close", "volume"]  # 必需的列
        # 初始化结果字典
        result_dict = {
            "instruments": [],
            "missing_col": [],
        }
        # 遍历所有数据
        for filename, df in self.data.items():
            # 检查是否所有必需的列都存在
            if not all(column in df.columns for column in required_columns):
                # 找到缺失的列
                missing_required_columns = [column for column in required_columns if column not in df.columns]
                # 记录文件名和缺失的列
                result_dict["instruments"].append(filename)
                result_dict["missing_col"] += missing_required_columns

        # 将结果转换为DataFrame
        result_df = pd.DataFrame(result_dict).set_index("instruments")
        if not result_df.empty:
            # 如果结果不为空，返回结果
            return result_df
        else:
            # 如果结果为空，记录日志并返回None
            logger.info(f"✅ 列（OLHCV）是完整的，没有缺失。")
            return None

    def check_missing_factor(self) -> Optional[pd.DataFrame]:
        """检查DataFrame中是否缺少'factor'列。"""
        # 初始化结果字典
        result_dict = {
            "instruments": [],
            "missing_factor_col": [],
            "missing_factor_data": [],
        }
        # 遍历所有数据
        for filename, df in self.data.items():
            # 跳过特定的指数文件
            if "000300" in filename or "000903" in filename or "000905" in filename:
                continue
            # 如果'factor'列不存在
            if "factor" not in df.columns:
                result_dict["instruments"].append(filename)
                result_dict["missing_factor_col"].append(True)
            # 如果'factor'列的数据全部为空
            if df["factor"].isnull().all():
                if filename in result_dict["instruments"]:
                    result_dict["missing_factor_data"].append(True)
                else:
                    result_dict["instruments"].append(filename)
                    result_dict["missing_factor_col"].append(False)
                    result_dict["missing_factor_data"].append(True)

        # 将结果转换为DataFrame
        result_df = pd.DataFrame(result_dict).set_index("instruments")
        if not result_df.empty:
            # 如果结果不为空，返回结果
            return result_df
        else:
            # 如果结果为空，记录日志并返回None
            logger.info(f"✅ `factor`列已存在且不为空。")
            return None

    def check_data(self):
        """运行所有检查"""
        # 运行各种检查
        check_missing_data_result = self.check_missing_data()
        check_large_step_changes_result = self.check_large_step_changes()
        check_required_columns_result = self.check_required_columns()
        check_missing_factor_result = self.check_missing_factor()
        # 如果任何检查发现问题
        if (
            check_large_step_changes_result is not None
            or check_large_step_changes_result is not None
            or check_required_columns_result is not None
            or check_missing_factor_result is not None
        ):
            # 打印摘要
            print(f"\n数据健康检查摘要 ({len(self.data)} 个文件已检查):")
            print("-------------------------------------------------")
            # 打印每个检查的结果
            if isinstance(check_missing_data_result, pd.DataFrame):
                logger.warning(f"存在缺失数据。")
                print(check_missing_data_result)
            if isinstance(check_large_step_changes_result, pd.DataFrame):
                logger.warning(f"OHLCV列存在大幅度阶跃变化。")
                print(check_large_step_changes_result)
            if isinstance(check_required_columns_result, pd.DataFrame):
                logger.warning(f"列（OLHCV）缺失。")
                print(check_required_columns_result)
            if isinstance(check_missing_factor_result, pd.DataFrame):
                logger.warning(f"factor列不存在或为空")
                print(check_missing_factor_result)


if __name__ == "__main__":
    # 使用fire库创建命令行界面
    fire.Fire(DataHealthChecker)
