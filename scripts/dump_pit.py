# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
待办事项:
- 需要一个设计更完善的PIT数据库。
    - 需要分离的插入、删除、更新、查询操作。
"""

# 导入shutil模块，用于文件操作
import shutil
# 导入struct模块，用于处理二进制数据
import struct
# 从pathlib导入Path类，用于面向对象的文件系统路径
from pathlib import Path
# 从typing导入Iterable，用于类型注解
from typing import Iterable
# 从functools导入partial，用于创建偏函数
from functools import partial
# 导入concurrent.futures，用于并发执行
from concurrent.futures import ProcessPoolExecutor

# 导入fire库，用于创建命令行界面
import fire
# 导入pandas库，用于数据处理
import pandas as pd
# 导入tqdm库，用于显示进度条
from tqdm import tqdm
# 导入loguru库，用于日志记录
from loguru import logger
# 从qlib.utils导入工具函数
from qlin.utils import fname_to_code, get_period_offset
# 从qlib.config导入配置C
from qlib.config import C


class DumpPitData:
    """
    转储Point-in-Time (PIT) 数据的类。
    PIT数据通常指财务数据，其特点是会在发布后进行多次修订。
    此脚本将CSV格式的PIT数据转换为qlib特定的二进制格式。
    """
    # 存放PIT数据的目录名
    PIT_DIR_NAME = "financial"
    # 输入的CSV文件的分隔符
    PIT_CSV_SEP = ","
    # 数据文件的后缀
    DATA_FILE_SUFFIX = ".data"
    # 索引文件的后缀
    INDEX_FILE_SUFFIX = ".index"

    # 数据频率：季度
    INTERVAL_quarterly = "quarterly"
    # 数据频率：年度
    INTERVAL_annual = "annual"

    # 从qlib配置中读取各种数据类型的二进制格式
    # 报告期（如20220331）的数据类型
    PERIOD_DTYPE = C.pit_record_type["period"]
    # 索引（指向数据文件中的位置）的数据类型
    INDEX_DTYPE = C.pit_record_type["index"]
    # 数据记录的完整数据类型，包括发布日期、报告期、值和下一个修订版本的索引
    DATA_DTYPE = "".join(
        [
            C.pit_record_type["date"],
            C.pit_record_type["period"],
            C.pit_record_type["value"],
            C.pit_record_type["index"],
        ]
    )
    # 表示空索引的特殊值
    NA_INDEX = C.pit_record_nan["index"]

    # 计算各种数据类型的大小（字节数）
    INDEX_DTYPE_SIZE = struct.calcsize(INDEX_DTYPE)
    PERIOD_DTYPE_SIZE = struct.calcsize(PERIOD_DTYPE)
    DATA_DTYPE_SIZE = struct.calcsize(DATA_DTYPE)

    # 运行模式
    UPDATE_MODE = "update"
    ALL_MODE = "all"

    def __init__(
        self,
        csv_path: str,
        qlib_dir: str,
        backup_dir: str = None,
        freq: str = "quarterly",
        max_workers: int = 16,
        date_column_name: str = "date",
        period_column_name: str = "period",
        value_column_name: str = "value",
        field_column_name: str = "field",
        file_suffix: str = ".csv",
        exclude_fields: str = "",
        include_fields: str = "",
        limit_nums: int = None,
    ):
        """
        初始化。

        Parameters
        ----------
        csv_path: str
            PIT数据的CSV文件路径或目录。
        qlib_dir: str
            qlib数据（转储后）的目录。
        backup_dir: str, default None
            如果提供，则备份qlib_dir。
        freq: str, default "quarterly"
            数据频率, "quarterly" 或 "annual"。
        max_workers: int, default 16
            最大工作进程数。
        date_column_name: str, default "date"
            CSV中发布日期的列名。
        period_column_name: str, default "period"
            CSV中报告期的列名。
        value_column_name: str, default "value"
            CSV中值的列名。
        field_column_name: str, default "field"
            CSV中字段名（如'revenue', 'net_profit'）的列名。
        file_suffix: str, default ".csv"
            数据文件后缀。
        exclude_fields: str
            需要排除的字段。
        include_fields: str
            需要包含的字段。
        limit_nums: int
            用于调试，限制处理的文件数量。
        """
        csv_path = Path(csv_path).expanduser()
        # 处理排除和包含字段
        if isinstance(exclude_fields, str):
            exclude_fields = exclude_fields.split(",")
        if isinstance(include_fields, str):
            include_fields = include_fields.split(",")
        self._exclude_fields = tuple(filter(lambda x: len(x) > 0, map(str.strip, exclude_fields)))
        self._include_fields = tuple(filter(lambda x: len(x) > 0, map(str.strip, include_fields)))
        self.file_suffix = file_suffix
        # 获取所有源CSV文件的路径
        self.csv_files = sorted(csv_path.glob(f"*{self.file_suffix}") if csv_path.is_dir() else [csv_path])
        if limit_nums is not None:
            self.csv_files = self.csv_files[: int(limit_nums)]

        self.qlib_dir = Path(qlib_dir).expanduser()
        self.backup_dir = backup_dir if backup_dir is None else Path(backup_dir).expanduser()
        if backup_dir is not None:
            self._backup_qlib_dir(Path(backup_dir).expanduser())

        self.works = max_workers
        self.date_column_name = date_column_name
        self.period_column_name = period_column_name
        self.value_column_name = value_column_name
        self.field_column_name = field_column_name

        self._mode = self.ALL_MODE

    def _backup_qlib_dir(self, target_dir: Path):
        """备份qlib目录。"""
        shutil.copytree(str(self.qlib_dir.resolve()), str(target_dir.resolve()))

    def get_source_data(self, file_path: Path) -> pd.DataFrame:
        """读取并预处理源CSV文件。"""
        df = pd.read_csv(str(file_path.resolve()), low_memory=False)
        # 将值列转换为float32
        df[self.value_column_name] = df[self.value_column_name].astype("float32")
        # 将日期列转换为YYYYMMDD格式的整数
        df[self.date_column_name] = df[self.date_column_name].str.replace("-", "").astype("int32")
        return df

    def get_symbol_from_file(self, file_path: Path) -> str:
        """从文件名中提取股票代码。"""
        return fname_to_code(file_path.name[: -len(self.file_suffix)].strip().lower())

    def get_dump_fields(self, df: pd.DataFrame) -> Iterable[str]:
        """根据include和exclude规则，确定需要转储的字段。"""
        all_fields = set(df[self.field_column_name])
        if self._include_fields:
            return set(self._include_fields)
        elif self._exclude_fields:
            return all_fields - set(self._exclude_fields)
        else:
            return all_fields

    def get_filenames(self, symbol: str, field: str, interval: str):
        """根据股票代码、字段和频率生成数据文件和索引文件的路径。"""
        dir_name = self.qlib_dir.joinpath(self.PIT_DIR_NAME, symbol)
        dir_name.mkdir(parents=True, exist_ok=True)
        return (
            dir_name.joinpath(f"{field}_{interval[0]}{self.DATA_FILE_SUFFIX}".lower()),
            dir_name.joinpath(f"{field}_{interval[0]}{self.INDEX_FILE_SUFFIX}".lower()),
        )

    def _dump_pit(
        self,
        file_path: Path,
        interval: str = "quarterly",
        overwrite: bool = False,
    ):
        """
        转储单个CSV文件的数据。
        数据格式如下:
            `<field>.data` 文件:
                [发布日期, 报告期, 值, 下一个修订版的索引]
                [发布日期, 报告期, 值, 下一个修订版的索引]
                [...]
            `<field>.index` 文件:
                [起始年份, 索引, 索引, ...]

        `<field.data>` 包含了按时间点顺序排列的数据：`报告期`的`值`在`发布日期`公布，
        其后续的修订值可以通过`下一个修订版的索引`（类似链表）找到。

        `<field>.index` 包含了每个报告期（季度或年度）的值在.data文件中的起始索引。
        为了节省空间，我们只存储`起始年份`，因为后续的报告期可以很容易地推断出来。

        Parameters
        ----------
        file_path: Path
            输入的csv文件路径
        interval: str
            数据频率 ("quarterly" or "annual")
        overwrite: bool
            是否覆盖现有数据或仅更新
        """
        symbol = self.get_symbol_from_file(file_path)
        df = self.get_source_data(file_path)
        if df.empty:
            logger.warning(f"{symbol} 文件为空")
            return

        # 遍历该文件中的所有字段（如收入、利润等）
        for field in self.get_dump_fields(df):
            # 筛选出当前字段的数据，并按发布日期排序
            df_sub = df.query(f'{self.field_column_name}=="{field}"').sort_values(self.date_column_name)
            if df_sub.empty:
                logger.warning(f"{symbol} 的字段 {field} 为空")
                continue
            data_file, index_file = self.get_filenames(symbol, field, interval)

            ## 计算起始和结束报告期
            start_period = df_sub[self.period_column_name].min()
            end_period = df_sub[self.period_column_name].max()

            # 将报告期转换为年份
            start_year = start_period // 100 if interval == self.INTERVAL_quarterly else start_period
            end_year = end_period // 100 if interval == self.INTERVAL_quarterly else end_period

            # 如果不是覆盖模式且索引文件已存在，调整起始年份
            if not overwrite and index_file.exists():
                with open(index_file, "rb") as fi:
                    (first_year,) = struct.unpack(self.PERIOD_DTYPE, fi.read(self.PERIOD_DTYPE_SIZE))
                    n_records = len(fi.read()) // self.INDEX_DTYPE_SIZE
                    n_years = n_records // 4 if interval == self.INTERVAL_quarterly else n_records
                    start_year = first_year + n_years
            else:
                # 否则，创建新的索引文件并写入起始年份
                with open(index_file, "wb") as f:
                    f.write(struct.pack(self.PERIOD_DTYPE, start_year))
                first_year = start_year

            # 如果新数据的起始年份不晚于已存在的年份，则跳过
            if start_year > end_year:
                logger.warning(f"{symbol}-{field} 数据已存在，跳过")
                continue

            # 用NA值填充新的索引条目
            with open(index_file, "ab") as fi:
                for year in range(start_year, end_year + 1):
                    if interval == self.INTERVAL_quarterly:
                        fi.write(struct.pack(self.INDEX_DTYPE * 4, *[self.NA_INDEX] * 4))
                    else:
                        fi.write(struct.pack(self.INDEX_DTYPE, self.NA_INDEX))

            # 如果数据文件已存在且不是覆盖模式，则只处理比上次最后记录更新的数据
            if not overwrite and data_file.exists():
                with open(data_file, "rb") as fd:
                    fd.seek(-self.DATA_DTYPE_SIZE, 2) # 移动到文件末尾的最后一条记录
                    last_date, _, _, _ = struct.unpack(self.DATA_DTYPE, fd.read())
                df_sub = df_sub.query(f"{self.date_column_name}>{last_date}")
            else:
                # 否则，根据模式创建或清空数据文件
                with open(data_file, "wb+" if overwrite else "ab+"):
                    pass

            # 以读写二进制模式打开数据和索引文件
            with open(data_file, "rb+") as fd, open(index_file, "rb+") as fi:
                # 遍历筛选后的数据行
                for i, row in df_sub.iterrows():
                    # 计算当前报告期在索引文件中的偏移量
                    offset = get_period_offset(first_year, row.period, interval == self.INTERVAL_quarterly)

                    # 读取当前报告期的索引值
                    fi.seek(self.PERIOD_DTYPE_SIZE + self.INDEX_DTYPE_SIZE * offset)
                    (cur_index,) = struct.unpack(self.INDEX_DTYPE, fi.read(self.INDEX_DTYPE_SIZE))

                    # 情况一：这是该报告期的第一条数据
                    if cur_index == self.NA_INDEX:
                        # 将索引文件中的NA值更新为当前数据在.data文件中的位置
                        fi.seek(self.PERIOD_DTYPE_SIZE + self.INDEX_DTYPE_SIZE * offset)
                        fi.write(struct.pack(self.INDEX_DTYPE, fd.tell()))
                    # 情况二：该报告期已有数据（即财务数据修订）
                    else:
                        # 找到链表的末尾
                        _cur_fd_pos = fd.tell() # 记录当前数据要写入的位置
                        prev_index = self.NA_INDEX
                        while cur_index != self.NA_INDEX:
                            # 移动到上一条记录的`_next`字段
                            fd.seek(cur_index + self.DATA_DTYPE_SIZE - self.INDEX_DTYPE_SIZE)
                            prev_index = cur_index
                            # 读取`_next`字段的值，继续寻找
                            (cur_index,) = struct.unpack(self.INDEX_DTYPE, fd.read(self.INDEX_DTYPE_SIZE))
                        # 更新链表末尾记录的`_next`字段，使其指向新数据的位置
                        fd.seek(prev_index + self.DATA_DTYPE_SIZE - self.INDEX_DTYPE_SIZE)
                        fd.write(struct.pack(self.INDEX_DTYPE, _cur_fd_pos))
                        # 将文件指针移回准备写入新数据的位置
                        fd.seek(_cur_fd_pos)

                    # 写入新的数据记录（date, period, value, NA_INDEX）
                    fd.write(struct.pack(self.DATA_DTYPE, row.date, row.period, row.value, self.NA_INDEX))

    def dump(self, interval="quarterly", overwrite=False):
        """执行PIT数据的转储。"""
        logger.info("开始转储PIT数据......")
        _dump_func = partial(self._dump_pit, interval=interval, overwrite=overwrite)

        # 使用进程池并行处理所有CSV文件
        with tqdm(total=len(self.csv_files)) as p_bar:
            with ProcessPoolExecutor(max_workers=self.works) as executor:
                for _ in executor.map(_dump_func, self.csv_files):
                    p_bar.update()

    def __call__(self, *args, **kwargs):
        """使类的实例可调用，直接执行dump方法。"""
        self.dump()


if __name__ == "__main__":
    # 使用fire库将DumpPitData类暴露为命令行工具
    fire.Fire(DumpPitData)
