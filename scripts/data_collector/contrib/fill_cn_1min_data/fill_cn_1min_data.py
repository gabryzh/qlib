# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import fire
import qlib
import pandas as pd
from tqdm import tqdm
from qlib.data import D
from loguru import logger

# 将上三级目录添加到系统路径，以便导入utils模块
CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent.parent))
from data_collector.utils import generate_minutes_calendar_from_daily


def get_date_range(data_1min_dir: Path, max_workers: int = 16, date_field_name: str = "date") -> tuple:
    """
    遍历指定目录下的所有1分钟CSV文件，找出所有文件中的最早和最晚日期。

    Parameters
    ----------
    data_1min_dir: Path
        存放1分钟线CSV数据的目录。
    max_workers: int
        用于并行读取文件的最大线程数。
    date_field_name: str
        CSV文件中表示日期的列名。

    Returns
    -------
    tuple:
        (最早日期, 最晚日期)
    """
    csv_files = list(data_1min_dir.glob("*.csv"))
    min_date = None
    max_date = None

    # 使用线程池并行读取CSV文件以提高效率
    with tqdm(total=len(csv_files), desc="正在确定日期范围") as p_bar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for _file, _result in zip(csv_files, executor.map(pd.read_csv, csv_files)):
                if not _result.empty:
                    _dates = pd.to_datetime(_result[date_field_name])

                    _tmp_min = _dates.min()
                    min_date = min(min_date, _tmp_min) if min_date is not None else _tmp_min

                    _tmp_max = _dates.max()
                    max_date = max(max_date, _tmp_max) if max_date is not None else _tmp_max
                p_bar.update()
    return min_date, max_date


def get_symbols(data_1min_dir: Path) -> list:
    """
    从目录中的CSV文件名获取所有股票代码。

    Parameters
    ----------
    data_1min_dir: Path
        存放1分钟线CSV数据的目录。

    Returns
    -------
    list:
        股票代码列表。
    """
    return [f.stem.upper() for f in data_1min_dir.glob("*.csv")]


def fill_1min_using_1d(
    data_1min_dir: [str, Path],
    qlib_data_1d_dir: [str, Path],
    max_workers: int = 16,
    date_field_name: str = "date",
    symbol_field_name: str = "symbol",
):
    """
    使用日线数据来填充分钟线数据中缺失的股票。
    对于在日线数据中存在、但在分钟线数据目录中不存在的股票，
    此函数会为其创建一个空的分钟线CSV文件，其中只包含完整的分钟时间戳和股票代码。

    Parameters
    ----------
    data_1min_dir: str or Path
        1分钟线CSV数据目录。
    qlib_data_1d_dir: str or Path
        Qlib格式的日线二进制数据目录。
    max_workers: int
        最大工作线程/进程数。
    date_field_name: str
        日期列名。
    symbol_field_name: str
        股票代码列名。
    """
    data_1min_dir = Path(data_1min_dir).expanduser().resolve()
    qlib_data_1d_dir = Path(qlib_data_1d_dir).expanduser().resolve()

    # 1. 确定现有分钟线数据的时间范围
    min_date, max_date = get_date_range(data_1min_dir, max_workers, date_field_name)
    if min_date is None or max_date is None:
        logger.error("无法从分钟线数据目录中确定日期范围，请检查目录是否为空或文件格式是否正确。")
        return

    # 2. 获取现有的分钟线股票列表
    symbols_1min = get_symbols(data_1min_dir)

    # 3. 初始化Qlib，加载日线数据
    qlib.init(provider_uri=str(qlib_data_1d_dir))
    data_1d = D.features(D.instruments("all"), ["$close"], start_time=min_date, end_time=max_date, freq="day")
    symbols_1d = set(data_1d.index.get_level_values(level="instrument").unique())

    # 4. 找出在日线数据中存在但在分钟线数据中缺失的股票
    miss_symbols = symbols_1d - set(symbols_1min)
    if not miss_symbols:
        logger.info("分钟线数据的股票列表比日线数据更全或相等，无需填充。")
        return

    logger.info(f"发现 {len(miss_symbols)} 只缺失的股票，将为其生成空的分钟线数据文件。")
    logger.info(f"缺失列表: {miss_symbols}")

    # 读取一个已有的分钟线文件作为模板，以获取列的顺序
    template_df = pd.read_csv(next(data_1min_dir.glob("*.csv")))
    columns = template_df.columns
    # 判断股票代码是应该大写还是小写
    is_lower = template_df.loc[template_df[symbol_field_name].first_valid_index(), symbol_field_name].islower()

    # 5. 为每只缺失的股票生成空的分钟线文件
    for symbol in tqdm(miss_symbols, desc="正在填充缺失的分钟线数据"):
        symbol_case_correct = symbol.lower() if is_lower else symbol.upper()

        # 获取该股票在指定范围内的日线交易日历
        try:
            index_1d = data_1d.loc(axis=0)[symbol.upper()].index
        except KeyError:
            logger.warning(f"股票 {symbol} 在日线数据中未找到，跳过。")
            continue

        # 根据日线日历生成完整的分钟线时间戳索引
        index_1min = generate_minutes_calendar_from_daily(index_1d)
        index_1min.name = date_field_name

        # 创建一个空的DataFrame，并设置好列和索引
        _df = pd.DataFrame(columns=columns, index=index_1min)
        if date_field_name in _df.columns:
            del _df[date_field_name] # 如果列名和索引名重复，则删除列
        _df.reset_index(inplace=True)

        # 填充股票代码和默认的paused_num
        _df[symbol_field_name] = symbol_case_correct
        _df["paused_num"] = 0 # 默认为0，表示未停牌

        # 保存文件
        _df.to_csv(data_1min_dir.joinpath(f"{symbol_case_correct}.csv"), index=False)


if __name__ == "__main__":
    fire.Fire(fill_1min_using_1d)
