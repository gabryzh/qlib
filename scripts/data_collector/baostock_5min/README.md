## 收集数据

### 方法一：直接获取处理好的Qlib数据（二进制格式）

这是最简单快捷的方式，直接下载已经由官方处理好的数据。

-   **命令**: `python scripts/get_data.py qlib_data`
-   **参数说明**:
    -   `target_dir`: 数据保存目录, 默认为 *~/.qlib/qlib_data/cn_data_5min*。
    -   `version`: 数据集版本, 可选值为 [`v2`], 默认为 `v2`。
        -   `v2` 版本的结束日期是 *2022年12月*。
    -   `interval`: 数据频率, 此处应为 `5min`。
    -   `region`: 市场区域, 此处应为 `hs300` (沪深300)。
    -   `delete_old`: 是否删除 `target_dir` 下已有的旧数据 (*features, calendars, instruments, dataset_cache, features_cache*), 可选值为 [`True`, `False`], 默认为 `True`。
    -   `exists_skip`: 如果目标目录 `target_dir` 已存在数据，则跳过下载, 可选值为 [`True`, `False`], 默认为 `False`。
-   **示例**:
    ```bash
    # 下载沪深300的5分钟数据
    python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/hs300_data_5min --region hs300 --interval 5min
    ```

### 方法二：手动从Baostock收集原始数据并转换为Qlib格式

> **适用场景**: 当你需要获取官方未提供的最新数据时。
>
> **流程**: 该方法分为三步：1. 下载原始数据 -> 2. 规范化数据 -> 3. 转换为Qlib二进制格式。

#### 第1步: 下载原始数据到CSV文件

**命令**: `python scripts/data_collector/baostock_5min/collector.py download_data`

**说明**: 这一步会从baostock下载原始的5分钟K线数据（包含日期、代码、开高低收、成交量、成交额、复权状态等字段），并为每只股票保存一个CSV文件。

-   **参数说明**:
    -   `source_dir`: 保存CSV文件的目录。
    -   `interval`: 数据频率, 此处为 `5min`。
    -   `region`: 市场区域, 此处为 `HS300` (沪深300)。
    -   `start`: 开始日期, 默认为 *None*。
    -   `end`: 结束日期, 默认为 *None*。
-   **示例**:
    ```bash
    # 下载2022年1月的沪深300成分股5分钟数据
    python collector.py download_data --source_dir ~/.qlib/stock_data/source/hs300_5min_original --start 2022-01-01 --end 2022-01-30 --interval 5min --region HS300
    ```

#### 第2步: 规范化数据

**命令**: `python scripts/data_collector/baostock_5min/collector.py normalize_data`

**说明**: 这一步会对原始CSV数据进行处理，主要包括：
1.  使用日线数据的复权因子，计算分钟线的前复权价格。
2.  添加停牌信息。
3.  与完整的交易日历对齐。

-   **参数说明**:
    -   `source_dir`: 第1步中保存的原始CSV文件目录。
    -   `normalize_dir`: 规范化后数据的保存目录。
    -   `interval`: 数据频率, 此处为 `5min`。
        > **注意**: 当 `interval` 为 `5min` 时, 必须提供 `qlib_data_1d_dir` 参数。
    -   `region`: 市场区域, 此处为 `HS300`。
    -   `date_field_name`: CSV文件中表示时间的列名, 默认为 `date`。
    -   `symbol_field_name`: CSV文件中表示股票代码的列名, 默认为 `symbol`。
    -   `end_date`: 如果提供，则只处理到该日期（包含）的数据, 默认为 `None`。
    -   `qlib_data_1d_dir`: **（必需）** Qlib格式的日线数据目录。规范化5分钟数据需要使用日线数据的复权因子和停牌信息。
        -   你可以通过以下命令获取日线数据:
            ```bash
            python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --interval 1d --region cn --version v3
            ```
-   **示例**:
    ```bash
    # 规范化5分钟数据
    python collector.py normalize_data --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source/hs300_5min_original --normalize_dir ~/.qlib/stock_data/source/hs300_5min_nor --region HS300 --interval 5min
    ```

#### 第3步: 将规范化数据转换为Qlib二进制格式

**命令**: `python scripts/dump_bin.py dump_all`

**说明**: 这一步会将第2步生成的规范化CSV文件转换为Qlib专用的二进制格式（`.bin`），这种格式可以被Qlib高效读取。

-   **参数说明**:
    -   `csv_path`: **（必需）** 第2步生成的规范化数据目录 (`normalize_dir`)。
    -   `qlib_dir`: **（必需）** 最终生成的Qlib二进制数据的存放目录。
    -   `freq`: 数据频率, 日线为 `day`, 5分钟线为 `5min`。
    -   `max_workers`: 并行转换的进程数, 默认为 *16*。
    -   `include_fields`: 需要包含的字段, 默认为 `""` (表示所有字段)。
    -   `exclude_fields`: 需要排除的字段, 默认为 `""`。
    -   `symbol_field_name`: CSV中的股票代码列名, 默认为 `symbol`。
    -   `date_field_name`: CSV中的日期列名, 默认为 `date`。
-   **示例**:
    ```bash
    # 将规范化后的5分钟数据转换为二进制格式
    python dump_bin.py dump_all --csv_path ~/.qlib/stock_data/source/hs300_5min_nor --qlib_dir ~/.qlib/qlib_data/hs300_5min_bin --freq 5min --exclude_fields date,symbol
    ```
