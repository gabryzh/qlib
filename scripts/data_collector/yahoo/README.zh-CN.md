
- [收集器数据](#收集器数据)
  - [获取Qlib数据](#获取qlib数据bin文件)
  - [将雅虎财经数据收集到qlib](#将雅虎财经数据收集到qlib)
  - [每日频率数据的自动更新](#每日频率数据的自动更新来自雅虎财经)
- [使用qlib数据](#使用qlib数据)


# 从雅虎财经收集数据

> *请**注意**，数据是从[雅虎财经](https://finance.yahoo.com/lookup)收集的，数据可能并不完美。如果用户有高质量的数据集，我们建议他们准备自己的数据。更多信息，用户可以参考[相关文档](https://qlib.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-qlib-format)*

**注意**：雅虎财经已屏蔽来自中国的访问。如果您想使用雅虎数据爬虫，请更换您的网络。

>  **异常数据示例**

- [SH000661](https://finance.yahoo.com/quote/000661.SZ/history?period1=1558310400&period2=1590796800&interval=1d&filter=history&frequency=1d)
- [SZ300144](https://finance.yahoo.com/quote/300144.SZ/history?period1=1557446400&period2=1589932800&interval=1d&filter=history&frequency=1d)

我们已经考虑了**股价调整**，但一些价格序列似乎仍然非常异常。

## 依赖要求

```bash
pip install -r requirements.txt
```

## 收集器数据

### 获取Qlib数据(`bin文件`)
  > 来自*雅虎财经*的`qlib-data`是已经转储并且可以在`qlib`中直接使用的数据。
  > 这个现成的qlib-data不会定期更新。如果用户想要最新的数据，请按照[这些步骤](#将雅虎财经数据收集到qlib)下载最新数据。

  - 获取数据: `python scripts/get_data.py qlib_data`
  - 参数:
    - `target_dir`: 保存目录, 默认为 *~/.qlib/qlib_data/cn_data*
    - `version`: 数据集版本, 可选值为 [`v1`, `v2`], 默认为 `v1`
      - `v2` 的结束日期是 *2021-06*, `v1` 的结束日期是 *2020-09*
      - 如果用户想增量更新数据，他们需要使用雅虎收集器从头开始[收集数据](#将雅虎财经数据收集到qlib)。
      - **qlib的[基准测试](https://github.com/microsoft/qlib/tree/main/examples/benchmarks)使用`v1`**，*由于雅虎财经对历史数据的不稳定访问，`v2`和`v1`之间存在一些差异*
    - `interval`: `1d` 或 `1min`, 默认为 `1d`
    - `region`: `cn` 或 `us` 或 `in`, 默认为 `cn`
    - `delete_old`: 从`target_dir`删除现有数据(*features, calendars, instruments, dataset_cache, features_cache*), 可选值为 [`True`, `False`], 默认为 `True`
    - `exists_skip`: `target_dir`数据已存在，跳过`get_data`, 可选值为 [`True`, `False`], 默认为 `False`
  - 示例:
    ```bash
    # A股 1d
    python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
    # A股 1min
    python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --region cn --interval 1min
    # 美股 1d
    python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us --interval 1d
    ```

### 将*雅虎财经*数据收集到qlib
> 收集*雅虎财经*数据并*转储*为`qlib`格式。
> 如果上述现成数据无法满足用户需求，用户可以按照本节内容抓取最新数据并将其转换为qlib-data。
  1. 将数据下载为csv: `python scripts/data_collector/yahoo/collector.py download_data`

     这将从雅虎下载原始数据，例如最高价、最低价、开盘价、收盘价、复权收盘价到本地目录。每个股票代码一个文件。

     - 参数:
          - `source_dir`: 保存目录
          - `interval`: `1d` 或 `1min`, 默认为 `1d`
            > **由于*雅虎财经API*的限制，`1min`数据仅提供最近一个月的数据**
          - `region`: `CN` 或 `US` 或 `IN` 或 `BR`, 默认为 `CN`
          - `delay`: `time.sleep(delay)`, 默认为 *0.5*
          - `start`: 开始日期时间, 默认为 *"2000-01-01"*; *闭区间(包括开始)*
          - `end`: 结束日期时间, 默认为 `pd.Timestamp(datetime.datetime.now() + pd.Timedelta(days=1))`; *开区间(不包括结束)*
          - `max_workers`: 获取并发符号的数量，为了保持符号数据的完整性，不建议更改此参数, 默认为 *1*
          - `check_data_length`: 检查每个*符号*的行数, 默认为 `None`
            > 如果 `len(symbol_df) < check_data_length`, 将会重新获取，重新获取的次数来自`max_collector_count`参数
          - `max_collector_count`: *“失败”*符号重试的次数, 默认为 2
     - 示例:
          ```bash
          # A股 1d 数据
          python collector.py download_data --source_dir ~/.qlib/stock_data/source/cn_data --start 2020-01-01 --end 2020-12-31 --delay 1 --interval 1d --region CN
          # A股 1min 数据
          python collector.py download_data --source_dir ~/.qlib/stock_data/source/cn_data_1min --delay 1 --interval 1min --region CN

          # 美股 1d 数据
          python collector.py download_data --source_dir ~/.qlib/stock_data/source/us_data --start 2020-01-01 --end 2020-12-31 --delay 1 --interval 1d --region US
          # 美股 1min 数据
          python collector.py download_data --source_dir ~/.qlib/stock_data/source/us_data_1min --delay 1 --interval 1min --region US

          # 印度 1d 数据
          python collector.py download_data --source_dir ~/.qlib/stock_data/source/in_data --start 2020-01-01 --end 2020-12-31 --delay 1 --interval 1d --region IN
          # 印度 1min 数据
          python collector.py download_data --source_dir ~/.qlib/stock_data/source/in_data_1min --delay 1 --interval 1min --region IN

          # 巴西 1d 数据
          python collector.py download_data --source_dir ~/.qlib/stock_data/source/br_data --start 2003-01-03 --end 2022-03-01 --delay 1 --interval 1d --region BR
          # 巴西 1min 数据
          python collector.py download_data --source_dir ~/.qlib/stock_data/source/br_data_1min --delay 1 --interval 1min --region BR
          ```
  2. 规范化数据: `python scripts/data_collector/yahoo/collector.py normalize_data`

     这将：
     1. 使用复权收盘价规范化最高价、最低价、收盘价、开盘价。
     2. 规范化最高价、最低价、收盘价、开盘价，使得第一个有效交易日的收盘价为1。

     - 参数:
          - `source_dir`: csv 目录
          - `normalize_dir`: 结果目录
          - `max_workers`: 并发数, 默认为 *1*
          - `interval`: `1d` 或 `1min`, 默认为 `1d`
            > 如果 **`interval == 1min`**, `qlib_data_1d_dir` 不能为空
          - `region`: `CN` 或 `US` 或 `IN`, 默认为 `CN`
          - `date_field_name`: csv文件中标识时间的列*名*, 默认为 `date`
          - `symbol_field_name`: csv文件中标识符号的列*名*, 默认为 `symbol`
          - `end_date`: 如果不为`None`，则规范化保存的最后日期（*包括end_date*）；如果为`None`，则将忽略此参数；默认为 `None`
          - `qlib_data_1d_dir`: qlib目录(1d数据)
            ```
            如果 interval==1min, qlib_data_1d_dir 不能为空，规范化 1min 需要使用 1d 数据;

                qlib_data_1d 可以这样获取:
                    $ python scripts/get_data.py qlib_data --target_dir <qlib_data_1d_dir> --interval 1d
                    $ python scripts/data_collector/yahoo/collector.py update_data_to_bin --qlib_data_1d_dir <qlib_data_1d_dir> --end_date <end_date>
                或者:
                    从雅虎财经下载1d数据

            ```
      - 示例:
        ```bash
        # 规范化 1d A股
        python collector.py normalize_data --source_dir ~/.qlib/stock_data/source/cn_data --normalize_dir ~/.qlib/stock_data/source/cn_1d_nor --region CN --interval 1d

        # 规范化 1min A股
        python collector.py normalize_data --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source/cn_data_1min --normalize_dir ~/.qlib/stock_data/source/cn_1min_nor --region CN --interval 1min

        # 规范化 1d 巴西
        python scripts/data_collector/yahoo/collector.py normalize_data --source_dir ~/.qlib/stock_data/source/br_data --normalize_dir ~/.qlib/stock_data/source/br_1d_nor --region BR --interval 1d

        # 规范化 1min 巴西
        python collector.py normalize_data --qlib_data_1d_dir ~/.qlib/qlib_data/br_data --source_dir ~/.qlib/stock_data/source/br_data_1min --normalize_dir ~/.qlib/stock_data/source/br_1min_nor --region BR --interval 1min
        ```
  3. 转储数据: `python scripts/dump_bin.py dump_all`

     这将把`feature`目录中的规范化csv转换为numpy数组，并将规范化数据存储为每个列一个文件，每个符号一个目录。

     - 参数:
       - `data_path`: 股票数据路径或目录, **规范化结果(normalize_dir)**
       - `qlib_dir`: qlib(转储)数据目录
       - `freq`: 交易频率, 默认为 `day`
         > `freq_map = {1d:day, 1mih: 1min}`
       - `max_workers`: 线程数, 默认为 *16*
       - `include_fields`: 转储字段, 默认为 `""`
       - `exclude_fields`: 未转储的字段, 默认为 `"""
         > dump_fields = `include_fields if include_fields else set(symbol_df.columns) - set(exclude_fields) exclude_fields else symbol_df.columns`
       - `symbol_field_name`: csv文件中标识符号的列*名*, 默认为 `symbol`
       - `date_field_name`: csv文件中标识时间的列*名*, 默认为 `date`
       - `file_suffix`: 股票数据文件格式, 默认为 ".csv"
     - 示例:
       ```bash
       # 转储 1d A股
       python dump_bin.py dump_all --data_path ~/.qlib/stock_data/source/cn_1d_nor --qlib_dir ~/.qlib/qlib_data/cn_data --freq day --exclude_fields date,symbol --file_suffix .csv
       # 转储 1min A股
       python dump_bin.py dump_all --data_path ~/.qlib/stock_data/source/cn_1min_nor --qlib_dir ~/.qlib/qlib_data/cn_data_1min --freq 1min --exclude_fields date,symbol --file_suffix .csv
       ```

### 每日频率数据的自动更新(来自雅虎财经)
  > 建议用户手动更新一次数据（--trading_date 2021-05-25），然后将其设置为自动更新。
  >
  > **注意**: 用户不能基于Qlib提供的离线数据增量更新数据（为减小数据大小，某些字段已被删除）。用户应使用[雅虎收集器](https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#automatic-update-of-daily-frequency-datafrom-yahoo-finance)从头下载雅虎数据，然后增量更新。
  >

  * 每个交易日自动将数据更新到“qlib”目录（Linux）
      * 使用 *crontab*: `crontab -e`
      * 设置定时任务:

        ```
        * * * * 1-5 python <script path> update_data_to_bin --qlib_data_1d_dir <user data dir>
        ```
        * **脚本路径**: *scripts/data_collector/yahoo/collector.py*

  * 手动更新数据
      ```
      python scripts/data_collector/yahoo/collector.py update_data_to_bin --qlib_data_1d_dir <user data dir> --end_date <end date>
      ```
      * `end_date`: 交易日结束(不包括)
      * `check_data_length`: 检查每个*符号*的行数, 默认为 `None`
        > 如果 `len(symbol_df) < check_data_length`, 将会重新获取，重新获取的次数来自`max_collector_count`参数

  * `scripts/data_collector/yahoo/collector.py update_data_to_bin` 参数:
      * `source_dir`: 保存从互联网收集的原始数据的目录，默认为 "Path(__file__).parent/source"
      * `normalize_dir`: 规范化数据的目录，默认为 "Path(__file__).parent/normalize"
      * `qlib_data_1d_dir`: 要更新的雅虎qlib数据，通常来自: [下载qlib数据](https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data)
      * `end_date`: 结束日期时间, 默认为 ``pd.Timestamp(trading_date + pd.Timedelta(days=1))``; 开区间(不包括结束)
      * `region`: 区域, 可选值为 ["CN", "US"], 默认为 "CN"
      * `interval`: 间隔, 默认为 "1d"(目前仅支持1d数据)
      * `exists_skip`: 存在则跳过, 默认为 False

## 使用qlib数据

  ```python
  import qlib
  from qlib.data import D

  # 1d 数据 A股
  # freq=day, freq 默认为 day
  qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
  df = D.features(D.instruments("all"), ["$close"], freq="day")

  # 1min 数据 A股
  # freq=1min
  qlib.init(provider_uri="~/.qlib/qlib_data/cn_data_1min", region="cn")
  inst = D.list_instruments(D.instruments("all"), freq="1min", as_list=True)
  # 获取 100 个符号
  df = D.features(inst[:100], ["$close"], freq="1min")
  # 获取所有符号数据
  # df = D.features(D.instruments("all"), ["$close"], freq="1min")

  # 1d 数据 美股
  qlib.init(provider_uri="~/.qlib/qlib_data/us_data", region="us")
  df = D.features(D.instruments("all"), ["$close"], freq="day")

  # 1min 数据 美股
  qlib.init(provider_uri="~/.qlib/qlib_data/us_data_1min", region="cn")
  inst = D.list_instruments(D.instruments("all"), freq="1min", as_list=True)
  # 获取 100 个符号
  df = D.features(inst[:100], ["$close"], freq="1min")
  # 获取所有符号数据
  # df = D.features(D.instruments("all"), ["$close"], freq="1min")
  ```
