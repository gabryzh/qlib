# 使用日线数据补全分钟线数据中缺失的股票

## 简介

本脚本用于解决一个常见问题：当你的分钟线数据（通常是CSV格式）和日线数据（Qlib二进制格式）的股票列表不一致时，即某些股票只有日线数据而没有分钟线数据，你可以使用此脚本来为这些缺失的股票生成一个“空的”分钟线数据文件。

这样做可以确保在进行量化分析时，分钟线级别的数据集拥有和日线级别一样完整的股票列表，避免因数据缺失导致的问题。

生成的“空”文件将包含：
-   基于该股票日线交易日历生成的完整分钟线时间戳。
-   股票代码。
-   其他字段将为空（NaN）。

## 环境要求

在运行脚本之前，请先安装所需的依赖包：

```bash
pip install -r requirements.txt
```

## 使用方法

通过以下命令运行脚本：

```bash
python fill_cn_1min_data.py fill_1min_using_1d --data_1min_dir <你的1分钟CSV数据目录> --qlib_data_1d_dir <你的Qlib日线数据目录>
```

### 示例

```bash
python fill_cn_1min_data.py fill_1min_using_1d --data_1min_dir ~/.qlib/csv_data/cn_data_1min --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data
```

## 参数说明

-   `data_1min_dir`: **（必需）** 存放CSV格式的1分钟线数据的目录。脚本会读取此目录下的文件以确定现有的股票列表和时间范围。
-   `qlib_data_1d_dir`: **（必需）** 存放Qlib二进制格式的日线数据的目录。脚本会从这里获取完整的股票列表和日线交易日历。
-   `max_workers`: 并行处理文件的最大线程数, 默认为 `16`。
-   `date_field_name`: CSV文件中的日期/时间列名, 默认为 `date`。
-   `symbol_field_name`: CSV文件中的股票代码列名, 默认为 `symbol`。
