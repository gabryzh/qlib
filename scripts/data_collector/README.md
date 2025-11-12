# 数据收集器

## 简介

此目录包含用于数据收集的脚本。

- `yahoo`: 从 *雅虎财经 (Yahoo Finance)* 获取 *美股/A股* 的股票数据。
- `fund`: 从 *天天基金网 (http://fund.eastmoney.com)* 获取公募基金数据。
- `cn_index`: 从 *中证指数官网 (http://www.csindex.com.cn)* 获取 *A股指数* 的成分股信息，例如 *沪深300* / *中证100*。
- `us_index`: 从 *维基百科 (https://en.wikipedia.org/wiki)* 获取 *美股指数* 的成分股信息，例如 *标普500* / *纳斯达克100* / *道琼斯工业平均指数* / *标普400*。
- `contrib`: 提供一些辅助功能的脚本。


## 自定义数据收集

> 具体实现可参考雅虎财经的示例：https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo

如果你想添加自己的数据源，可以遵循以下步骤：

1. 在当前目录 (`data_collector`) 下为你的数据源创建一个新的文件夹。
2. 在新文件夹中添加 `collector.py` 文件。
   - 添加收集器类 (`Collector` class)，继承自 `BaseCollector`:
     ```python
     # 文件的开头部分，用于将上级目录添加到系统路径中
     CUR_DIR = Path(__file__).resolve().parent
     sys.path.append(str(CUR_DIR.parent.parent))
     from data_collector.base import BaseCollector, BaseNormalize, BaseRun

     class YourCollector(BaseCollector):
         # 在这里实现你的数据收集逻辑
         ...
     ```
   - 添加数据规范化类 (`Normalize` class)，继承自 `BaseNormalize`:
     ```python
     class YourNormalize(BaseNormalize):
         # 在这里实现你的数据规范化逻辑（例如，对齐交易日历，计算复权因子等）
         ...
     ```
   - 添加命令行接口类 (`CLI` class)，继承自 `BaseRun`:
     ```python
     class Run(BaseRun):
         # 这个类将你的收集器和规范化器连接起来，并通过命令行暴露出来
         ...
     ```
3. 添加 `README.md` 文件，说明你的数据源、用法和注意事项。
4. 添加 `requirements.txt` 文件，列出运行你的收集器所需的Python依赖包。


## Qlib数据目录结构说明

  | 类别 | 描述 |
  |---|---|
  | **Features (特征)** | 存放具体的行情和财务数据，通常是二进制格式。<br> **价格/成交量**: <br>&nbsp;&nbsp; - $close(收盘价)/$open(开盘价)/$low(最低价)/$high(最高价)/$volume(成交量)/$change(涨跌幅)/$factor(复权因子) |
  | **Calendar (日历)** | 存放交易日历文件。<br> **\<freq>.txt**: <br>&nbsp;&nbsp; - `day.txt` (日线交易日历)<br>&nbsp;&nbsp;  - `1min.txt` (分钟线交易日历) |
  | **Instruments (股票列表)** | 存放不同市场或指数的股票列表。<br> **\<market>.txt**: <br>&nbsp;&nbsp; - **必需**: `all.txt` (包含该市场所有股票的列表及其上市、退市日期); <br>&nbsp;&nbsp;  - 可选: `csi300.txt`/`csi500.txt`/`sp500.txt` (特定指数的成分股及其在指数内的起止日期) |

  - `Features`: 存放的是数值型数据。
    - 如果数据是 **未复权** 的，那么 **`factor`（复权因子）应为1**。

### 依赖数据的Qlib组件

> 为了让这些组件正常运行，需要准备好相应的数据。

  | 组件 | 所需数据 |
  |---|---|
  | 数据检索 (Data Retrieval) | Features, Calendar, Instruments (特征、日历、股票列表) |
  | 策略回测 (Backtest) | **Features[价格/成交量]**, Calendar, Instruments |
