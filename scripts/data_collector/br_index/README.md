# IBOVESPA 指数历史成分股收集说明

## 环境要求

-   安装 `requirements.txt` 文件中列出的依赖库：

    ```bash
    pip install -r requirements.txt
    ```
-   注意：`requirements.txt` 文件是基于 Python 3.8 环境生成的。

---

## 关于 IBOVESPA (IBOV) 指数的数据收集

### `get_new_companies` 方法详解

#### **指数起始日期**

-   根据维基百科，IBOVESPA 指数始于1968年1月2日。然而，要在我们的 `bench_start_date` 方法中使用这个日期，必须满足两个条件：
    1.  用于下载巴西股票历史价格的API（例如 Yahoo Finance）必须提供从1968年1月2日至今的数据。
    2.  必须有某个网站或API提供从那天起的完整历史成分股列表。

-   由于上述两个条件难以满足，我们在 `collector.py` 中将 `bench_start_date` 设置为 `pd.Timestamp("2003-01-03")`，原因如下：
    1.  目前能找到的最早的IBOVESPA指数成分股数据源是从2003年第一季度开始的。
    2.  Yahoo Finance 提供的数据也大致从这个日期开始。

-   在 `get_new_companies` 方法中，我们实现了一个逻辑来获取每只成分股在Yahoo Finance上有记录的最早日期。

#### **代码逻辑**

-   最初的设想是对B3交易所的[官方网站](https://sistemaswebb3-listados.b3.com.br/indexPage/day/IBOV?language=pt-br)进行网络爬虫，该网站显示了当天的指数成分股。

-   然而，直接使用 `requests` + `Beautiful Soup` 的方法行不通，因为该网站上的成分股表格是通过内部脚本动态加载的，存在延迟。

-   为了解决这个问题，我们曾考虑使用 `selenium` 来模拟浏览器行为以获取数据。但最终，我们采用了一个更稳定的数据源（见下一节）。

---

### `get_changes` 方法详解

-   我们没能找到一个官方的、持续更新的IBOVESPA历史成分股数据源。

-   最终，我们使用了一个第三方的GitHub仓库：[https://github.com/igor17400/IBOV-HCI](https://github.com/igor17400/IBOV-HCI)。这个仓库提供了从2003年第一季度到2021年第三季度的指数历史成分数据。

-   基于这个数据源，我们可以逐个周期地（每四个月）比较成分股列表，从而推断出每个周期内哪些股票被调入指数，哪些被调出。

---

### 数据收集命令

```bash
# 解析完整的成分股历史，生成 qlib/instruments/ibov.txt 文件
# 这个文件记录了每只成分股在指数中的完整生命周期
python collector.py get_instruments --index_name IBOV --qlib_dir ~/.qlib/qlib_data/br_data --method parse_instruments

# 仅获取最新的成分股列表，并保存到 qlib/instruments/ibov_only_new.txt
python collector.py get_instruments --index_name IBOV --qlib_dir ~/.qlib/qlib_data/br_data --method save_new_companies
```
