# 收集基金数据

> *请**注意**，数据是从[天天基金网](https://fund.eastmoney.com/)收集的，数据可能并不完美。如果用户有高质量的数据集，我们建议他们准备自己的数据。更多信息，用户可以参考[相关文档](https://qlib.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-qlib-format)*

## 依赖要求

```bash
pip install -r requirements.txt
```

## 收集器数据


### A股数据

#### 来自天天基金网的1d数据

```bash

# 从天天基金网下载
python collector.py download_data --source_dir ~/.qlib/fund_data/source/cn_data --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1d

# 规范化
python collector.py normalize_data --source_dir ~/.qlib/fund_data/source/cn_data --normalize_dir ~/.qlib/fund_data/source/cn_1d_nor --region CN --interval 1d --date_field_name FSRQ

# 转储数据
cd qlib/scripts
python dump_bin.py dump_all --data_path ~/.qlib/fund_data/source/cn_1d_nor --qlib_dir ~/.qlib/qlib_data/cn_fund_data --freq day --date_field_name FSRQ --include_fields DWJZ,LJJZ

```

### 使用数据

```python
import qlib
from qlib.data import D

qlib.init(provider_uri="~/.qlib/qlib_data/cn_fund_data")
df = D.features(D.instruments(market="all"), ["$DWJZ", "$LJJZ"], freq="day")
```


### 帮助
```bash
pythono collector.py collector_data --help
```

## 参数

- interval: 1d
- region: CN

## 免责声明

本项目仅供学习研究使用，不作为任何行为的指导和建议，由此而引发任何争议和纠纷，与本项目无任何关系
