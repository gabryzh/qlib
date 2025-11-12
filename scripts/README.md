<!--
This is a Chinese translation and annotation of the original README.md file.
这是一个对原始README.md文件的中文翻译和注释。
-->

- [下载Qlib数据](#下载Qlib数据)
  - [下载A股数据](#下载A股数据)
  - [下载美股数据](#下载美股数据)
  - [下载A股简化版数据](#下载A股简化版数据)
  - [获取帮助](#获取帮助)
- [在Qlib中使用数据](#在Qlib中使用数据)
  - [美股数据](#美股数据)
  - [A股数据](#A股数据)
- [使用社区众包数据](#使用社区众包数据)


## 下载Qlib数据

本节介绍如何使用`get_data.py`脚本下载`Qlib`所需的示例数据。


### 下载A股数据

```bash
# 下载日线数据
# --target_dir 指定数据存放目录，默认为 ~/.qlib/qlib_data/cn_data
# --region 指定市场区域为中国 (cn)
python get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

# 下载1分钟线数据 (可选，仅在运行非高频策略时需要)
# --interval 指定数据频率为 1min
python get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --region cn --interval 1min
```

### 下载美股数据

```bash
# 下载美股日线数据
# --region 指定市场区域为美国 (us)
python get_data.py qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us
```

### 下载A股简化版数据

这个版本的数据量更小，方便快速体验。
```bash
# --name 指定了使用名为 qlib_data_simple 的数据源配置
python get_data.py qlib_data --name qlib_data_simple --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

### 获取帮助

查看`get_data.py`脚本的详细用法和参数说明。
```bash
python get_data.py qlib_data --help
```

## 在Qlib中使用数据
> 获取更多关于Qlib初始化的信息，请访问官方文档：https://qlib.readthedocs.io/en/latest/start/initialization.html


### 美股数据

> 使用前需要先下载数据：[下载美股数据](#下载美股数据)

```python
import qlib
from qlib.config import REG_US  # 导入美股市场区域常量

# provider_uri 指向你下载数据时设置的 target_dir
provider_uri = "~/.qlib/qlib_data/us_data"
# 初始化Qlib，指定数据路径和市场区域
qlib.init(provider_uri=provider_uri, region=REG_US)
```

### A股数据

> 使用前需要先下载数据：[下载A股数据](#下载A股数据)

```python
import qlib
from qlib.constant import REG_CN  # 导入A股市场区域常量

# provider_uri 指向你下载数据时设置的 target_dir
provider_uri = "~/.qlib/qlib_data/cn_data"
# 初始化Qlib，指定数据路径和市场区域
qlib.init(provider_uri=provider_uri, region=REG_CN)
```

## 使用社区众包数据
除了官方提供的数据外，还有一个由社区贡献和维护的 [Qlib数据版本](data_collector/crowd_source/README.md)：https://github.com/chenditc/investment_data/releases

你可以使用以下命令下载并解压到指定目录：
```bash
# 从GitHub下载最新的数据压缩包
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz

#解压数据到默认的A股数据目录
# --strip-components=2 用于去除压缩包内前两层的目录结构
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=2
```
