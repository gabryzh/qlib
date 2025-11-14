# 介绍

这个例子试图演示Qlib如何支持没有固定共享频率的数据。

例如，
- 每日价格成交量数据是固定频率的数据。数据以固定的频率（即每日）出现
- 订单不是固定的数据，它们可能在任何时间点出现

为了支持这种非固定频率，Qlib实现了一个基于Arctic的后端。
这是一个基于这个后端导入和查询数据的例子。

# 安装

请参考mongodb的[安装文档](https://docs.mongodb.com/manual/installation/)。
当前版本的脚本默认值试图连接本地主机的**默认端口，无需身份验证**。

运行以下命令安装必要的库
```
pip install pytest coverage gdown
pip install arctic  # 注意：pip可能无法解析正确的包依赖关系！！！请确保满足依赖关系。
```

# 导入示例数据


1. （可选）请按照[本节](https://github.com/microsoft/qlib#data-preparation)的第一部分**获取Qlib的1分钟数据**。
2. 请按照以下步骤下载示例数据
```bash
cd examples/orderbook_data/
gdown https://drive.google.com/uc?id=15FuUqWn2rkCi8uhJYGEQWKakcEqLJNDG  # 这里可能需要代理。
python ../../scripts/get_data.py _unzip --file_path highfreq_orderbook_example_data.zip --target_dir .
```

3. 请将示例数据导入您的mongo db
```bash
python create_dataset.py initialize_library  # 初始化库
python create_dataset.py import_data  # 初始化库
```

# 查询示例

导入这些数据后，您可以运行`example.py`来创建一些高频特征。
```bash
pytest -s --disable-warnings example.py   # 如果您想运行所有示例
pytest -s --disable-warnings example.py::TestClass::test_exp_10  # 如果您想运行特定示例
```


# 已知限制
目前尚不支持不同频率之间的表达式计算
