# KRNN
* 代码: [https://github.com/microsoft/FOST/blob/main/fostool/model/krnn.py](https://github.com/microsoft/FOST/blob/main/fostool/model/krnn.py)


# 关于设置/配置的介绍。
* FOST的原始模型中使用了Torch_geometric，但我们没有使用它。
* 确保您的CUDA版本与torch版本匹配以允许使用GPU，我们使用CUDA==10.2和torch.__version__==1.12.1
