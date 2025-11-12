# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import data_formatters.base
import expt_settings.configs
import libs.hyperparam_opt
import libs.tft_model
import libs.utils as utils
import os
import datetime as dte


from qlib.model.base import ModelFT
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


# 要注册新的数据集，请在此处添加它们。
ALLOW_DATASET = ["Alpha158", "Alpha360"]
# 要注册新的数据集，请在此处添加它们的配置。
DATASET_SETTING = {
    "Alpha158": {
        "feature_col": [
            "RESI5",
            "WVMA5",
            "RSQR5",
            "KLEN",
            "RSQR10",
            "CORR5",
            "CORD5",
            "CORR10",
            "ROC60",
            "RESI10",
            "VSTD5",
            "RSQR60",
            "CORR60",
            "WVMA60",
            "STD5",
            "RSQR20",
            "CORD60",
            "CORD10",
            "CORR20",
            "KLOW",
        ],
        "label_col": "LABEL0",
    },
    "Alpha360": {
        "feature_col": [
            "HIGH0",
            "LOW0",
            "OPEN0",
            "CLOSE1",
            "HIGH1",
            "VOLUME1",
            "LOW1",
            "VOLUME3",
            "OPEN1",
            "VOLUME4",
            "CLOSE2",
            "CLOSE4",
            "VOLUME5",
            "LOW2",
            "CLOSE3",
            "VOLUME2",
            "HIGH2",
            "LOW4",
            "VOLUME8",
            "VOLUME11",
        ],
        "label_col": "LABEL0",
    },
}


def get_shifted_label(data_df, shifts=5, col_shift="LABEL0"):
    """
    获取移位后的标签。

    :param data_df: 输入的数据帧。
    :param shifts: 移位的位数。
    :param col_shift: 要移位的列名。
    :return: 移位后的标签列。
    """
    return data_df[[col_shift]].groupby("instrument", group_keys=False).apply(lambda df: df.shift(shifts))


def fill_test_na(test_df):
    """
    填充测试数据中的缺失值。

    :param test_df: 测试数据帧。
    :return: 填充缺失值后的测试数据帧。
    """
    test_df_res = test_df.copy()
    feature_cols = ~test_df_res.columns.str.contains("label", case=False)
    test_feature_fna = (
        test_df_res.loc[:, feature_cols].groupby("datetime", group_keys=False).apply(lambda df: df.fillna(df.mean()))
    )
    test_df_res.loc[:, feature_cols] = test_feature_fna
    return test_df_res


def process_qlib_data(df, dataset, fillna=False):
    """准备数据以适应 TFT 模型。

    参数:
      df: 原始 DataFrame。
      fillna: 是否用平均值填充数据。

    返回:
      转换后的 DataFrame。

    """
    # 手动选择的几个特征
    feature_col = DATASET_SETTING[dataset]["feature_col"]
    label_col = [DATASET_SETTING[dataset]["label_col"]]
    temp_df = df.loc[:, feature_col + label_col]
    if fillna:
        temp_df = fill_test_na(temp_df)
    temp_df = temp_df.swaplevel()
    temp_df = temp_df.sort_index()
    temp_df = temp_df.reset_index(level=0)
    dates = pd.to_datetime(temp_df.index)
    temp_df["date"] = dates
    temp_df["day_of_week"] = dates.dayofweek
    temp_df["month"] = dates.month
    temp_df["year"] = dates.year
    temp_df["const"] = 1.0
    return temp_df


def process_predicted(df, col_name):
    """将 TFT 预测的数据转换为 Qlib 格式。

    参数:
      df: 原始 DataFrame。
      col_name: 新列名。

    返回:
      转换后的 DataFrame。

    """
    df_res = df.copy()
    df_res = df_res.rename(columns={"forecast_time": "datetime", "identifier": "instrument", "t+4": col_name})
    df_res = df_res.set_index(["datetime", "instrument"]).sort_index()
    df_res = df_res[[col_name]]
    return df_res


def format_score(forecast_df, col_name="pred", label_shift=5):
    """
    格式化分数。

    :param forecast_df: 预测数据帧。
    :param col_name: 预测列的名称。
    :param label_shift: 标签移位的位数。
    :return: 格式化后的分数。
    """
    pred = process_predicted(forecast_df, col_name=col_name)
    pred = get_shifted_label(pred, shifts=-label_shift, col_shift=col_name)
    pred = pred.dropna()[col_name]
    return pred


def transform_df(df, col_name="LABEL0"):
    """
    转换数据帧。

    :param df: 输入的数据帧。
    :param col_name: 标签列的名称。
    :return: 转换后的数据帧。
    """
    df_res = df["feature"]
    df_res[col_name] = df["label"]
    return df_res


class TFTModel(ModelFT):
    """TFT 模型"""

    def __init__(self, **kwargs):
        self.model = None
        self.params = {"DATASET": "Alpha158", "label_shift": 5}
        self.params.update(kwargs)

    def _prepare_data(self, dataset: DatasetH):
        """
        准备数据。

        :param dataset: DatasetH 对象。
        :return: 训练和验证数据帧。
        """
        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        return transform_df(df_train), transform_df(df_valid)

    def fit(self, dataset: DatasetH, MODEL_FOLDER="qlib_tft_model", USE_GPU_ID=0, **kwargs):
        """
        拟合模型。

        :param dataset: DatasetH 对象。
        :param MODEL_FOLDER: 模型文件夹。
        :param USE_GPU_ID: 使用的 GPU ID。
        """
        DATASET = self.params["DATASET"]
        LABEL_SHIFT = self.params["label_shift"]
        LABEL_COL = DATASET_SETTING[DATASET]["label_col"]

        if DATASET not in ALLOW_DATASET:
            raise AssertionError("不支持该数据集，请创建一个新的格式化程序来适应此数据集")

        dtrain, dvalid = self._prepare_data(dataset)
        dtrain.loc[:, LABEL_COL] = get_shifted_label(dtrain, shifts=LABEL_SHIFT, col_shift=LABEL_COL)
        dvalid.loc[:, LABEL_COL] = get_shifted_label(dvalid, shifts=LABEL_SHIFT, col_shift=LABEL_COL)

        train = process_qlib_data(dtrain, DATASET, fillna=True).dropna()
        valid = process_qlib_data(dvalid, DATASET, fillna=True).dropna()

        ExperimentConfig = expt_settings.configs.ExperimentConfig
        config = ExperimentConfig(DATASET)
        self.data_formatter = config.make_data_formatter()
        self.model_folder = MODEL_FOLDER
        self.gpu_id = USE_GPU_ID
        self.label_shift = LABEL_SHIFT
        self.expt_name = DATASET
        self.label_col = LABEL_COL

        use_gpu = (True, self.gpu_id)
        # ===========================训练过程===========================
        ModelClass = libs.tft_model.TemporalFusionTransformer
        if not isinstance(self.data_formatter, data_formatters.base.GenericDataFormatter):
            raise ValueError(
                "数据格式化程序应继承自"
                + "AbstractDataFormatter! 类型={}".format(type(self.data_formatter))
            )

        default_keras_session = tf.keras.backend.get_session()

        if use_gpu[0]:
            self.tf_config = utils.get_default_tensorflow_config(tf_device="gpu", gpu_id=use_gpu[1])
        else:
            self.tf_config = utils.get_default_tensorflow_config(tf_device="cpu")

        self.data_formatter.set_scalers(train)

        # 设置默认参数
        fixed_params = self.data_formatter.get_experiment_params()
        params = self.data_formatter.get_default_model_params()

        params = {**params, **fixed_params}

        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
        params["model_folder"] = self.model_folder

        print("*** 开始训练 ***")
        best_loss = np.Inf

        tf.reset_default_graph()

        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.sess = tf.Session(config=self.tf_config)
            tf.keras.backend.set_session(self.sess)
            self.model = ModelClass(params, use_cudnn=use_gpu[0])
            self.sess.run(tf.global_variables_initializer())
            self.model.fit(train_df=train, valid_df=valid)
            print("*** 训练完成 ***")
            saved_model_dir = self.model_folder + "/" + "saved_model"
            if not os.path.exists(saved_model_dir):
                os.makedirs(saved_model_dir)
            self.model.save(saved_model_dir)

            def extract_numerical_data(data):
                """剥离 forecast_time 和 identifier 列。"""
                return data[[col for col in data.columns if col not in {"forecast_time", "identifier"}]]

            tf.keras.backend.set_session(default_keras_session)
        print("训练于 {} 完成。".format(dte.datetime.now()))
        # ===========================训练过程===========================

    def predict(self, dataset):
        """
        预测。

        :param dataset: DatasetH 对象。
        :return: 预测结果。
        """
        if self.model is None:
            raise ValueError("模型尚未拟合！")
        d_test = dataset.prepare("test", col_set=["feature", "label"])
        d_test = transform_df(d_test)
        d_test.loc[:, self.label_col] = get_shifted_label(d_test, shifts=self.label_shift, col_shift=self.label_col)
        test = process_qlib_data(d_test, self.expt_name, fillna=True).dropna()

        use_gpu = (True, self.gpu_id)
        # ===========================预测过程===========================
        default_keras_session = tf.keras.backend.get_session()

        # 设置默认参数
        fixed_params = self.data_formatter.get_experiment_params()
        params = self.data_formatter.get_default_model_params()
        params = {**params, **fixed_params}

        print("*** 开始预测 ***")
        tf.reset_default_graph()

        with self.tf_graph.as_default():
            tf.keras.backend.set_session(self.sess)
            output_map = self.model.predict(test, return_targets=True)
            targets = self.data_formatter.format_predictions(output_map["targets"])
            p50_forecast = self.data_formatter.format_predictions(output_map["p50"])
            p90_forecast = self.data_formatter.format_predictions(output_map["p90"])
            tf.keras.backend.set_session(default_keras_session)

        predict50 = format_score(p50_forecast, "pred", 1)
        predict90 = format_score(p90_forecast, "pred", 1)
        predict = (predict50 + predict90) / 2  # self.label_shift
        # ===========================预测过程===========================
        return predict

    def finetune(self, dataset: DatasetH):
        """
        微调模型
        参数
        ----------
        dataset : DatasetH
            用于微调的数据集
        """
        pass

    def to_pickle(self, path: Union[Path, str]):
        """
        Tensorflow 模型不能直接转储。
        所以数据应该分开保存

        **TODO**: 请实现加载文件的函数

        参数
        ----------
        path : Union[Path, str]
            要转储的目标路径
        """
        # FIXME: 实现保存 tensorflow 模型
        # 保存 tensorflow 模型
        # path = Path(path)
        # path.mkdir(parents=True)
        # self.model.save(path)

        # 保存 qlib 模型包装器
        drop_attrs = ["model", "tf_graph", "sess", "data_formatter"]
        orig_attr = {}
        for attr in drop_attrs:
            orig_attr[attr] = getattr(self, attr)
            setattr(self, attr, None)
        super(TFTModel, self).to_pickle(path)
        for attr in drop_attrs:
            setattr(self, attr, orig_attr[attr])
