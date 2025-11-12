# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

from typing import Union, List, Type
from scipy.stats import percentileofscore
from .base import Expression, ExpressionOps, Feature, PFeature
from ..log import get_module_logger
from ..utils import get_callable_kwargs

try:
    from ._libs.rolling import rolling_slope, rolling_rsquare, rolling_resi
    from ._libs.expanding import expanding_slope, expanding_rsquare, expanding_resi
except ImportError:
    print(
        "#### 不要在仓库目录中导入 qlib 包，以免在未编译的情况下从 . 导入 qlib #####"
    )
    raise
except ValueError:
    print("!!!!!!!! 导入基于 Cython 实现的运算符时发生错误。!!!!!!!!")
    print("!!!!!!!! 它们将被禁用。请升级您的 numpy 以启用它们     !!!!!!!!")
    # 我们捕获此错误是因为某些平台无法升级其包（例如 Kaggle）
    # https://www.kaggle.com/general/293387
    # https://www.kaggle.com/product-feedback/98562


np.seterr(invalid="ignore")


#################### 逐元素运算符 ####################
class ElemOperator(ExpressionOps):
    """逐元素运算符

    参数
    ----------
    feature : Expression
        特征实例

    返回
    ----------
    Expression
        特征运算输出
    """

    def __init__(self, feature):
        self.feature = feature

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.feature)

    def get_longest_back_rolling(self):
        return self.feature.get_longest_back_rolling()

    def get_extended_window_size(self):
        return self.feature.get_extended_window_size()


class ChangeInstrument(ElemOperator):
    """更改金融工具运算符
    在某些情况下，可能需要在计算时更改为另一个金融工具，例如，
    计算某只股票相对于市场指数的 beta 值。
    这将需要将特征的计算从股票（原始金融工具）更改为
    指数（参考金融工具）
    参数
    ----------
    instrument: 新的金融工具，下游操作将在此金融工具上执行。
                例如，SH000300（沪深300指数），或 ^GPSC（标普500指数）。

    feature: 要为新金融工具计算的特征。
    返回
    ----------
    Expression
        特征运算输出
    """

    def __init__(self, instrument, feature):
        self.instrument = instrument
        self.feature = feature

    def __str__(self):
        return "{}('{}',{})".format(type(self).__name__, self.instrument, self.feature)

    def load(self, instrument, start_index, end_index, *args):
        # 忽略第一个 `instrument`
        return super().load(self.instrument, start_index, end_index, *args)

    def _load_internal(self, instrument, start_index, end_index, *args):
        return self.feature.load(instrument, start_index, end_index, *args)


class NpElemOperator(ElemOperator):
    """Numpy 逐元素运算符

    参数
    ----------
    feature : Expression
        特征实例
    func : str
        numpy 特征运算方法

    返回
    ----------
    Expression
        特征运算输出
    """

    def __init__(self, feature, func):
        self.func = func
        super(NpElemOperator, self).__init__(feature)

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        return getattr(np, self.func)(series)


class Abs(NpElemOperator):
    """特征绝对值

    参数
    ----------
    feature : Expression
        特征实例

    返回
    ----------
    Expression
        一个输出绝对值的特征实例
    """

    def __init__(self, feature):
        super(Abs, self).__init__(feature, "abs")


class Sign(NpElemOperator):
    """特征符号

    参数
    ----------
    feature : Expression
        特征实例

    返回
    ----------
    Expression
        一个带符号的特征实例
    """

    def __init__(self, feature):
        super(Sign, self).__init__(feature, "sign")

    def _load_internal(self, instrument, start_index, end_index, *args):
        """
        为避免布尔类型输入引发错误，我们将数据转换为 float32。
        """
        series = self.feature.load(instrument, start_index, end_index, *args)
        # TODO:  更多精度类型应可配置
        series = series.astype(np.float32)
        return getattr(np, self.func)(series)


class Log(NpElemOperator):
    """特征对数

    参数
    ----------
    feature : Expression
        特征实例

    返回
    ----------
    Expression
        一个带对数的特征实例
    """

    def __init__(self, feature):
        super(Log, self).__init__(feature, "log")


class Mask(NpElemOperator):
    """特征掩码

    参数
    ----------
    feature : Expression
        特征实例
    instrument : str
        金融工具掩码

    返回
    ----------
    Expression
        一个带掩码金融工具的特征实例
    """

    def __init__(self, feature, instrument):
        super(Mask, self).__init__(feature, "mask")
        self.instrument = instrument

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature, self.instrument.lower())

    def _load_internal(self, instrument, start_index, end_index, *args):
        return self.feature.load(self.instrument, start_index, end_index, *args)


class Not(NpElemOperator):
    """非运算符

    参数
    ----------
    feature : Expression
        特征实例

    返回
    ----------
    Feature:
        特征逐元素非运算输出
    """

    def __init__(self, feature):
        super(Not, self).__init__(feature, "bitwise_not")


#################### 双目运算符 ####################
class PairOperator(ExpressionOps):
    """双目运算符

    参数
    ----------
    feature_left : Expression
        特征实例或数值
    feature_right : Expression
        特征实例或数值

    返回
    ----------
    Feature:
        两个特征的运算输出
    """

    def __init__(self, feature_left, feature_right):
        self.feature_left = feature_left
        self.feature_right = feature_right

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature_left, self.feature_right)

    def get_longest_back_rolling(self):
        if isinstance(self.feature_left, (Expression,)):
            left_br = self.feature_left.get_longest_back_rolling()
        else:
            left_br = 0

        if isinstance(self.feature_right, (Expression,)):
            right_br = self.feature_right.get_longest_back_rolling()
        else:
            right_br = 0
        return max(left_br, right_br)

    def get_extended_window_size(self):
        if isinstance(self.feature_left, (Expression,)):
            ll, lr = self.feature_left.get_extended_window_size()
        else:
            ll, lr = 0, 0

        if isinstance(self.feature_right, (Expression,)):
            rl, rr = self.feature_right.get_extended_window_size()
        else:
            rl, rr = 0, 0
        return max(ll, rl), max(lr, rr)


class NpPairOperator(PairOperator):
    """Numpy 双目运算符

    参数
    ----------
    feature_left : Expression
        特征实例或数值
    feature_right : Expression
        特征实例或数值
    func : str
        运算符函数

    返回
    ----------
    Feature:
        两个特征的运算输出
    """

    def __init__(self, feature_left, feature_right, func):
        self.func = func
        super(NpPairOperator, self).__init__(feature_left, feature_right)

    def _load_internal(self, instrument, start_index, end_index, *args):
        assert any(
            [isinstance(self.feature_left, (Expression,)), self.feature_right, Expression]
        ), "两个输入中至少有一个是 Expression 实例"
        if isinstance(self.feature_left, (Expression,)):
            series_left = self.feature_left.load(instrument, start_index, end_index, *args)
        else:
            series_left = self.feature_left  # 数值
        if isinstance(self.feature_right, (Expression,)):
            series_right = self.feature_right.load(instrument, start_index, end_index, *args)
        else:
            series_right = self.feature_right
        check_length = isinstance(series_left, (np.ndarray, pd.Series)) and isinstance(
            series_right, (np.ndarray, pd.Series)
        )
        if check_length:
            warning_info = (
                f"加载 {instrument}: {str(self)}; np.{self.func}(series_left, series_right), "
                f"series_left 和 series_right 的长度不同: ({len(series_left)}, {len(series_right)}), "
                f"series_left 是 {str(self.feature_left)}, series_right 是 {str(self.feature_right)}. 请检查数据"
            )
        else:
            warning_info = (
                f"加载 {instrument}: {str(self)}; np.{self.func}(series_left, series_right), "
                f"series_left 是 {str(self.feature_left)}, series_right 是 {str(self.feature_right)}. 请检查数据"
            )
        try:
            res = getattr(np, self.func)(series_left, series_right)
        except ValueError as e:
            get_module_logger("ops").debug(warning_info)
            raise ValueError(f"{str(e)}. \n\t{warning_info}") from e
        else:
            if check_length and len(series_left) != len(series_right):
                get_module_logger("ops").debug(warning_info)
        return res


class Power(NpPairOperator):
    """幂运算符

    参数
    ----------
    feature_left : Expression
        特征实例
    feature_right : Expression
        特征实例

    返回
    ----------
    Feature:
        feature_left 中的基数乘以 feature_right 中的指数
    """

    def __init__(self, feature_left, feature_right):
        super(Power, self).__init__(feature_left, feature_right, "power")


class Add(NpPairOperator):
    """加法运算符

    参数
    ----------
    feature_left : Expression
        特征实例
    feature_right : Expression
        特征实例

    返回
    ----------
    Feature:
        两个特征的和
    """

    def __init__(self, feature_left, feature_right):
        super(Add, self).__init__(feature_left, feature_right, "add")


class Sub(NpPairOperator):
    """减法运算符

    参数
    ----------
    feature_left : Expression
        特征实例
    feature_right : Expression
        特征实例

    返回
    ----------
    Feature:
        两个特征的差
    """

    def __init__(self, feature_left, feature_right):
        super(Sub, self).__init__(feature_left, feature_right, "subtract")


class Mul(NpPairOperator):
    """乘法运算符

    参数
    ----------
    feature_left : Expression
        特征实例
    feature_right : Expression
        特征实例

    返回
    ----------
    Feature:
        两个特征的积
    """

    def __init__(self, feature_left, feature_right):
        super(Mul, self).__init__(feature_left, feature_right, "multiply")


class Div(NpPairOperator):
    """除法运算符

    参数
    ----------
    feature_left : Expression
        特征实例
    feature_right : Expression
        特征实例

    返回
    ----------
    Feature:
        两个特征的商
    """

    def __init__(self, feature_left, feature_right):
        super(Div, self).__init__(feature_left, feature_right, "divide")


class Greater(NpPairOperator):
    """较大值运算符

    参数
    ----------
    feature_left : Expression
        特征实例
    feature_right : Expression
        特征实例

    返回
    ----------
    Feature:
        从输入的两个特征中取出的较大元素
    """

    def __init__(self, feature_left, feature_right):
        super(Greater, self).__init__(feature_left, feature_right, "maximum")


class Less(NpPairOperator):
    """较小值运算符

    参数
    ----------
    feature_left : Expression
        特征实例
    feature_right : Expression
        特征实例

    返回
    ----------
    Feature:
        从输入的两个特征中取出的较小元素
    """

    def __init__(self, feature_left, feature_right):
        super(Less, self).__init__(feature_left, feature_right, "minimum")


class Gt(NpPairOperator):
    """大于运算符

    参数
    ----------
    feature_left : Expression
        特征实例
    feature_right : Expression
        特征实例

    返回
    ----------
    Feature:
        指示 `left > right` 的布尔序列
    """

    def __init__(self, feature_left, feature_right):
        super(Gt, self).__init__(feature_left, feature_right, "greater")


class Ge(NpPairOperator):
    """大于等于运算符

    参数
    ----------
    feature_left : Expression
        特征实例
    feature_right : Expression
        特征实例

    返回
    ----------
    Feature:
        指示 `left >= right` 的布尔序列
    """

    def __init__(self, feature_left, feature_right):
        super(Ge, self).__init__(feature_left, feature_right, "greater_equal")


class Lt(NpPairOperator):
    """小于运算符

    参数
    ----------
    feature_left : Expression
        特征实例
    feature_right : Expression
        特征实例

    返回
    ----------
    Feature:
        指示 `left < right` 的布尔序列
    """

    def __init__(self, feature_left, feature_right):
        super(Lt, self).__init__(feature_left, feature_right, "less")


class Le(NpPairOperator):
    """小于等于运算符

    参数
    ----------
    feature_left : Expression
        特征实例
    feature_right : Expression
        特征实例

    返回
    ----------
    Feature:
        指示 `left <= right` 的布尔序列
    """

    def __init__(self, feature_left, feature_right):
        super(Le, self).__init__(feature_left, feature_right, "less_equal")


class Eq(NpPairOperator):
    """等于运算符

    参数
    ----------
    feature_left : Expression
        特征实例
    feature_right : Expression
        特征实例

    返回
    ----------
    Feature:
        指示 `left == right` 的布尔序列
    """

    def __init__(self, feature_left, feature_right):
        super(Eq, self).__init__(feature_left, feature_right, "equal")


class Ne(NpPairOperator):
    """不等于运算符

    参数
    ----------
    feature_left : Expression
        特征实例
    feature_right : Expression
        特征实例

    返回
    ----------
    Feature:
        指示 `left != right` 的布尔序列
    """

    def __init__(self, feature_left, feature_right):
        super(Ne, self).__init__(feature_left, feature_right, "not_equal")


class And(NpPairOperator):
    """与运算符

    参数
    ----------
    feature_left : Expression
        特征实例
    feature_right : Expression
        特征实例

    返回
    ----------
    Feature:
        两个特征的逐行 & 输出
    """

    def __init__(self, feature_left, feature_right):
        super(And, self).__init__(feature_left, feature_right, "bitwise_and")


class Or(NpPairOperator):
    """或运算符

    参数
    ----------
    feature_left : Expression
        特征实例
    feature_right : Expression
        特征实例

    返回
    ----------
    Feature:
        两个特征的逐行 | 输出
    """

    def __init__(self, feature_left, feature_right):
        super(Or, self).__init__(feature_left, feature_right, "bitwise_or")


#################### 三目运算符 ####################
class If(ExpressionOps):
    """If 运算符

    参数
    ----------
    condition : Expression
        带布尔值的特征实例作为条件
    feature_left : Expression
        特征实例
    feature_right : Expression
        特征实例
    """

    def __init__(self, condition, feature_left, feature_right):
        self.condition = condition
        self.feature_left = feature_left
        self.feature_right = feature_right

    def __str__(self):
        return "If({},{},{})".format(self.condition, self.feature_left, self.feature_right)

    def _load_internal(self, instrument, start_index, end_index, *args):
        series_cond = self.condition.load(instrument, start_index, end_index, *args)
        if isinstance(self.feature_left, (Expression,)):
            series_left = self.feature_left.load(instrument, start_index, end_index, *args)
        else:
            series_left = self.feature_left
        if isinstance(self.feature_right, (Expression,)):
            series_right = self.feature_right.load(instrument, start_index, end_index, *args)
        else:
            series_right = self.feature_right
        series = pd.Series(np.where(series_cond, series_left, series_right), index=series_cond.index)
        return series

    def get_longest_back_rolling(self):
        if isinstance(self.feature_left, (Expression,)):
            left_br = self.feature_left.get_longest_back_rolling()
        else:
            left_br = 0

        if isinstance(self.feature_right, (Expression,)):
            right_br = self.feature_right.get_longest_back_rolling()
        else:
            right_br = 0

        if isinstance(self.condition, (Expression,)):
            c_br = self.condition.get_longest_back_rolling()
        else:
            c_br = 0
        return max(left_br, right_br, c_br)

    def get_extended_window_size(self):
        if isinstance(self.feature_left, (Expression,)):
            ll, lr = self.feature_left.get_extended_window_size()
        else:
            ll, lr = 0, 0

        if isinstance(self.feature_right, (Expression,)):
            rl, rr = self.feature_right.get_extended_window_size()
        else:
            rl, rr = 0, 0

        if isinstance(self.condition, (Expression,)):
            cl, cr = self.condition.get_extended_window_size()
        else:
            cl, cr = 0, 0
        return max(ll, rl, cl), max(lr, rr, cr)


#################### 滚动 ####################
# 注意：像 `rolling.mean` 这样的方法使用 cython 进行了优化，
# 并且比 `rolling.apply(np.mean)` 快得多


class Rolling(ExpressionOps):
    """滚动运算符
    在 pandas 中，rolling 和 expanding 的含义相同。
    当窗口设置为 0 时，运算符的行为应遵循 `expanding`
    否则，它遵循 `rolling`

    参数
    ----------
    feature : Expression
        特征实例
    N : int
        滚动窗口大小
    func : str
        滚动方法

    返回
    ----------
    Expression
        滚动输出
    """

    def __init__(self, feature, N, func):
        self.feature = feature
        self.N = N
        self.func = func

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if isinstance(self.N, int) and self.N == 0:
            series = getattr(series.expanding(min_periods=1), self.func)()
        elif isinstance(self.N, float) and 0 < self.N < 1:
            series = series.ewm(alpha=self.N, min_periods=1).mean()
        else:
            series = getattr(series.rolling(self.N, min_periods=1), self.func)()
        return series

    def get_longest_back_rolling(self):
        if self.N == 0:
            return np.inf
        if 0 < self.N < 1:
            return int(np.log(1e-6) / np.log(1 - self.N))  # (1 - N)**window == 1e-6
        return self.feature.get_longest_back_rolling() + self.N - 1

    def get_extended_window_size(self):
        if self.N == 0:
            get_module_logger(self.__class__.__name__).warning("Rolling(ATTR, 0) 的计算可能不准确")
            return self.feature.get_extended_window_size()
        elif 0 < self.N < 1:
            lft_etd, rght_etd = self.feature.get_extended_window_size()
            size = int(np.log(1e-6) / np.log(1 - self.N))
            lft_etd = max(lft_etd + size - 1, lft_etd)
            return lft_etd, rght_etd
        else:
            lft_etd, rght_etd = self.feature.get_extended_window_size()
            lft_etd = max(lft_etd + self.N - 1, lft_etd)
            return lft_etd, rght_etd


class Ref(Rolling):
    """特征引用

    参数
    ----------
    feature : Expression
        特征实例
    N : int
        N = 0, 检索第一个数据；N > 0, 检索 N 个周期前的数据；N < 0, 未来数据

    返回
    ----------
    Expression
        一个带目标引用的特征实例
    """

    def __init__(self, feature, N):
        super(Ref, self).__init__(feature, N, "ref")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if series.empty:
            return series
        elif self.N == 0:
            series = pd.Series(series.iloc[0], index=series.index)
        else:
            series = series.shift(self.N)
        return series

    def get_longest_back_rolling(self):
        if self.N == 0:
            return np.inf
        return self.feature.get_longest_back_rolling() + self.N

    def get_extended_window_size(self):
        if self.N == 0:
            get_module_logger(self.__class__.__name__).warning("Ref(ATTR, 0) 的计算可能不准确")
            return self.feature.get_extended_window_size()
        else:
            lft_etd, rght_etd = self.feature.get_extended_window_size()
            lft_etd = max(lft_etd + self.N, lft_etd)
            rght_etd = max(rght_etd - self.N, rght_etd)
            return lft_etd, rght_etd


class Mean(Rolling):
    """滚动平均值 (MA)

    参数
    ----------
    feature : Expression
        特征实例
    N : int
        滚动窗口大小

    返回
    ----------
    Expression
        一个带滚动平均值的特征实例
    """

    def __init__(self, feature, N):
        super(Mean, self).__init__(feature, N, "mean")


class Sum(Rolling):
    """滚动求和

    参数
    ----------
    feature : Expression
        特征实例
    N : int
        滚动窗口大小

    返回
    ----------
    Expression
        一个带滚动和的特征实例
    """

    def __init__(self, feature, N):
        super(Sum, self).__init__(feature, N, "sum")


class Std(Rolling):
    """滚动标准差

    参数
    ----------
    feature : Expression
        特征实例
    N : int
        滚动窗口大小

    返回
    ----------
    Expression
        一个带滚动标准差的特征实例
    """

    def __init__(self, feature, N):
        super(Std, self).__init__(feature, N, "std")


class Var(Rolling):
    """滚动方差

    参数
    ----------
    feature : Expression
        特征实例
    N : int
        滚动窗口大小

    返回
    ----------
    Expression
        一个带滚动方差的特征实例
    """

    def __init__(self, feature, N):
        super(Var, self).__init__(feature, N, "var")


class Skew(Rolling):
    """滚动偏度

    参数
    ----------
    feature : Expression
        特征实例
    N : int
        滚动窗口大小

    返回
    ----------
    Expression
        一个带滚动偏度的特征实例
    """

    def __init__(self, feature, N):
        if N != 0 and N < 3:
            raise ValueError("偏度运算的滚动窗口大小应 >= 3")
        super(Skew, self).__init__(feature, N, "skew")


class Kurt(Rolling):
    """滚动峰度

    参数
    ----------
    feature : Expression
        特征实例
    N : int
        滚动窗口大小

    返回
    ----------
    Expression
        一个带滚动峰度的特征实例
    """

    def __init__(self, feature, N):
        if N != 0 and N < 4:
            raise ValueError("峰度运算的滚动窗口大小应 >= 4")
        super(Kurt, self).__init__(feature, N, "kurt")


class Max(Rolling):
    """滚动最大值

    参数
    ----------
    feature : Expression
        特征实例
    N : int
        滚动窗口大小

    返回
    ----------
    Expression
        一个带滚动最大值的特征实例
    """

    def __init__(self, feature, N):
        super(Max, self).__init__(feature, N, "max")


class IdxMax(Rolling):
    """滚动最大值索引

    参数
    ----------
    feature : Expression
        特征实例
    N : int
        滚动窗口大小

    返回
    ----------
    Expression
        一个带滚动最大值索引的特征实例
    """

    def __init__(self, feature, N):
        super(IdxMax, self).__init__(feature, N, "idxmax")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = series.expanding(min_periods=1).apply(lambda x: x.argmax() + 1, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(lambda x: x.argmax() + 1, raw=True)
        return series


class Min(Rolling):
    """滚动最小值

    参数
    ----------
    feature : Expression
        特征实例
    N : int
        滚动窗口大小

    返回
    ----------
    Expression
        一个带滚动最小值的特征实例
    """

    def __init__(self, feature, N):
        super(Min, self).__init__(feature, N, "min")


class IdxMin(Rolling):
    """滚动最小值索引

    参数
    ----------
    feature : Expression
        特征实例
    N : int
        滚动窗口大小

    返回
    ----------
    Expression
        一个带滚动最小值索引的特征实例
    """

    def __init__(self, feature, N):
        super(IdxMin, self).__init__(feature, N, "idxmin")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = series.expanding(min_periods=1).apply(lambda x: x.argmin() + 1, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(lambda x: x.argmin() + 1, raw=True)
        return series


class Quantile(Rolling):
    """滚动分位数

    参数
    ----------
    feature : Expression
        特征实例
    N : int
        滚动窗口大小

    返回
    ----------
    Expression
        一个带滚动分位数的特征实例
    """

    def __init__(self, feature, N, qscore):
        super(Quantile, self).__init__(feature, N, "quantile")
        self.qscore = qscore

    def __str__(self):
        return "{}({},{},{})".format(type(self).__name__, self.feature, self.N, self.qscore)

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = series.expanding(min_periods=1).quantile(self.qscore)
        else:
            series = series.rolling(self.N, min_periods=1).quantile(self.qscore)
        return series


class Med(Rolling):
    """滚动中位数

    参数
    ----------
    feature : Expression
        特征实例
    N : int
        滚动窗口大小

    返回
    ----------
    Expression
        一个带滚动中位数的特征实例
    """

    def __init__(self, feature, N):
        super(Med, self).__init__(feature, N, "median")


class Mad(Rolling):
    """滚动平均绝对偏差

    参数
    ----------
    feature : Expression
        特征实例
    N : int
        滚动窗口大小

    返回
    ----------
    Expression
        一个带滚动平均绝对偏差的特征实例
    """

    def __init__(self, feature, N):
        super(Mad, self).__init__(feature, N, "mad")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        def mad(x):
            x1 = x[~np.isnan(x)]
            return np.mean(np.abs(x1 - x1.mean()))

        if self.N == 0:
            series = series.expanding(min_periods=1).apply(mad, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(mad, raw=True)
        return series


class Rank(Rolling):
    """滚动排名（百分位数）

    参数
    ----------
    feature : Expression
        特征实例
    N : int
        滚动窗口大小

    返回
    ----------
    Expression
        一个带滚动排名的特征实例
    """

    def __init__(self, feature, N):
        super(Rank, self).__init__(feature, N, "rank")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)

        rolling_or_expending = series.expanding(min_periods=1) if self.N == 0 else series.rolling(self.N, min_periods=1)
        if hasattr(rolling_or_expending, "rank"):
            return rolling_or_expending.rank(pct=True)

        def rank(x):
            if np.isnan(x[-1]):
                return np.nan
            x1 = x[~np.isnan(x)]
            if x1.shape[0] == 0:
                return np.nan
            return percentileofscore(x1, x1[-1]) / 100

        return rolling_or_expending.apply(rank, raw=True)


class Count(Rolling):
    """滚动计数

    参数
    ----------
    feature : Expression
        特征实例
    N : int
        滚动窗口大小

    返回
    ----------
    Expression
        一个带滚动非 NaN 元素计数的特征实例
    """

    def __init__(self, feature, N):
        super(Count, self).__init__(feature, N, "count")


class Delta(Rolling):
    """滚动差值

    参数
    ----------
    feature : Expression
        特征实例
    N : int
        滚动窗口大小

    返回
    ----------
    Expression
        一个带滚动窗口中结束值减去开始值的特征实例
    """

    def __init__(self, feature, N):
        super(Delta, self).__init__(feature, N, "delta")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = series - series.iloc[0]
        else:
            series = series - series.shift(self.N)
        return series


class Slope(Rolling):
    """滚动斜率
    此运算符计算 `idx` 和 `feature` 之间的斜率。
    （例如 [<feature_t1>, <feature_t2>, <feature_t3>] 和 [1, 2, 3]）

    用法示例：
    - "Slope($close, %d)/$close"

    参数
    ----------
    feature : Expression
        特征实例
    N : int
        滚动窗口大小

    返回
    ----------
    Expression
        一个带给定窗口线性回归斜率的特征实例
    """

    def __init__(self, feature, N):
        super(Slope, self).__init__(feature, N, "slope")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = pd.Series(expanding_slope(series.values), index=series.index)
        else:
            series = pd.Series(rolling_slope(series.values, self.N), index=series.index)
        return series


class Rsquare(Rolling):
    """滚动 R 平方值

    参数
    ----------
    feature : Expression
        特征实例
    N : int
        滚动窗口大小

    返回
    ----------
    Expression
        一个带给定窗口线性回归 R 平方值的特征实例
    """

    def __init__(self, feature, N):
        super(Rsquare, self).__init__(feature, N, "rsquare")

    def _load_internal(self, instrument, start_index, end_index, *args):
        _series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = pd.Series(expanding_rsquare(_series.values), index=_series.index)
        else:
            series = pd.Series(rolling_rsquare(_series.values, self.N), index=_series.index)
            series.loc[np.isclose(_series.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)] = np.nan
        return series


class Resi(Rolling):
    """滚动回归残差

    参数
    ----------
    feature : Expression
        特征实例
    N : int
        滚动窗口大小

    返回
    ----------
    Expression
        一个带给定窗口回归残差的特征实例
    """

    def __init__(self, feature, N):
        super(Resi, self).__init__(feature, N, "resi")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = pd.Series(expanding_resi(series.values), index=series.index)
        else:
            series = pd.Series(rolling_resi(series.values, self.N), index=series.index)
        return series


class WMA(Rolling):
    """滚动加权移动平均

    参数
    ----------
    feature : Expression
        特征实例
    N : int
        滚动窗口大小

    返回
    ----------
    Expression
        一个带加权移动平均输出的特征实例
    """

    def __init__(self, feature, N):
        super(WMA, self).__init__(feature, N, "wma")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        def weighted_mean(x):
            w = np.arange(len(x)) + 1
            w = w / w.sum()
            return np.nanmean(w * x)

        if self.N == 0:
            series = series.expanding(min_periods=1).apply(weighted_mean, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(weighted_mean, raw=True)
        return series


class EMA(Rolling):
    """滚动指数移动平均 (EMA)

    参数
    ----------
    feature : Expression
        特征实例
    N : int, float
        滚动窗口大小

    返回
    ----------
    Expression
        一个带指数移动平均输出的特征实例
    """

    def __init__(self, feature, N):
        super(EMA, self).__init__(feature, N, "ema")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)

        def exp_weighted_mean(x):
            a = 1 - 2 / (1 + len(x))
            w = a ** np.arange(len(x))[::-1]
            w /= w.sum()
            return np.nansum(w * x)

        if self.N == 0:
            series = series.expanding(min_periods=1).apply(exp_weighted_mean, raw=True)
        elif 0 < self.N < 1:
            series = series.ewm(alpha=self.N, min_periods=1).mean()
        else:
            series = series.ewm(span=self.N, min_periods=1).mean()
        return series


#################### 双特征滚动 ####################
class PairRolling(ExpressionOps):
    """双特征滚动运算符

    参数
    ----------
    feature_left : Expression
        特征实例
    feature_right : Expression
        特征实例
    N : int
        滚动窗口大小

    返回
    ----------
    Expression
        一个带两个输入特征滚动输出的特征实例
    """

    def __init__(self, feature_left, feature_right, N, func):
        self.feature_left = feature_left
        self.feature_right = feature_right
        self.N = N
        self.func = func

    def __str__(self):
        return "{}({},{},{})".format(type(self).__name__, self.feature_left, self.feature_right, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        assert any(
            [isinstance(self.feature_left, Expression), self.feature_right, Expression]
        ), "两个输入中至少有一个是 Expression 实例"

        if isinstance(self.feature_left, Expression):
            series_left = self.feature_left.load(instrument, start_index, end_index, *args)
        else:
            series_left = self.feature_left
        if isinstance(self.feature_right, Expression):
            series_right = self.feature_right.load(instrument, start_index, end_index, *args)
        else:
            series_right = self.feature_right

        if self.N == 0:
            series = getattr(series_left.expanding(min_periods=1), self.func)(series_right)
        else:
            series = getattr(series_left.rolling(self.N, min_periods=1), self.func)(series_right)
        return series

    def get_longest_back_rolling(self):
        if self.N == 0:
            return np.inf
        if isinstance(self.feature_left, Expression):
            left_br = self.feature_left.get_longest_back_rolling()
        else:
            left_br = 0

        if isinstance(self.feature_right, Expression):
            right_br = self.feature_right.get_longest_back_rolling()
        else:
            right_br = 0
        return max(left_br, right_br)

    def get_extended_window_size(self):
        if isinstance(self.feature_left, Expression):
            ll, lr = self.feature_left.get_extended_window_size()
        else:
            ll, lr = 0, 0
        if isinstance(self.feature_right, Expression):
            rl, rr = self.feature_right.get_extended_window_size()
        else:
            rl, rr = 0, 0
        if self.N == 0:
            get_module_logger(self.__class__.__name__).warning(
                "PairRolling(ATTR, 0) 的计算可能不准确"
            )
            return -np.inf, max(lr, rr)
        else:
            return max(ll, rl) + self.N - 1, max(lr, rr)


class Corr(PairRolling):
    """滚动相关性

    参数
    ----------
    feature_left : Expression
        特征实例
    feature_right : Expression
        特征实例
    N : int
        滚动窗口大小

    返回
    ----------
    Expression
        一个带两个输入特征滚动相关性的特征实例
    """

    def __init__(self, feature_left, feature_right, N):
        super(Corr, self).__init__(feature_left, feature_right, N, "corr")

    def _load_internal(self, instrument, start_index, end_index, *args):
        res: pd.Series = super(Corr, self)._load_internal(instrument, start_index, end_index, *args)

        series_left = self.feature_left.load(instrument, start_index, end_index, *args)
        series_right = self.feature_right.load(instrument, start_index, end_index, *args)
        res.loc[
            np.isclose(series_left.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)
            | np.isclose(series_right.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)
        ] = np.nan
        return res


class Cov(PairRolling):
    """滚动协方差

    参数
    ----------
    feature_left : Expression
        特征实例
    feature_right : Expression
        特征实例
    N : int
        滚动窗口大小

    返回
    ----------
    Expression
        一个带两个输入特征滚动协方差的特征实例
    """

    def __init__(self, feature_left, feature_right, N):
        super(Cov, self).__init__(feature_left, feature_right, N, "cov")


#################### 仅支持带时间索引的数据的运算符 ####################
class TResample(ElemOperator):
    def __init__(self, feature, freq, func):
        """
        将数据重采样到目标频率。
        使用 pandas 的 resample 函数。

        - 重采样后，时间戳将位于时间跨度的开始。

        参数
        ----------
        feature : Expression
            用于计算特征的表达式
        freq : str
            它将被传递到 resample 方法中，以基于给定频率进行重采样
        func : method
            获取重采样值的方法
        """
        self.feature = feature
        self.freq = freq
        self.func = func

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature, self.freq)

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)

        if series.empty:
            return series
        else:
            if self.func == "sum":
                return getattr(series.resample(self.freq), self.func)(min_count=1)
            else:
                return getattr(series.resample(self.freq), self.func)()


TOpsList = [TResample]
OpsList = [
    ChangeInstrument,
    Rolling,
    Ref,
    Max,
    Min,
    Sum,
    Mean,
    Std,
    Var,
    Skew,
    Kurt,
    Med,
    Mad,
    Slope,
    Rsquare,
    Resi,
    Rank,
    Quantile,
    Count,
    EMA,
    WMA,
    Corr,
    Cov,
    Delta,
    Abs,
    Sign,
    Log,
    Power,
    Add,
    Sub,
    Mul,
    Div,
    Greater,
    Less,
    And,
    Or,
    Not,
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Ne,
    Mask,
    IdxMax,
    IdxMin,
    If,
    Feature,
    PFeature,
] + [TResample]


class OpsWrapper:
    """运算符包装器"""

    def __init__(self):
        self._ops = {}

    def reset(self):
        self._ops = {}

    def register(self, ops_list: List[Union[Type[ExpressionOps], dict]]):
        """注册运算符

        参数
        ----------
        ops_list : List[Union[Type[ExpressionOps], dict]]
            - 如果 type(ops_list) 是 List[Type[ExpressionOps]]，ops_list 的每个元素代表运算符类，它应该是 `ExpressionOps` 的子类。
            - 如果 type(ops_list) 是 List[dict]，ops_list 的每个元素代表运算符的配置，格式如下：

                .. code-block:: text

                    {
                        "class": class_name,
                        "module_path": path,
                    }

                注意：`class` 应该是运算符的类名，`module_path` 应该是一个 python 模块或文件路径。
        """
        for _operator in ops_list:
            if isinstance(_operator, dict):
                _ops_class, _ = get_callable_kwargs(_operator)
            else:
                _ops_class = _operator

            if not issubclass(_ops_class, (Expression,)):
                raise TypeError("运算符必须是 ExpressionOps 的子类，而不是 {}".format(_ops_class))

            if _ops_class.__name__ in self._ops:
                get_module_logger(self.__class__.__name__).warning(
                    "自定义运算符 [{}] 将覆盖 qlib 默认定义".format(_ops_class.__name__)
                )
            self._ops[_ops_class.__name__] = _ops_class

    def __getattr__(self, key):
        if key not in self._ops:
            raise AttributeError("运算符 [{0}] 未注册".format(key))
        return self._ops[key]


Operators = OpsWrapper()


def register_all_ops(C):
    """注册所有运算符"""
    logger = get_module_logger("ops")

    from qlib.data.pit import P, PRef

    Operators.reset()
    Operators.register(OpsList + [P, PRef])

    if getattr(C, "custom_ops", None) is not None:
        Operators.register(C.custom_ops)
        logger.debug("注册自定义运算符 {}".format(C.custom_ops))
