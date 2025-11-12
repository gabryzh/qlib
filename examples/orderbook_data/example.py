# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from arctic.arctic import Arctic
import qlib
from qlib.data import D
import unittest


class TestClass(unittest.TestCase):
    """
    有用的命令
    - 运行所有测试: pytest examples/orderbook_data/example.py
    - 运行单个测试:  pytest -s --pdb --disable-warnings examples/orderbook_data/example.py::TestClass::test_basic01
    """

    def setUp(self):
        """
        为 arctic 进行配置
        """
        provider_uri = "~/.qlib/qlib_data/yahoo_cn_1min"
        qlib.init(
            provider_uri=provider_uri,
            mem_cache_size_limit=1024**3 * 2,
            mem_cache_type="sizeof",
            kernels=1,
            expression_provider={"class": "LocalExpressionProvider", "kwargs": {"time2idx": False}},
            feature_provider={
                "class": "ArcticFeatureProvider",
                "module_path": "qlib.contrib.data.data",
                "kwargs": {"uri": "127.0.0.1"},
            },
            dataset_provider={
                "class": "LocalDatasetProvider",
                "kwargs": {
                    "align_time": False,  # 订单簿不是固定的，因此无法与共享的固定频率日历对齐
                },
            },
        )
        self.stocks_list = ["SZ000725"]

    def test_basic(self):
        """测试基础功能"""
        # 注意：此数据在 $askX 和 $bidX 中包含大量零
        df = D.features(
            self.stocks_list,
            fields=["$ask1", "$ask2", "$bid1", "$bid2"],
            freq="ticks",
            start_time="20201230",
            end_time="20210101",
        )
        print(df)

    def test_basic_without_time(self):
        """测试没有时间的基础功能"""
        df = D.features(self.stocks_list, fields=["$ask1"], freq="ticks")
        print(df)

    def test_basic01(self):
        """测试基础功能 01"""
        df = D.features(
            self.stocks_list,
            fields=["TResample($ask1, '1min', 'last')"],
            freq="ticks",
            start_time="20201230",
            end_time="20210101",
        )
        print(df)

    def test_basic02(self):
        """测试基础功能 02"""
        df = D.features(
            self.stocks_list,
            fields=["$function_code"],
            freq="transaction",
            start_time="20201230",
            end_time="20210101",
        )
        print(df)

    def test_basic03(self):
        """测试基础功能 03"""
        df = D.features(
            self.stocks_list,
            fields=["$function_code"],
            freq="order",
            start_time="20201230",
            end_time="20210101",
        )
        print(df)

    # 以下是一些流行的高频表达式
    # 1) 一些共享表达式
    expr_sum_buy_ask_1 = "(TResample($ask1, '1min', 'last') + TResample($bid1, '1min', 'last'))"
    total_volume = (
        "TResample("
        + "+".join([f"${name}{i}" for i in range(1, 11) for name in ["asize", "bsize"]])
        + ", '1min', 'sum')"
    )

    @staticmethod
    def total_func(name, method):
        return "TResample(" + "+".join([f"${name}{i}" for i in range(1, 11)]) + ",'1min', '{}')".format(method)

    def test_exp_01(self):
        """测试表达式 01"""
        exprs = []
        names = []
        for name in ["asize", "bsize"]:
            for i in range(1, 11):
                exprs.append(f"TResample(${name}{i}, '1min', 'mean') / ({self.total_volume})")
                names.append(f"v_{name}_{i}")
        df = D.features(self.stocks_list, fields=exprs, freq="ticks")
        df.columns = names
        print(df)

    def test_exp_02(self):
        """测试表达式 02"""
        spread_func = (
            lambda index: f"2 * TResample($ask{index} - $bid{index}, '1min', 'last') / {self.expr_sum_buy_ask_1}"
        )
        mid_func = (
            lambda index: f"2 * TResample(($ask{index} + $bid{index})/2, '1min', 'last') / {self.expr_sum_buy_ask_1}"
        )

        exprs = []
        names = []
        for i in range(1, 11):
            exprs.extend([spread_func(i), mid_func(i)])
            names.extend([f"p_spread_{i}", f"p_mid_{i}"])
        df = D.features(self.stocks_list, fields=exprs, freq="ticks")
        df.columns = names
        print(df)

    def test_exp_03(self):
        """测试表达式 03"""
        expr3_func1 = (
            lambda name, index_left, index_right: f"2 * TResample(Abs(${name}{index_left} - ${name}{index_right}), '1min', 'last') / {self.expr_sum_buy_ask_1}"
        )
        for name in ["ask", "bid"]:
            for i in range(1, 10):
                exprs = [expr3_func1(name, i + 1, i)]
                names = [f"p_diff_{name}_{i}_{i+1}"]
        exprs.extend([expr3_func1("ask", 10, 1), expr3_func1("bid", 1, 10)])
        names.extend(["p_diff_ask_10_1", "p_diff_bid_1_10"])
        df = D.features(self.stocks_list, fields=exprs, freq="ticks")
        df.columns = names
        print(df)

    def test_exp_04(self):
        """测试表达式 04"""
        exprs = []
        names = []
        for name in ["asize", "bsize"]:
            exprs.append(f"(({ self.total_func(name, 'mean')}) / 10) / {self.total_volume}")
            names.append(f"v_avg_{name}")

        df = D.features(self.stocks_list, fields=exprs, freq="ticks")
        df.columns = names
        print(df)

    def test_exp_05(self):
        """测试表达式 05"""
        exprs = [
            f"2 * Sub({ self.total_func('ask', 'last')}, {self.total_func('bid', 'last')})/{self.expr_sum_buy_ask_1}",
            f"Sub({ self.total_func('asize', 'mean')}, {self.total_func('bsize', 'mean')})/{self.total_volume}",
        ]
        names = ["p_accspread", "v_accspread"]

        df = D.features(self.stocks_list, fields=exprs, freq="ticks")
        df.columns = names
        print(df)

    def test_exp_06(self):
        """测试表达式 06"""
        t = 3
        expr6_price_func = (
            lambda name, index, method: f'2 * (TResample(${name}{index}, "{t}s", "{method}") - Ref(TResample(${name}{index}, "{t}s", "{method}"), 1)) / {t}'
        )
        exprs = []
        names = []
        for i in range(1, 11):
            for name in ["bid", "ask"]:
                exprs.append(
                    f"TResample({expr6_price_func(name, i, 'last')}, '1min', 'mean') / {self.expr_sum_buy_ask_1}"
                )
                names.append(f"p_diff_{name}{i}_{t}s")

        for i in range(1, 11):
            for name in ["asize", "bsize"]:
                exprs.append(f"TResample({expr6_price_func(name, i, 'mean')}, '1min', 'mean') / {self.total_volume}")
                names.append(f"v_diff_{name}{i}_{t}s")

        df = D.features(self.stocks_list, fields=exprs, freq="ticks")
        df.columns = names
        print(df)

    @staticmethod
    def expr7_init(funccode, ordercode, time_interval):
        # 注意：基于订单频率 (即 freq="order")
        return f"Rolling(Eq($function_code,  {ord(funccode)}) & Eq($order_kind ,{ord(ordercode)}), '{time_interval}s', 'sum') / Rolling($function_code, '{time_interval}s', 'count')"

    def test_exp_07_1(self):
        """测试表达式 07_1"""
        # 注意：基于交易频率 (即 freq="transaction")
        expr7_3 = (
            lambda funccode, code, time_interval: f"TResample(Rolling(Eq($function_code,  {ord(funccode)}) & {code}($ask_order, $bid_order) , '{time_interval}s', 'sum')   / Rolling($function_code, '{time_interval}s', 'count') , '1min', 'mean')"
        )

        exprs = [expr7_3("C", "Gt", "3"), expr7_3("C", "Lt", "3")]
        names = ["ca_intensity_3s", "cb_intensity_3s"]

        df = D.features(self.stocks_list, fields=exprs, freq="transaction")
        df.columns = names
        print(df)

    trans_dict = {"B": "a", "S": "b", "0": "l", "1": "m"}

    def test_exp_07_2(self):
        """测试表达式 07_2"""
        # 注意：基于订单频率
        expr7 = (
            lambda funccode, ordercode, time_interval: f"TResample({self.expr7_init(funccode, ordercode, time_interval)}, '1min', 'mean')"
        )

        exprs = []
        names = []
        for funccode in ["B", "S"]:
            for ordercode in ["0", "1"]:
                exprs.append(expr7(funccode, ordercode, "3"))
                names.append(self.trans_dict[ordercode] + self.trans_dict[funccode] + "_intensity_3s")
        df = D.features(self.stocks_list, fields=exprs, freq="transaction")
        df.columns = names
        print(df)

    @staticmethod
    def expr7_3_init(funccode, code, time_interval):
        # 注意：它取决于交易频率
        return f"Rolling(Eq($function_code,  {ord(funccode)}) & {code}($ask_order, $bid_order) , '{time_interval}s', 'sum') / Rolling($function_code, '{time_interval}s', 'count')"

    def test_exp_08_1(self):
        """测试表达式 08_1"""
        expr8_1 = (
            lambda funccode, ordercode, time_interval_short, time_interval_long: f"TResample(Gt({self.expr7_init(funccode, ordercode, time_interval_short)},{self.expr7_init(funccode, ordercode, time_interval_long)}), '1min', 'mean')"
        )

        exprs = []
        names = []
        for funccode in ["B", "S"]:
            for ordercode in ["0", "1"]:
                exprs.append(expr8_1(funccode, ordercode, "10", "900"))
                names.append(self.trans_dict[ordercode] + self.trans_dict[funccode] + "_relative_intensity_10s_900s")

        df = D.features(self.stocks_list, fields=exprs, freq="order")
        df.columns = names
        print(df)

    def test_exp_08_2(self):
        """测试表达式 08_2"""
        # 注意：它取决于交易频率
        expr8_2 = (
            lambda funccode, ordercode, time_interval_short, time_interval_long: f"TResample(Gt({self.expr7_3_init(funccode, ordercode, time_interval_short)},{self.expr7_3_init(funccode, ordercode, time_interval_long)}), '1min', 'mean')"
        )

        exprs = [expr8_2("C", "Gt", "10", "900"), expr8_2("C", "Lt", "10", "900")]
        names = ["ca_relative_intensity_10s_900s", "cb_relative_intensity_10s_900s"]

        df = D.features(self.stocks_list, fields=exprs, freq="transaction")
        df.columns = names
        print(df)

    def test_exp_09_trans(self):
        """测试表达式 09_trans"""
        exprs = [
            f'TResample(Div(Sub(TResample({self.expr7_3_init("C", "Gt", "3")}, "3s", "last"), Ref(TResample({self.expr7_3_init("C", "Gt", "3")}, "3s","last"), 1)), 3), "1min", "mean")',
            f'TResample(Div(Sub(TResample({self.expr7_3_init("C", "Lt", "3")}, "3s", "last"), Ref(TResample({self.expr7_3_init("C", "Lt", "3")}, "3s","last"), 1)), 3), "1min", "mean")',
        ]
        names = ["ca_diff_intensity_3s_3s", "cb_diff_intensity_3s_3s"]
        df = D.features(self.stocks_list, fields=exprs, freq="transaction")
        df.columns = names
        print(df)

    def test_exp_09_order(self):
        """测试表达式 09_order"""
        exprs = []
        names = []
        for funccode in ["B", "S"]:
            for ordercode in ["0", "1"]:
                exprs.append(
                    f'TResample(Div(Sub(TResample({self.expr7_init(funccode, ordercode, "3")}, "3s", "last"), Ref(TResample({self.expr7_init(funccode, ordercode, "3")},"3s", "last"), 1)), 3) ,"1min", "mean")'
                )
                names.append(self.trans_dict[ordercode] + self.trans_dict[funccode] + "_diff_intensity_3s_3s")
        df = D.features(self.stocks_list, fields=exprs, freq="order")
        df.columns = names
        print(df)

    def test_exp_10(self):
        """测试表达式 10"""
        exprs = []
        names = []
        for i in [5, 10, 30, 60]:
            exprs.append(
                f'TResample(Ref(TResample($ask1 + $bid1, "1s", "ffill"), {-i}) / TResample($ask1 + $bid1, "1s", "ffill") - 1, "1min", "mean" )'
            )
            names.append(f"lag_{i}_change_rate" for i in [5, 10, 30, 60])
        df = D.features(self.stocks_list, fields=exprs, freq="ticks")
        df.columns = names
        print(df)


if __name__ == "__main__":
    unittest.main()
