# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import torch
import numpy as np
import pandas as pd

from qlib.data.dataset import DatasetH


device = "cuda" if torch.cuda.is_available() else "cpu"


def _to_tensor(x):
    """将输入转换为张量"""
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=torch.float, device=device)
    return x


def _create_ts_slices(index, seq_len):
    """
    从 pandas 索引创建时间序列切片

    Args:
        index (pd.MultiIndex): <instrument, datetime> 顺序的 pandas 多重索引
        seq_len (int): 序列长度
    """
    assert index.is_lexsorted(), "索引应已排序"

    # 每个代码的日期数
    sample_count_by_codes = pd.Series(0, index=index).groupby(level=0, group_keys=False).size().values

    # 每个代码的 start_index
    start_index_of_codes = np.roll(np.cumsum(sample_count_by_codes), 1)
    start_index_of_codes[0] = 0

    # 所有特征的 [start, stop) 索引
    # [start, stop) 之间的特征用于预测 `stop - 1` 标签
    slices = []
    for cur_loc, cur_cnt in zip(start_index_of_codes, sample_count_by_codes):
        for stop in range(1, cur_cnt + 1):
            end = cur_loc + stop
            start = max(end - seq_len, 0)
            slices.append(slice(start, end))
    slices = np.array(slices)

    return slices


def _get_date_parse_fn(target):
    """获取日期解析函数

    此方法用于将日期参数解析为目标类型。

    Example:
        get_date_parse_fn('20120101')('2017-01-01') => '20170101'
        get_date_parse_fn(20120101)('2017-01-01') => 20170101
    """
    if isinstance(target, pd.Timestamp):
        _fn = lambda x: pd.Timestamp(x)  # Timestamp('2020-01-01')
    elif isinstance(target, str) and len(target) == 8:
        _fn = lambda x: str(x).replace("-", "")[:8]  # '20200201'
    elif isinstance(target, int):
        _fn = lambda x: int(str(x).replace("-", "")[:8])  # 20200201
    else:
        _fn = lambda x: x
    return _fn


class MTSDatasetH(DatasetH):
    """内存增强时间序列数据集

    Args:
        handler (DataHandler): 数据处理器
        segments (dict): 数据拆分段
        seq_len (int): 时间序列序列长度
        horizon (int): 标签范围（用于为 TRA 屏蔽历史损失）
        num_states (int): 要添加的内存状态数（用于 TRA）
        batch_size (int): 批量大小（<0 表示每日批量）
        shuffle (bool): 是否打乱数据
        pin_memory (bool): 是否将数据固定到 gpu 内存
        drop_last (bool): 是否丢弃最后一个小于 batch_size 的批次
    """

    def __init__(
        self,
        handler,
        segments,
        seq_len=60,
        horizon=0,
        num_states=1,
        batch_size=-1,
        shuffle=True,
        pin_memory=False,
        drop_last=False,
        **kwargs,
    ):
        assert horizon > 0, "请指定 `horizon` 以避免数据泄漏"

        self.seq_len = seq_len
        self.horizon = horizon
        self.num_states = num_states
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.params = (batch_size, drop_last, shuffle)  # 用于训练/评估切换

        super().__init__(handler, segments, **kwargs)

    def setup_data(self, handler_kwargs: dict = None, **kwargs):
        """设置数据"""
        super().setup_data()

        # 将索引更改为 <code, date>
        # 注意：我们将使用就地排序来减少内存使用
        df = self.handler._data
        df.index = df.index.swaplevel()
        df.sort_index(inplace=True)

        self._data = df["feature"].values.astype("float32")
        self._label = df["label"].squeeze().astype("float32")
        self._index = df.index

        # 将内存添加到特征
        self._data = np.c_[self._data, np.zeros((len(self._data), self.num_states), dtype=np.float32)]

        # 填充张量
        self.zeros = np.zeros((self.seq_len, self._data.shape[1]), dtype=np.float32)

        # 固定内存
        if self.pin_memory:
            self._data = _to_tensor(self._data)
            self._label = _to_tensor(self._label)
            self.zeros = _to_tensor(self.zeros)

        # 创建批处理切片
        self.batch_slices = _create_ts_slices(self._index, self.seq_len)

        # 创建每日切片
        index = [slc.stop - 1 for slc in self.batch_slices]
        act_index = self.restore_index(index)
        daily_slices = {date: [] for date in sorted(act_index.unique(level=1))}
        for i, (code, date) in enumerate(act_index):
            daily_slices[date].append(self.batch_slices[i])
        self.daily_slices = list(daily_slices.values())

    def _prepare_seg(self, slc, **kwargs):
        """准备数据段"""
        fn = _get_date_parse_fn(self._index[0][1])

        if isinstance(slc, slice):
            start, stop = slc.start, slc.stop
        elif isinstance(slc, (list, tuple)):
            start, stop = slc
        else:
            raise NotImplementedError(f"不支持此类型的输入")
        start_date = fn(start)
        end_date = fn(stop)
        obj = copy.copy(self)  # 浅拷贝
        # 注意：Seriable 将禁用复制 `self._data`，因此我们在此处手动分配它们
        obj._data = self._data
        obj._label = self._label
        obj._index = self._index
        new_batch_slices = []
        for batch_slc in self.batch_slices:
            date = self._index[batch_slc.stop - 1][1]
            if start_date <= date <= end_date:
                new_batch_slices.append(batch_slc)
        obj.batch_slices = np.array(new_batch_slices)
        new_daily_slices = []
        for daily_slc in self.daily_slices:
            date = self._index[daily_slc[0].stop - 1][1]
            if start_date <= date <= end_date:
                new_daily_slices.append(daily_slc)
        obj.daily_slices = new_daily_slices
        return obj

    def restore_index(self, index):
        """恢复索引"""
        if isinstance(index, torch.Tensor):
            index = index.cpu().numpy()
        return self._index[index]

    def assign_data(self, index, vals):
        """分配数据"""
        if isinstance(self._data, torch.Tensor):
            vals = _to_tensor(vals)
        elif isinstance(vals, torch.Tensor):
            vals = vals.detach().cpu().numpy()
            index = index.detach().cpu().numpy()
        self._data[index, -self.num_states :] = vals

    def clear_memory(self):
        """清除内存"""
        self._data[:, -self.num_states :] = 0

    # TODO: 更好的训练/评估模式设计
    def train(self):
        """启用训练模式"""
        self.batch_size, self.drop_last, self.shuffle = self.params

    def eval(self):
        """启用评估模式"""
        self.batch_size = -1
        self.drop_last = False
        self.shuffle = False

    def _get_slices(self):
        """获取切片"""
        if self.batch_size < 0:
            slices = self.daily_slices.copy()
            batch_size = -1 * self.batch_size
        else:
            slices = self.batch_slices.copy()
            batch_size = self.batch_size
        return slices, batch_size

    def __len__(self):
        slices, batch_size = self._get_slices()
        if self.drop_last:
            return len(slices) // batch_size
        return (len(slices) + batch_size - 1) // batch_size

    def __iter__(self):
        slices, batch_size = self._get_slices()
        if self.shuffle:
            np.random.shuffle(slices)

        for i in range(len(slices))[::batch_size]:
            if self.drop_last and i + batch_size > len(slices):
                break
            # 获取此批次的切片
            slices_subset = slices[i : i + batch_size]
            if self.batch_size < 0:
                slices_subset = np.concatenate(slices_subset)
            # 收集数据
            data = []
            label = []
            index = []
            for slc in slices_subset:
                _data = self._data[slc].clone() if self.pin_memory else self._data[slc].copy()
                if len(_data) != self.seq_len:
                    if self.pin_memory:
                        _data = torch.cat([self.zeros[: self.seq_len - len(_data)], _data], axis=0)
                    else:
                        _data = np.concatenate([self.zeros[: self.seq_len - len(_data)], _data], axis=0)
                if self.num_states > 0:
                    _data[-self.horizon :, -self.num_states :] = 0
                data.append(_data)
                label.append(self._label[slc.stop - 1])
                index.append(slc.stop - 1)
            # 连接
            index = torch.tensor(index, device=device)
            if isinstance(data[0], torch.Tensor):
                data = torch.stack(data)
                label = torch.stack(label)
            else:
                data = _to_tensor(np.stack(data))
                label = _to_tensor(np.stack(label))
            # yield -> 生成器
            yield {"data": data, "label": label, "index": index}
