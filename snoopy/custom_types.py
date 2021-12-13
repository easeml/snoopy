from __future__ import annotations

from enum import Enum
from typing import NamedTuple

import tensorflow as tf
import torch as pt

PT2dTensor = pt.Tensor


# TODO: move classes to appropriate files
class EmbeddingSlice(NamedTuple):
    features: pt.Tensor
    labels: pt.Tensor


class PartialResult:
    def __init__(self, values: pt.Tensor, labels: pt.Tensor):
        assert values.shape == labels.shape, \
            f"Values and labels should have same shape, but have values: {values.shape}, labels: {labels.shape}"
        assert len(values.shape) == 2, \
            f"Values and labels must be 2-dimensional, but are {len(values.shape)}-dimensional"
        assert values.dtype == pt.float32, f"Values should have dtype float32, but have {values.dtype}"
        assert labels.dtype == pt.int64, f"Labels should have dtype int64, but have {labels.dtype}"

        self.values = values
        self.labels = labels

    @property
    def size(self):
        return self.values.shape

    def __repr__(self):
        return f"PartialResult with size: {self.size}"


class Expandable2D:
    def __init__(self, n: int, dim: int, dtype: pt.dtype, device: pt.device = pt.device("cpu")):
        self._data = pt.zeros((n, dim), dtype=dtype, device=device)
        self._dim = dim
        self._dtype = dtype
        self._device = device
        self._occupancy = 0
        self._max_occupancy = n

    @staticmethod
    def _get_message(underlying, supplied_data):
        return f"Underlying storage: {underlying}, supplied data: {supplied_data}"

    def add(self, new_data: pt.Tensor):
        assert len(new_data.shape) == 2, \
            f"Supplied data must be 2. dimensional, supplied data has {len(new_data.shape)} dimension(s)!"
        num_points_to_add = new_data.shape[0]
        assert new_data.device == self._device, "Supplied data must be on same device as underlying storage! " \
                                                + self._get_message(self._device, new_data.device)
        assert new_data.dtype == self._dtype, "Supplied data must have same dtype as underlying storage! " \
                                              + self._get_message(self._dtype, new_data.dtype)
        assert new_data.shape[1] == self._dim, "Supplied data must have same 2. dimension as underlying storage! " \
                                               + self._get_message(self._dim, new_data.shape[1])
        assert self._occupancy + num_points_to_add <= self._max_occupancy, \
            "Underlying storage is not big enough to add new data!"

        self._data[self._occupancy:self._occupancy + num_points_to_add, :] = new_data
        self._occupancy += num_points_to_add

    @property
    def data(self):
        return self._data.narrow(0, 0, self._occupancy)

    @property
    def size(self):
        return self._occupancy


class DataWithInfo(NamedTuple):
    data: tf.data.Dataset
    size: int
    num_labels: int


class CacheType(Enum):
    DEVICE = 0
    CPU = 1
    NONE = 2


class DataType(Enum):
    IMAGE = "image"
    TEXT = "text"
    ANY = "any"
