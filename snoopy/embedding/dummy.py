from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import torch as pt

from .base import EmbeddingModel, EmbeddingModelSpec, _ImageTransformationHelper as Ith
from ..custom_types import DataType


@dataclass
class DummySpec(EmbeddingModelSpec):
    output_dimension: int

    def load(self, device: pt.device) -> EmbeddingModel:
        return _DummyInner(self, device)

    @property
    def data_type(self) -> DataType:
        return DataType.ANY


class _DummyInner(EmbeddingModel):
    def __init__(self, spec: DummySpec, device: pt.device):
        self._output_dimension = spec.output_dimension
        self._device = device

    def move_to(self, device: pt.device) -> None:
        self._device = device

    @property
    def output_dimension(self) -> int:
        return self._output_dimension

    def get_data_preparation_function(self) -> Callable:
        return lambda x: x

    def apply_embedding(self, features: np.ndarray) -> pt.Tensor:
        return pt.as_tensor(features, device=self._device)


@dataclass
class ImageReshapeSpec(EmbeddingModelSpec):
    target_image_size: Tuple[int, int]
    num_channels: int

    def load(self, device: pt.device) -> EmbeddingModel:
        return _ImageReshapeInner(self, device)

    @property
    def data_type(self) -> DataType:
        return DataType.IMAGE


class _ImageReshapeInner(EmbeddingModel):
    def __init__(self, spec: ImageReshapeSpec, device: pt.device):
        self._device = device
        self._target_image_size = spec.target_image_size
        self._num_channels = spec.num_channels

    def move_to(self, device: pt.device) -> None:
        self._device = device

    @property
    def output_dimension(self) -> int:
        return self._target_image_size[0] * self._target_image_size[1] * self._num_channels

    def get_data_preparation_function(self) -> Callable:
        return lambda x: Ith.raw_image_with_central_crop_and_resize(x, self._target_image_size)

    def apply_embedding(self, features: np.ndarray) -> pt.Tensor:
        return pt.as_tensor(features.squeeze(1), device=self._device)
