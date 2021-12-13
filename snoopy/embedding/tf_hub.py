from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
import torch as pt

from .base import EmbeddingModel, EmbeddingModelSpec, _ImageTransformationHelper as Ith
from .._logging import get_logger
from .._utils import get_tf_device
from ..custom_types import DataType

_logger = get_logger(__name__)


@dataclass
class TFHubImageSpec(EmbeddingModelSpec):
    url: str
    output_dimension: int
    required_image_size: Tuple[int, int]

    def load(self, device: pt.device) -> EmbeddingModel:
        return _ImageInner.get_instance(self, device)

    @property
    def data_type(self) -> DataType:
        return DataType.IMAGE


class _ImageInner(EmbeddingModel):
    _instances = dict()

    def __init__(self, spec: TFHubImageSpec, device: pt.device):
        self._device = device
        self._tf_device = get_tf_device(device)
        with tf.device(self._tf_device):
            self._model = tf_hub.KerasLayer(spec.url, trainable=False)
        self._required_image_size = spec.required_image_size
        self._output_dimension = spec.output_dimension

    @classmethod
    def get_instance(cls, spec: TFHubImageSpec, device: pt.device) -> EmbeddingModel:
        combination_string = (spec.url, get_tf_device(device))
        if combination_string not in cls._instances:
            _logger.info(f"Initializing {combination_string[0]} on {combination_string[1]}")
            cls._instances[combination_string] = cls(spec, device)

        return cls._instances[combination_string]

    def move_to(self, device: pt.device) -> None:
        self._device = device
        self._tf_device = get_tf_device(device)

    @property
    def output_dimension(self) -> int:
        return self._output_dimension

    def get_data_preparation_function(self) -> Callable:
        return lambda feature: Ith.central_crop_with_resize_3_channels(feature, self._required_image_size)

    def apply_embedding(self, features: np.ndarray) -> pt.Tensor:
        with tf.device(self._tf_device):
            tf_tensor = tf.convert_to_tensor(features)
            tf_result = self._model(tf_tensor)

        return pt.as_tensor(tf_result.numpy(), device=self._device)


@dataclass
class TFHubTextSpec(EmbeddingModelSpec):
    url: str
    output_dimension: int

    def load(self, device: pt.device) -> EmbeddingModel:
        return _TextInner.get_instance(self, device)

    @property
    def data_type(self) -> DataType:
        return DataType.TEXT


class _TextInner(EmbeddingModel):
    _instances = dict()

    def __init__(self, spec: TFHubTextSpec, device: pt.device):
        self._device = device
        self._tf_device = get_tf_device(device)
        with tf.device(self._tf_device):
            self._model = tf_hub.KerasLayer(spec.url, trainable=False)
        self._output_dimension = spec.output_dimension

    @classmethod
    def get_instance(cls, spec: TFHubTextSpec, device: pt.device) -> EmbeddingModel:
        combination_string = (spec.url, get_tf_device(device))
        if combination_string not in cls._instances:
            _logger.info(f"Initializing {combination_string[0]} on {combination_string[1]}")
            cls._instances[combination_string] = cls(spec, device)

        return cls._instances[combination_string]

    def move_to(self, device: pt.device) -> None:
        self._device = device
        self._tf_device = get_tf_device(device)

    @property
    def output_dimension(self) -> int:
        return self._output_dimension

    def get_data_preparation_function(self) -> Callable:
        return lambda x: x

    def apply_embedding(self, features: np.ndarray) -> pt.Tensor:
        with tf.device(self._tf_device):
            tf_tensor = tf.convert_to_tensor(features)
            tf_result = self._model(tf_tensor)

        return pt.as_tensor(tf_result.numpy(), device=self._device)
