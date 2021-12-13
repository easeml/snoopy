from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
import torch as pt
from efficientnet_pytorch import EfficientNet

from .base import EmbeddingModel, EmbeddingModelSpec, _ImageTransformationHelper as Ith
from .._logging import get_logger
from .._utils import get_tf_device
from ..custom_types import DataType

_logger = get_logger(__name__)


@dataclass
class TorchHubImageSpec(EmbeddingModelSpec):
    name: str
    output_dimension: int
    layer_extractor: Callable
    required_image_size: Tuple[int, int]

    def load(self, device: pt.device) -> EmbeddingModel:
        return _Inner.get_instance(self, device)

    @property
    def data_type(self) -> DataType:
        return DataType.IMAGE


class _Inner(EmbeddingModel):
    _instances = dict()
    _efficientnet_names = set([f"efficientnet-b{i}" for i in range(9)])

    def __init__(self, spec: TorchHubImageSpec, device: pt.device):
        # Prepare model
        if spec.name in self._efficientnet_names:
            _logger.debug(f"Using package EfficientNet-PyTorch for model {spec.name}")
            model = EfficientNet.from_pretrained(spec.name)

        else:
            _logger.debug(f"Using PyTorch vision for model {spec.name}")
            model = pt.hub.load('pytorch/vision:v0.6.0', spec.name, pretrained=True)
        model.eval()
        model.to(device)

        # Prepare hook that will store the result once the model is called
        self._model = model
        self._result = None
        spec.layer_extractor(self._model).register_forward_hook(lambda _x, _y, result: self._store_result(result))

        # Other params
        self._output_dimension = spec.output_dimension
        self._required_image_size = spec.required_image_size
        self._device = device

    @classmethod
    def get_instance(cls, spec: TorchHubImageSpec, device: pt.device) -> EmbeddingModel:
        combination_string = (spec.name, get_tf_device(device))
        if combination_string not in cls._instances:
            _logger.info(f"Initializing {combination_string[0]} on {combination_string[1]}")
            cls._instances[combination_string] = cls(spec, device)

        return cls._instances[combination_string]

    def _store_result(self, result):
        self._result = result

    def move_to(self, device: pt.device) -> None:
        self._model.to(device)
        self._device = device

    @property
    def output_dimension(self) -> int:
        return self._output_dimension

    def get_data_preparation_function(self) -> Callable:
        def fn(feature):
            resized_img = Ith.central_crop_with_resize_3_channels(feature, self._required_image_size)

            # Normalization specified in https://pytorch.org/docs/stable/torchvision/models.html#torchvision-models
            return tf.divide(tf.subtract(resized_img, tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)),
                             tf.constant([0.229, 0.224, 0.225], dtype=tf.float32))

        return fn

    def apply_embedding(self, features: np.ndarray) -> pt.Tensor:
        features_pt = pt.as_tensor(features, dtype=pt.float32, device=self._device)

        with pt.no_grad():
            # Swap dimensions from (batch x H x W x C) to (batch x C x H x W)
            model_input = features_pt.permute(0, 3, 1, 2)

            # Call model
            self._model(model_input)

        # In some cases result has more than 2 dimensions, so squeeze must be used
        return_value = self._result.squeeze()

        # TODO: unit test testing that
        # If there was only 1 point in batch, 1. dimension got removed too, so it should be introduced again
        if len(return_value.size()) == 1:
            return_value = return_value.unsqueeze(0)

        return return_value
