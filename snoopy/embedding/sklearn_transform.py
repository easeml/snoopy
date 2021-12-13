from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import torch as pt
from sklearn.decomposition import PCA

from .base import EmbeddingModel, EmbeddingModelSpec, _ImageTransformationHelper as Ith
from .._logging import get_logger
from ..custom_types import DataType

_logger = get_logger(__name__)


# TODO: check that output_dimension is smaller than resized image (but need to take into account also channels)
@dataclass
class PCASpec(EmbeddingModelSpec):
    output_dimension: int
    target_image_size: Tuple[int, int]

    def load(self, device: pt.device) -> EmbeddingModel:
        return _InnerPCA.get_instance(self, device)

    @property
    def data_type(self) -> DataType:
        return DataType.IMAGE


class _InnerPCA(EmbeddingModel):
    _instances = dict()

    def __init__(self, spec: PCASpec, device: pt.device):
        self._output_dimension = spec.output_dimension
        self._target_image_size = spec.target_image_size
        self._device = device

        self._num_apply_embedding_calls = 0
        self._pca = PCA(n_components=spec.output_dimension, svd_solver="full")

        if device.type != "cpu":
            _logger.warning("Running PCA on other device than 'cpu' is not possible. It will be run on CPU.")

    def move_to(self, device: pt.device) -> None:
        if device.type != "cpu":
            _logger.warning("Running PCA on other device than 'cpu' is not possible. It will be run on CPU.")

    @classmethod
    def get_instance(cls, spec: PCASpec, device: pt.device) -> EmbeddingModel:
        combination_string = spec.output_dimension
        if combination_string not in cls._instances:
            _logger.info(f"Initializing PCA with output dimension {combination_string} on CPU")
            cls._instances[combination_string] = cls(spec, device)

        return cls._instances[combination_string]

    @property
    def output_dimension(self) -> int:
        return self._output_dimension

    def get_data_preparation_function(self) -> Callable:
        return lambda x: Ith.raw_image_with_central_crop_and_resize(x, self._target_image_size)

    def apply_embedding(self, features: np.ndarray) -> pt.Tensor:
        if self._num_apply_embedding_calls > 1:
            raise RuntimeError("apply_embedding function can be called only twice!")

        squeezed_features = features.squeeze()

        if self._num_apply_embedding_calls == 0:
            self._pca.fit(squeezed_features)

        self._num_apply_embedding_calls += 1

        return pt.as_tensor(self._pca.transform(squeezed_features), device=self._device)
