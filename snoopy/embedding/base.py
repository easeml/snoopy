from __future__ import annotations

from abc import ABC, abstractmethod
from time import time
from typing import Callable, NamedTuple, Optional, Tuple

import numpy as np
import tensorflow as tf
import torch as pt

from .._logging import get_logger
from .._utils import get_num_splits, get_tf_device
from ..custom_types import CacheType, DataType, DataWithInfo, EmbeddingSlice, Expandable2D
from ..reader import UNKNOWN_LABEL

_logger = get_logger(__name__)

CPU_CACHE_NUM_BATCHES_IN_GPU = 20


class _ImageTransformationHelper:
    @staticmethod
    def _central_crop_with_resize(feature: tf.Tensor, required_image_size: Tuple[int, int]) -> tf.Tensor:
        converted_img = tf.image.convert_image_dtype(feature, dtype=tf.float32, saturate=False)
        shape = tf.shape(converted_img)
        min_dim = tf.minimum(shape[0], shape[1])
        cropped_img = tf.image.resize_with_crop_or_pad(converted_img, min_dim, min_dim)
        return tf.image.resize(cropped_img, required_image_size)

    @staticmethod
    def central_crop_with_resize_3_channels(feature: tf.Tensor, required_image_size: Tuple[int, int]) -> tf.Tensor:
        resized_img = _ImageTransformationHelper._central_crop_with_resize(feature, required_image_size)
        # For 1 channel, repeats 3 times; for 3 channels, repeats 1 time
        return tf.repeat(resized_img, 3 - tf.shape(resized_img)[2] + 1, axis=2)

    @staticmethod
    def raw_image_with_central_crop_and_resize(feature: tf.Tensor, required_image_size: Tuple[int, int]) -> tf.Tensor:
        resized_img = _ImageTransformationHelper._central_crop_with_resize(feature, required_image_size)
        return tf.reshape(resized_img, (1, -1))


class EmbeddingModel(ABC):
    @abstractmethod
    def move_to(self, device: pt.device) -> None:
        pass

    @property
    @abstractmethod
    def output_dimension(self) -> int:
        pass

    @abstractmethod
    def get_data_preparation_function(self) -> Callable:
        pass

    @abstractmethod
    def apply_embedding(self, features: np.ndarray) -> pt.Tensor:
        pass


class EmbeddingModelSpec(ABC):

    @abstractmethod
    def load(self, device: pt.device) -> EmbeddingModel:
        pass

    @property
    @abstractmethod
    def data_type(self) -> DataType:
        pass


class EmbeddingConfig(NamedTuple):
    embedding_model_spec: EmbeddingModelSpec
    batch_size: int

    # NOTE: prefetch size significantly increases RAM usage
    prefetch_size: int
    label_noise_amount: Optional[float] = None


# TODO: A class that contain input_data and config + a method that accepts cache_type and device and returns
#  EmbeddingDataset
class EmbeddingDataset:
    def __init__(self, input_data: DataWithInfo, config: EmbeddingConfig, cache_type: CacheType, device: pt.device):
        # Check that parameters in config are valid
        assert config.batch_size > 0, f"Batch size of specified {type(config)} must be a positive number!"
        assert config.prefetch_size > 0, f"Prefetch size of specified {type(config)} must be a positive number!"
        assert config.label_noise_amount is None or 0.0 < config.label_noise_amount <= 1.0, \
            f"Label noise of specified {type(config)} must be in interval (0, 1]! For no noise, set it to None."

        # Initialize model
        self._embedding_model = config.embedding_model_spec.load(device)
        feature_fn = self._embedding_model.get_data_preparation_function()

        def label_alert_fn(feature: tf.Tensor, label: tf.Tensor):
            if label == UNKNOWN_LABEL:
                tf.py_function(
                    func=lambda: _logger.error(f"Label {UNKNOWN_LABEL} (unknown label) detected!"),
                    inp=[],
                    Tout=[]
                )

            return feature, label

        def label_noise(label: tf.Tensor, num_labels: int, noise_level: float) -> tf.Tensor:
            if tf.random.uniform([1])[0] < noise_level:
                # First two parameters are irrelevant in this case!
                return tf.random.uniform_candidate_sampler(
                    true_classes=[[0]], num_true=1, num_sampled=1, unique=True, range_max=num_labels
                ).sampled_candidates[0]
            else:
                return label

        def preparation_fn_no_label_noise(feature: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            return feature_fn(feature), label

        def preparation_fn_with_label_noise(feature: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            return feature_fn(feature), label_noise(label, input_data.num_labels, config.label_noise_amount)

        if config.label_noise_amount:
            preparation_fn = preparation_fn_with_label_noise
        else:
            preparation_fn = preparation_fn_no_label_noise

        # Prepare input data
        self._input_dataset = input_data.data \
            .map(label_alert_fn) \
            .map(preparation_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .batch(config.batch_size) \
            .prefetch(config.prefetch_size) \
            .as_numpy_iterator()

        self._output_dimension = self._embedding_model.output_dimension
        self._apply_embedding = self._embedding_model.apply_embedding
        self._size = input_data.size
        self._batch_size = config.batch_size
        self._device = device

        # Cache data
        self._feature_cache = None
        self._label_cache = None

        # Set when 'prepare' is called
        self._cache_type = cache_type
        self._prepared = False

    @property
    def device(self) -> pt.device:
        return self._device

    @property
    def size(self) -> int:
        return self._size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def cache_type(self) -> CacheType:
        return self._cache_type

    def _compute_embedding(self, num_batches_to_return: Optional[int]) -> Optional[EmbeddingSlice]:
        if num_batches_to_return:
            result_size = self._batch_size * num_batches_to_return
            num_iters = num_batches_to_return
        else:
            result_size = self._size
            num_iters = get_num_splits(self._size, self._batch_size)

        embeddings_to_return = Expandable2D(result_size, self._output_dimension, dtype=pt.float32, device=self._device)
        labels_to_return = Expandable2D(result_size, 1, dtype=pt.int64, device=self._device)

        feature_retrieve_times = []
        for i in range(num_iters):
            try:
                start = time()
                current_features_np, current_labels_np = next(self._input_dataset)
                feature_retrieve_times.append(time() - start)
            except StopIteration:
                break

            # Copy tf.Tensor (CPU) to pt.Tensor(GPU)
            current_labels = pt.as_tensor(current_labels_np, dtype=pt.int64, device=self._device).view((-1, 1))

            # Apply embedding on mini-batch
            current_embeddings = self._apply_embedding(current_features_np)

            # Store embedding and labels
            embeddings_to_return.add(current_embeddings)
            labels_to_return.add(current_labels)

            if i % 100 == 0 and i > 0:
                _logger.debug(f"Batches processed: {i}")

        _logger.debug(f"Average feature retrieve time was {np.mean(np.array(feature_retrieve_times)): .3f} seconds")

        # Return relevant data (current batch could be smaller than batch size)
        if embeddings_to_return.size > 0:
            return EmbeddingSlice(embeddings_to_return.data, labels_to_return.data)

        else:
            return None

    def _prepare_device_cache(self) -> None:
        self._feature_cache, self._label_cache = self._compute_embedding(None)

    def _prepare_cpu_cache(self) -> None:
        global CPU_CACHE_NUM_BATCHES_IN_GPU
        features = Expandable2D(self._size, self._output_dimension, dtype=pt.float32)
        labels = Expandable2D(self._size, 1, dtype=pt.int64)

        num_splits = get_num_splits(self._size, self._batch_size * CPU_CACHE_NUM_BATCHES_IN_GPU)
        for split_index in range(num_splits):
            _logger.info(f"Processing split: {split_index + 1}/{num_splits} on {get_tf_device(self._device)}")
            result = self._compute_embedding(CPU_CACHE_NUM_BATCHES_IN_GPU)
            features.add(result.features.cpu())
            labels.add(result.labels.cpu())

        self._feature_cache = features.data
        self._label_cache = labels.data

    def prepare(self):
        if not self._prepared:
            if self._cache_type == CacheType.DEVICE:
                self._prepare_device_cache()

            elif self._cache_type == CacheType.CPU:
                self._prepare_cpu_cache()

        self._prepared = True

    def get_cache(self, start_index: int, end_index: int, copy_to_device: bool = False) -> Optional[EmbeddingSlice]:
        assert self._prepared, "EmbeddingDataset was not prepared when use_cache was called!"
        assert self._cache_type != CacheType.NONE, "This method can only be used if cache is precomputed!"
        assert 0 <= start_index <= self._size and 0 <= end_index <= self.size and start_index <= end_index, \
            "Provided indices are not valid!"

        if end_index - start_index == 0:
            return None

        features_to_return = pt.narrow(self._feature_cache, 0, start_index, end_index - start_index)
        labels_to_return = pt.narrow(self._label_cache, 0, start_index, end_index - start_index)

        # Copy to the device on which embedding was computed
        if copy_to_device and features_to_return.device != self._device:
            features_to_return = features_to_return.to(self._device)
            labels_to_return = labels_to_return.to(self._device)

        return EmbeddingSlice(features_to_return, labels_to_return)

    def get_next(self, num_batches_to_return: int) -> Optional[EmbeddingSlice]:
        assert self._prepared, "EmbeddingDataset was not prepared when get_next was called!"
        assert self._cache_type == CacheType.NONE, "This method can only be used if no cache is used!"
        return self._compute_embedding(num_batches_to_return)

    def get_iterator(self, batches_per_iter) -> EmbeddingIterator:
        return EmbeddingIterator(self, batches_per_iter)


class EmbeddingDatasetsTuple(NamedTuple):
    train: EmbeddingDataset
    test: EmbeddingDataset


class EmbeddingIterator:
    def __init__(self, embedding_dataset: EmbeddingDataset, num_batches_per_iter: int):
        self._embedding_dataset = embedding_dataset
        self._embedding_dataset.prepare()

        # Embedding dataset properties
        self._size = embedding_dataset.size
        self._batch_size = embedding_dataset.batch_size
        self._cache_type = embedding_dataset.cache_type

        # Tracking where in dataset are we
        self._num_iters_done = 0
        self._num_iters_available = get_num_splits(self._size, self._batch_size * num_batches_per_iter)
        self._num_batches_per_iter = num_batches_per_iter
        self._start_index = 0
        self._end_index = 0

    def reset(self) -> None:
        assert self._cache_type != CacheType.NONE, "Iterator cannot be reset if there is no cache!"

        self._start_index = 0
        self._end_index = 0
        self._num_iters_done = 0

    @property
    def device(self) -> pt.device:
        return self._embedding_dataset.device

    @property
    def size(self) -> int:
        return self._size

    def has_next(self) -> bool:
        return self._num_iters_done < self._num_iters_available

    def next(self) -> EmbeddingSlice:
        assert self.has_next(), "'.next()' was called on iterator that was iterated to the end!"

        self._num_iters_done += 1

        if self._cache_type == CacheType.NONE:
            return self._embedding_dataset.get_next(self._num_batches_per_iter)

        else:
            self._start_index = min(self._end_index, self._size)
            self._end_index = min(self._start_index + self._batch_size * self._num_batches_per_iter, self._size)

            return self._embedding_dataset.get_cache(self._start_index, self._end_index)
