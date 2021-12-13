# Adapted from https://github.com/facebookresearch/faiss/blob/master/gpu/test/test_pytorch_faiss.py
import faiss
import torch as pt

from .base import KNNAlgorithm
from ..custom_types import EmbeddingSlice, PartialResult


class FaissKNN(KNNAlgorithm):
    def __init__(self):
        self._res = faiss.StandardGpuResources()
        self._res.setDefaultNullStreamAllDevices()
        self._res.setTempMemory(64 * 1024 * 1024)

    @staticmethod
    def swig_ptr_from_float_tensor(x):
        assert x.is_contiguous()
        return faiss.cast_integer_to_float_ptr(x.storage().data_ptr() + x.storage_offset() * 4)

    @staticmethod
    def swig_ptr_from_long_tensor(x):
        assert x.is_contiguous()
        return faiss.cast_integer_to_long_ptr(x.storage().data_ptr() + x.storage_offset() * 8)

    def knn(self, train_dataset: EmbeddingSlice, test_dataset: EmbeddingSlice, k: int) -> PartialResult:
        train_features = train_dataset.features
        test_features = test_dataset.features

        num_test, dim_test = test_features.size()
        if test_features.is_contiguous():
            test_row_major = True
        elif test_features.t().is_contiguous():
            test_features = test_features.t()
            test_row_major = False
        else:
            raise TypeError('matrix should be row or column-major')
        x_test_ptr = self.swig_ptr_from_float_tensor(test_features)

        num_train, dim_train = train_features.size()
        assert dim_train == dim_test
        if train_features.is_contiguous():
            train_row_major = True
        elif train_features.t().is_contiguous():
            train_features = train_features.t()
            train_row_major = False
        else:
            raise TypeError('matrix should be row or column-major')
        x_train_ptr = self.swig_ptr_from_float_tensor(train_features)

        distances = pt.empty(num_test, k, device=train_features.device, dtype=pt.float32)
        indices = pt.empty(num_test, k, device=train_features.device, dtype=pt.int64)

        distances_ptr = self.swig_ptr_from_float_tensor(distances)
        indices_ptr = self.swig_ptr_from_long_tensor(indices)

        faiss.bruteForceKnn(self._res, faiss.METRIC_L2, x_train_ptr, train_row_major, num_train, x_test_ptr,
                            test_row_major, num_test, dim_test, k, distances_ptr, indices_ptr)

        relevant_labels = train_dataset.labels.view(-1)[indices]

        return PartialResult(distances, relevant_labels)
