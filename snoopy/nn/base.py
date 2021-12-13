from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import torch as pt

from ..custom_types import EmbeddingSlice, Expandable2D, PartialResult
from ..embedding import EmbeddingIterator


def get_top_k(values: pt.Tensor, k: int) -> Tuple[pt.Tensor, pt.Tensor]:
    num_test, num_train = values.shape

    if num_train <= k:
        return values, pt.arange(0, num_train, dtype=pt.long, device=values.device).repeat((num_test, 1))

    else:
        values, indices = pt.topk(values, k=k, sorted=False, largest=False)
        return values, indices


class KNNAlgorithmType(Enum):
    FAISS = "Faiss"
    TOP_K = "Top K"


class KNNAlgorithm(ABC):
    @abstractmethod
    def knn(self, train_dataset: EmbeddingSlice, test_dataset: EmbeddingSlice, k: int) -> PartialResult:
        pass


@dataclass
class ProgressResult:
    num_train_points_processed: int
    num_errors: int

    @classmethod
    def empty(cls):
        return cls(-1, -1)

    def __str__(self):
        return f"PR(n={self.num_train_points_processed}, err={self.num_errors})"

    def __repr__(self):
        return self.__str__()


# Used for simpler unit testing
class Arm(ABC):
    @property
    @abstractmethod
    def initial_error(self) -> ProgressResult:
        pass

    @abstractmethod
    def can_progress(self) -> bool:
        pass

    @abstractmethod
    def progress(self) -> ProgressResult:
        pass


class KNNProgression(Arm):
    def __init__(self, train_iter: EmbeddingIterator, test_iter: EmbeddingIterator, k: int,
                 knn_algorithm: KNNAlgorithm):
        assert train_iter.device == test_iter.device, "Training and test data must be on same device! Train is on: " \
                                                      f"{train_iter.device}, test is on: {test_iter.device}"
        self._train_iter = train_iter
        self._test_iter = test_iter
        self._k = k
        self._knn_algorithm = knn_algorithm
        self._device = train_iter.device

        # Creation of temporary result storage
        self._num_test = self._test_iter.size
        self._best_values = pt.zeros((self._num_test, 0), device=self._device)
        self._best_labels = pt.zeros((self._num_test, 0), dtype=pt.int64, device=self._device)
        self._test_labels = Expandable2D(self._num_test, 1, dtype=pt.int64, device=self._device)
        self._num_train_points_processed = 0

    @property
    def initial_error(self) -> ProgressResult:
        return ProgressResult(0, self._test_iter.size)

    def can_progress(self) -> bool:
        return self._train_iter.has_next()

    def progress(self) -> ProgressResult:
        assert self.can_progress(), "Progress cannot be made!"

        train_batch = self._train_iter.next()
        train_batch_size = train_batch.features.shape[0]
        self._num_train_points_processed += train_batch_size

        # Prepare tensors that will store distances and labels for nearest neighbors from current train batch
        # Handle case, when there are less training points in a batch than k
        actual_k = min(train_batch_size, self._k)
        current_train_batch_values = Expandable2D(self._num_test, actual_k, dtype=pt.float32, device=self._device)
        current_train_batch_labels = Expandable2D(self._num_test, actual_k, dtype=pt.int64, device=self._device)

        # Iterate through whole test set and determine nearest neighbors from current train batch
        self._test_iter.reset()
        while self._test_iter.has_next():
            test_batch = self._test_iter.next()

            # Collect all test labels only when progress is called for the first time
            if self._test_labels.size != self._num_test:
                self._test_labels.add(test_batch.labels)

            # Add data from current test batch
            result = self._knn_algorithm.knn(train_batch, test_batch, self._k)
            current_train_batch_values.add(result.values)
            current_train_batch_labels.add(result.labels)

        # Merge nearest neighbors from all previous training batches and current training batch
        self._best_values, self._best_labels = self._reduce_nearest_neighbors(self._best_values,
                                                                              current_train_batch_values.data,
                                                                              self._best_labels,
                                                                              current_train_batch_labels.data)

        # Count how many wrong predictions occur
        current_error = self._calculate_error(self._best_labels, self._test_labels.data)
        return ProgressResult(self._num_train_points_processed, current_error)

    def _reduce_nearest_neighbors(self, old_values: pt.Tensor, new_values: pt.Tensor, old_labels: pt.Tensor,
                                  new_labels: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        best_distances, best_indices = get_top_k(pt.cat([old_values, new_values], dim=1), self._k)
        best_labels = pt.gather(pt.cat([old_labels, new_labels], dim=1), 1, best_indices)

        return best_distances, best_labels

    def _calculate_error(self, train_labels: pt.Tensor, test_labels: pt.Tensor) -> int:
        if self._k == 1:
            return pt.sum(train_labels != test_labels).item()

        # Check whether most commonly occurring label among k nearest neighbors is same as test label
        num_errors = 0

        test_point_index = 0
        # TODO: more efficient implementation (argmax and [] used only once if possible)
        for nn_labels in train_labels:
            nn_labels_unique, nn_labels_counts = pt.unique(nn_labels, return_counts=True)
            most_common_label_index = pt.argmax(nn_labels_counts)
            predicted_label = nn_labels_unique[most_common_label_index]

            if predicted_label != test_labels[test_point_index]:
                num_errors += 1

            test_point_index += 1

        return num_errors
