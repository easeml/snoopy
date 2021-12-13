import torch as pt

from .base import KNNAlgorithm, get_top_k
from ..custom_types import EmbeddingSlice, PartialResult


class TopK(KNNAlgorithm):

    def knn(self, train_dataset: EmbeddingSlice, test_dataset: EmbeddingSlice, k: int) -> PartialResult:
        # TODO: remove x_test_2 part, since it does not change the result
        dist = -2 * pt.matmul(test_dataset.features, train_dataset.features.t())
        x_train_2 = pt.reshape(pt.sum(pt.square(train_dataset.features), dim=1), (1, -1))
        x_test_2 = pt.reshape(pt.sum(pt.square(test_dataset.features), dim=1), (-1, 1))
        dist = pt.add(dist, x_train_2)
        dist = pt.add(dist, x_test_2)

        highest_values, relevant_indices = get_top_k(dist, k)
        relevant_labels = train_dataset.labels.view(-1)[relevant_indices]

        return PartialResult(highest_values, relevant_labels)
