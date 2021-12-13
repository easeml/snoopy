import unittest

import numpy as np
import tensorflow as tf
import torch as pt

from snoopy.custom_types import CacheType, DataWithInfo, EmbeddingSlice, PartialResult
from snoopy.embedding import DummySpec, EmbeddingConfig, EmbeddingDataset
from snoopy.nn import FaissKNN, KNNProgression, ProgressResult, TopK

train_points = tf.constant([[-2, 3], [-2, 0], [-2, -2], [-2, -4], [0, 0], [6, 2], [0, 4], [5, 0], [4, -3], [2, 3],
                            [5, 3], [3, 2], [1, 0]], dtype=tf.float32)
train_points_labels = tf.constant([1, 2, 1, 3, 3, 1, 2, 2, 3, 1, 3, 3, 2], dtype=tf.int64)
test_points = tf.constant([[-1, 5], [1, 1], [4, 3], [0, -3], [6, 0]], dtype=tf.float32)
test_points_labels = tf.constant([2, 2, 3, 2, 3], dtype=tf.int64)


class TestKNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tensor_train_dataset = EmbeddingSlice(pt.as_tensor(train_points.numpy()),
                                                  pt.as_tensor(train_points_labels.numpy()))
        cls.tensor_test_dataset = EmbeddingSlice(pt.as_tensor(test_points.numpy()),
                                                 pt.as_tensor(test_points_labels.numpy()))

    @staticmethod
    def check_correct_result(result: PartialResult):
        values = result.values.numpy()
        labels = result.labels.numpy()

        true_squared_distances_to_4_closest = np.array([[2, 5, 13, 25], [1, 2, 5, 5], [1, 2, 4, 5], [5, 5, 9, 10],
                                                        [1, 4, 10, 13]], dtype=np.float32)
        true_labels_of_4_closest = np.array([[2, 1, 1, 3], [2, 3, 1, 3], [3, 3, 1, 1], [1, 3, 3, 2], [2, 1, 3, 3]],
                                            dtype=np.int64)

        # Make sure that distances are computed correctly
        np.testing.assert_allclose(np.sort(values, axis=1), np.sort(true_squared_distances_to_4_closest, axis=1))

        # Make sure that labels of 4 nearest neighbors are correct (do not care about their order)
        np.testing.assert_equal(np.sort(labels, axis=1), np.sort(true_labels_of_4_closest, axis=1))

    def test_top_k(self):
        top_k = TopK()
        result = top_k.knn(self.tensor_train_dataset, self.tensor_test_dataset, k=4)
        self.check_correct_result(result)

    def test_faiss(self):
        f = FaissKNN()
        result = f.knn(self.tensor_train_dataset, self.tensor_test_dataset, k=4)
        self.check_correct_result(result)


class TestKNNProgression(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train_dataset = DataWithInfo(tf.data.Dataset.from_tensor_slices((train_points, train_points_labels)),
                                         train_points.shape[0])
        cls.test_dataset = DataWithInfo(tf.data.Dataset.from_tensor_slices((test_points, test_points_labels)),
                                        test_points.shape[0])
        cls.embedding_config = EmbeddingConfig(DummySpec(output_dimension=2), batch_size=1, prefetch_size=1)

    def get_knn_progression(self, batches_per_iter: int, k: int) -> KNNProgression:
        if pt.cuda.is_available():
            device = pt.device("cuda:0")
        else:
            device = pt.device("cpu")
        train_iterator = EmbeddingDataset(self.train_dataset, self.embedding_config, CacheType.NONE, device) \
            .get_iterator(batches_per_iter=batches_per_iter)
        test_iterator = EmbeddingDataset(self.test_dataset, self.embedding_config, CacheType.DEVICE, device) \
            .get_iterator(batches_per_iter=batches_per_iter)

        return KNNProgression(train_iterator, test_iterator, k=k, knn_algorithm=TopK())

    def test_train_batch_smaller_than_k(self):
        knn_progression = self.get_knn_progression(batches_per_iter=2, k=5)
        _ = knn_progression.progress()

    def test_whole_train_dataset_smaller_than_k(self):
        knn_progression = self.get_knn_progression(batches_per_iter=2, k=100)
        _ = knn_progression.progress()

    def test_k_1(self):
        knn_progression = self.get_knn_progression(batches_per_iter=5, k=1)

        for i in range(5):
            if i <= 2:
                result = knn_progression.progress()

                if i == 0:
                    self.assertEqual(result, ProgressResult(5, 3))

                elif i == 1:
                    self.assertEqual(result, ProgressResult(10, 4))

                elif i == 2:
                    self.assertEqual(result, ProgressResult(13, 2))

            else:
                with self.assertRaises(AssertionError):
                    knn_progression.progress()


if __name__ == "__main__":
    unittest.main()
