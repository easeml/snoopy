import unittest

import numpy as np
import tensorflow as tf
import torch as pt
from torchvision import transforms

from snoopy.custom_types import CacheType, DataWithInfo
from snoopy.embedding import DummySpec, EmbeddingConfig, EmbeddingDataset, EmbeddingIterator, vgg19


class TestEmbedding(unittest.TestCase):

    def setUp(self) -> None:
        # Dataset with length 1000 where each data point consists of tensor of length 10 and label '0'
        dataset = tf.data.Dataset.from_tensor_slices(tf.random.uniform((1000, 10), dtype=tf.float32)).map(
            lambda x: (x, tf.constant(0, dtype=tf.int64)))
        data = DataWithInfo(dataset, 1000)
        conf = EmbeddingConfig(DummySpec(output_dimension=10), batch_size=2, prefetch_size=1)
        if pt.cuda.is_available():
            device = pt.device("cuda:0")
        else:
            device = pt.device("cpu")
        self.get_embedding_dataset = lambda cache_type: EmbeddingDataset(data, conf, cache_type, device)

    def check_correct_output_size(self, iterator: EmbeddingIterator):
        for result_counter in range(20):
            if result_counter <= 16:
                result = iterator.next()

                if result_counter <= 15:
                    self.assertEqual(result.features.shape[0], 60)
                    self.assertEqual(result.labels.shape[0], 60)
                elif result_counter == 16:
                    self.assertEqual(result.features.shape[0], 40)
                    self.assertEqual(result.labels.shape[0], 40)
            else:
                with self.assertRaises(AssertionError):
                    iterator.next()

    def test_no_cache(self):
        iterator = self.get_embedding_dataset(CacheType.NONE).get_iterator(batches_per_iter=30)
        self.check_correct_output_size(iterator)

    def test_gpu_cache(self):
        iterator = self.get_embedding_dataset(CacheType.DEVICE).get_iterator(batches_per_iter=30)
        self.check_correct_output_size(iterator)
        iterator.reset()
        self.check_correct_output_size(iterator)

    def test_cpu_cache(self):
        iterator = self.get_embedding_dataset(CacheType.CPU).get_iterator(batches_per_iter=30)
        self.check_correct_output_size(iterator)
        iterator.reset()
        self.check_correct_output_size(iterator)

    def test_always_same_output(self):
        # No cache
        iterator_no_cache = self.get_embedding_dataset(CacheType.NONE).get_iterator(batches_per_iter=7)

        # GPU cache
        iterator_gpu_cache = self.get_embedding_dataset(CacheType.DEVICE).get_iterator(batches_per_iter=7)

        # CPU cache
        iterator_cpu_cache = self.get_embedding_dataset(CacheType.CPU).get_iterator(batches_per_iter=7)

        for i in range(100):
            if i <= 71:
                data_no_cache = iterator_no_cache.next()
                data_gpu_cache = iterator_gpu_cache.next()
                data_cpu_cache = iterator_cpu_cache.next()

                features_no_cache, labels_no_cache = data_no_cache
                features_gpu_cache, labels_gpu_cache = data_gpu_cache
                features_cpu_cache, labels_cpu_cache = data_cpu_cache

                # Test features
                np.testing.assert_allclose(features_no_cache.cpu(), features_gpu_cache.cpu())
                np.testing.assert_allclose(features_gpu_cache.cpu(), features_cpu_cache.cpu())

                # Test labels
                np.testing.assert_allclose(labels_no_cache.cpu(), labels_gpu_cache.cpu())
                np.testing.assert_allclose(labels_gpu_cache.cpu(), labels_cpu_cache.cpu())

            else:
                with self.assertRaises(AssertionError):
                    iterator_no_cache.next()

                with self.assertRaises(AssertionError):
                    iterator_gpu_cache.next()

                with self.assertRaises(AssertionError):
                    iterator_cpu_cache.next()

    def test_iterator_has_next(self):
        iterator_no_cache = self.get_embedding_dataset(CacheType.NONE).get_iterator(batches_per_iter=7)

        cnt = 0
        while iterator_no_cache.has_next():
            cnt += 1
            _ = iterator_no_cache.next()

        self.assertEqual(cnt, 72)


class TestTorchHubEmbedding(unittest.TestCase):

    def test_data_normalization(self):
        # Config is mocked, and self._required_image_size is set to None. prepare_data method is called on
        # TorchHubImageConfig class rather than mocked instance in order to be able to run the method. Mocked class
        # is needed, because prepare_data is an instance method, so the mocked object is passed as 'self'.
        if pt.cuda.is_available():
            device = pt.device("cuda:0")
        else:
            device = pt.device("cpu")
        model = vgg19.load(device)

        # Generate random image of size 224 x 224
        img = pt.randint(low=0, high=256, size=(224, 224, 3), dtype=pt.float32).numpy()

        # Process image with PyTorch, swap axis so that the result can be compared with TensorFlow
        preprocess = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        pt_processed_img = preprocess(img).permute(1, 2, 0).numpy()

        # Swap axis to conform to TensorFlow specification
        tf_img = tf.convert_to_tensor(img, dtype=tf.float32)
        fn = model.get_data_preparation_function()
        tf_processed_img = fn(tf_img).numpy()

        np.testing.assert_allclose(tf_processed_img, pt_processed_img)


if __name__ == "__main__":
    unittest.main()
