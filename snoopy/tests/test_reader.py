import unittest

from tensorflow_datasets import Split

from snoopy.reader import TFDSImageConfig, data_factory


class TestTFDS(unittest.TestCase):
    def test_tfds_image_correct_size(self):
        cifar_train = data_factory(TFDSImageConfig("cifar10", Split.TRAIN))
        self.assertEqual(cifar_train.size, 50_000)

        cifar_test = data_factory(TFDSImageConfig("cifar10", Split.TEST))
        self.assertEqual(cifar_test.size, 10_000)

        fashion_mnist_train = data_factory(TFDSImageConfig("fashion_mnist", Split.TRAIN))
        self.assertEqual(fashion_mnist_train.size, 60_000)

        fashion_mnist_test = data_factory(TFDSImageConfig("fashion_mnist", Split.TEST))
        self.assertEqual(fashion_mnist_test.size, 10_000)


if __name__ == "__main__":
    unittest.main()
