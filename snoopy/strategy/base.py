from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import OrderedDict

from ..embedding import EmbeddingDatasetsTuple
from ..nn import Arm, KNNAlgorithm, KNNProgression
from ..result import Observer


def _get_arm(dataset: EmbeddingDatasetsTuple, train_size: int, test_size: int, knn: KNNAlgorithm) -> Arm:
    train, test = dataset

    # Check that number of train points ('train_size') evaluated at one lever pull is divisible by the number of
    # train points in a batch
    # Same for test points ('test_size'), but here this is just a number of test points evaluated at a time in order not
    # to perform too much computation (of order 'train_size' x 'test_size') at once
    assert train_size % train.batch_size == 0 and train_size > 0, \
        "Number of points in training batch must be divisible by the training batch size!"
    assert test_size % test.batch_size == 0 and test_size > 0, \
        "Number of points in test batch must be divisible by the test batch size!"

    train_iterator = train.get_iterator(batches_per_iter=train_size // train.batch_size)
    test_iterator = test.get_iterator(batches_per_iter=test_size // test.batch_size)

    return KNNProgression(train_iterator, test_iterator, k=1, knn_algorithm=knn)


def _notify_initial_error(arms: OrderedDict[str, Arm], observer: Observer):
    # Notify initial error
    for arm_name in arms:
        observer.on_update(arm_name, arms[arm_name].initial_error)


@dataclass(frozen=True)
class StrategyConfig:
    pass


class Strategy(ABC):
    @abstractmethod
    def execute(self, datasets: OrderedDict[str, EmbeddingDatasetsTuple], observer: Observer) -> None:
        pass


class StrategyAlgorithm(ABC):
    pass
