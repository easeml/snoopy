from dataclasses import dataclass
from typing import OrderedDict

from .base import Strategy, StrategyConfig, _get_arm
from .._logging import get_logger
from ..embedding import EmbeddingDatasetsTuple
from ..nn import KNNAlgorithmType, knn_algorithm_factory
from ..result import Observer

_logger = get_logger(__name__)


@dataclass(frozen=True)
class SimpleStrategyConfig(StrategyConfig):
    train_size: int
    test_size: int


class SimpleStrategy(Strategy):

    def __init__(self, config: SimpleStrategyConfig):
        self._train_size = config.train_size
        self._test_size = config.test_size
        self._knn = knn_algorithm_factory(KNNAlgorithmType.TOP_K)

    def execute(self, datasets: OrderedDict[str, EmbeddingDatasetsTuple], observer: Observer) -> None:
        for key in datasets:
            _logger.debug(f"Preparing test dataset for embedding '{key}'")
            arm = _get_arm(datasets[key], self._train_size, self._test_size, self._knn)
            _logger.debug(f"Test dataset for embedding '{key}' is now prepared")
            observer.on_update(key, arm.initial_error)

            while arm.can_progress():
                observer.on_update(key, arm.progress())
