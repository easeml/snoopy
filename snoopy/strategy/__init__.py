from .base import Strategy, StrategyConfig
from .best_first import BestFirstAlgorithm
from .simple import SimpleStrategy, SimpleStrategyConfig
from .successive_halving import SuccessiveHalvingAlgorithm, SuccessiveHalvingStrategy, SuccessiveHalvingStrategyConfig
from .uniform_allocation import UniformAllocationAlgorithm


def strategy_factory(config: StrategyConfig) -> Strategy:
    if isinstance(config, SuccessiveHalvingStrategyConfig):
        return SuccessiveHalvingStrategy(config)

    elif isinstance(config, SimpleStrategyConfig):
        return SimpleStrategy(config)

    else:
        raise NotImplementedError
