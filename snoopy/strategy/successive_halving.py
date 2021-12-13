from collections import OrderedDict as OrdD, defaultdict as dd
from dataclasses import dataclass
from math import ceil, floor, log
from typing import List, OrderedDict

import numpy as np

from .base import Strategy, StrategyConfig, _get_arm, _notify_initial_error
from .._logging import get_logger
from ..embedding import EmbeddingDatasetsTuple
from ..nn import Arm, KNNAlgorithmType, knn_algorithm_factory
from ..result import Observer

_logger = get_logger(__name__)


@dataclass(frozen=True)
class SuccessiveHalvingStrategyConfig(StrategyConfig):
    train_size: int
    test_size: int
    reduce_factor: int


class SuccessiveHalvingAlgorithm:
    def __init__(self, arms: OrderedDict[str, Arm], observer: Observer):
        _notify_initial_error(arms, observer)

        self._arms = arms
        self._observer = observer

        # Number of arms
        self._num_arms = len(arms)

        # Number of times an armed has been pulled for each arm
        self._pulls_performed = {arm_name: 0 for arm_name in self._arms}

        # Each arms has a mapping: num pulls -> error, for fast access
        # -1 is returned when certain number of pulls cannot be reached by that arm
        self._partial_results = {arm_name: dd(lambda: -1) for arm_name in self._arms}

    def _successive_halving(self, budget: int, eta: int, arm_names: List[str] = None) -> None:
        _logger.debug("(Inner) Running successive halving")
        # Names of remaining arms
        if arm_names is None:
            arm_names = list(self._arms.keys())

        # Number of arms considered currently
        num_arms_left = len(arm_names)

        # Number of arms remaining / competing
        abs_big_s_k = len(arm_names)  # = n

        # Budget
        big_b = budget

        # Cumulative sum of arm pulls over iterations
        big_r_k = 0

        # One iteration of successive halving
        # Last iteration will have k = ceil(log_eta(n))
        log_value = max(1, ceil(log(num_arms_left, eta)) + 1)
        for k in range(log_value):
            # Number of pulls in current iteration
            r_k = floor(big_b / (abs_big_s_k * log_value))
            if r_k == 0:
                break

            _logger.debug(f"(Inner) Iteration: {k}, target num. pulls: {big_r_k + r_k}, "
                          f"{', '.join(arm_names)} are left")

            # Iterate through arms that are still competing
            for arm_name in arm_names:
                current_arm = self._arms[arm_name]

                # Try to pull it r_k times
                for pull_index in range(r_k):
                    # Cumulative number of pulls required by current iteration sub step
                    target_num_pulls = big_r_k + pull_index + 1

                    # Only try to pull if arm is 'behind' the number of pulls required
                    if self._pulls_performed[arm_name] < target_num_pulls:
                        # Arm can be pulled -> pull it
                        if current_arm.can_progress():
                            # Pull arm
                            current_progress = current_arm.progress()
                            _logger.debug(f"(Inner) Pulled arm: {arm_name}")

                            # Notify observer of progress
                            self._observer.on_update(arm_name, current_progress)

                            # Store partial result for potential reuse
                            self._pulls_performed[arm_name] = target_num_pulls
                            self._partial_results[arm_name][target_num_pulls] = current_progress.num_errors

                        # Arm cannot be pulled
                        else:
                            break

            # Update cumulative sum of arm pulls
            big_r_k += r_k

            # Determine indices of arms that are still competing after current iteration
            # Those are the arms that are among (1 / eta) * remaining arms with smallest losses

            # Example of difference between indices of losses and indices of arms:
            # 1. We have an array of arms: [arm0, arm1, ..., arm7]
            # 2. Let indices of arms left be: [4, 6, 2, 1] -> they represent arm4, arm6, arm2 and arm1
            # 3. Let the corresponding losses after certain number of pulls be [200, 500, 100, 600]
            # # Thus: arm4: 200, arm6: 500, arm2: 100, arm1: 600
            # 4. Indices of losses sorted by loss are: [2, 0, 1, 3]
            # 5. Indices of arms sorted according to indices from 4. step are: [2, 4, 6, 1]

            losses = [self._partial_results[i][big_r_k] for i in arm_names]

            # TODO: possible replacement: simple ignore those arms
            # End if one of the arms could not execute all pulls
            if -1 in losses:
                break

            indices_of_losses_sorted_according_to_loss = np.argsort(losses)
            indices_of_losses_to_retain = indices_of_losses_sorted_according_to_loss[:(abs_big_s_k // eta)]
            arm_names = [arm_names[i] for i in indices_of_losses_to_retain]
            abs_big_s_k = len(arm_names)

            # Do not execute all iterations in case that next iteration would not return retain any arm
            # This is needed when number of arms is not a power of eta
            # TODO: possible replacement: abs_big_s_k < eta
            if abs_big_s_k == 0:
                break

    def successive_halving_with_doubling(self, initial_budget: int, eta: int):
        # Arms that have not reached the end yet
        arm_names_for_next_run = list(self._arms.keys())

        # Budget to start each iteration with
        current_budget = initial_budget

        # While there exist arms that have not reached the end yet
        while len(arm_names_for_next_run) > 0:
            # Run successive halving
            _logger.debug(f"(Outer) Running with budget: {current_budget}")
            _logger.debug(f"(Outer) Remaining arms: {', '.join(arm_names_for_next_run)}")
            self._successive_halving(current_budget, eta, arm_names=arm_names_for_next_run)

            # 'Doubling trick'
            current_budget *= 2

            # If one of the arms reached the end in the last run, do not use it anymore and reset budget
            arm_names_to_remove = []
            for arm_name in arm_names_for_next_run:
                if not self._arms[arm_name].can_progress():
                    _logger.debug(f"(Outer) Removing arm: {arm_name}")
                    arm_names_to_remove.append(arm_name)

            if len(arm_names_to_remove) > 0:
                arm_names_for_next_run = [i for i in arm_names_for_next_run if i not in arm_names_to_remove]
                current_budget = initial_budget


class SuccessiveHalvingStrategy(Strategy):

    def __init__(self, config: SuccessiveHalvingStrategyConfig):
        self._train_size = config.train_size
        self._test_size = config.test_size
        self._knn = knn_algorithm_factory(KNNAlgorithmType.TOP_K)
        self._reduce_factor = config.reduce_factor

    def execute(self, datasets: OrderedDict[str, EmbeddingDatasetsTuple], observer: Observer) -> None:
        # Prepare 'arms' that can be 'pulled'
        arms = OrdD()
        for dataset_name in datasets:
            _logger.debug(f"Preparing test dataset for embedding '{dataset_name}'")
            arms[dataset_name] = _get_arm(datasets[dataset_name], self._train_size, self._test_size, self._knn)
            _logger.debug(f"Test dataset for embedding '{dataset_name}' is now prepared")

        sha = SuccessiveHalvingAlgorithm(arms, observer)
        sha.successive_halving_with_doubling(initial_budget=1, eta=self._reduce_factor)
