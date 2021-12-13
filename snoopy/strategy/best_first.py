from typing import OrderedDict

from snoopy.nn import Arm
from snoopy.result import Observer
from .base import _notify_initial_error
from .._logging import get_logger

_logger = get_logger(__name__)


class BestFirstAlgorithm:
    def __init__(self, arms: OrderedDict[str, Arm], observer: Observer):
        _notify_initial_error(arms, observer)

        self._arms = arms
        self._observer = observer

        # TODO: run_min

    def best_first(self):
        num_pulls_per_arm = {arm_name: 0 for arm_name in self._arms}
        slope_per_arm = {arm_name: 0 for arm_name in self._arms}
        num_errors_per_arm = {arm_name: self._arms[arm_name].initial_error.num_errors for arm_name in self._arms}

        # Run one around to assess initial slope
        for arm_name in self._arms:
            arm = self._arms[arm_name]
            assert arm.can_progress(), "Arm should be able to be pulled at least once!"

            # Pull arm once and compute slope
            result = arm.progress()
            num_pulls_per_arm[arm_name] += 1
            slope_per_arm[arm_name] = num_errors_per_arm[arm_name] - result.num_errors
            num_errors_per_arm[arm_name] = result.num_errors

            # Update observer immediately after the pull
            self._observer.on_update(arm_name, result)

        names_of_arms_left = set(self._arms.keys())
        while len(names_of_arms_left) > 0:
            for arm_name_candidate in sorted(self._arms, key=lambda x: num_errors_per_arm[x]):
                if arm_name_candidate not in names_of_arms_left:
                    continue

                # If selected arm cannot be pulled anymore, remove it
                if not self._arms[arm_name_candidate].can_progress():
                    names_of_arms_left.remove(arm_name_candidate)
                    break

                _logger.debug(f"Trying: {arm_name_candidate}")

                is_candidate_suitable = True
                candidate_error_count = num_errors_per_arm[arm_name_candidate]

                for arm_name_competitor in names_of_arms_left:
                    # Only compare arms 'behind'
                    if arm_name_competitor == arm_name_candidate or \
                            num_pulls_per_arm[arm_name_candidate] <= num_pulls_per_arm[arm_name_competitor]:
                        continue

                    # Check if tangent of competitor is below candidate
                    current_errors = num_errors_per_arm[arm_name_competitor]
                    slope = slope_per_arm[arm_name_competitor]
                    pull_diff = num_pulls_per_arm[arm_name_candidate] - num_pulls_per_arm[arm_name_competitor]
                    competitor_projection = current_errors - slope * pull_diff

                    _logger.debug(f"\tComparing with: {arm_name_competitor}, Can: {candidate_error_count}, "
                                  f"Com: {competitor_projection}")

                    if competitor_projection < candidate_error_count:
                        is_candidate_suitable = False
                        break

                # No tangent is below candidate -> Pull arm
                if is_candidate_suitable:
                    _logger.debug(f"\tCandidate {arm_name_candidate} is suitable")
                    candidate_arm = self._arms[arm_name_candidate]
                    result = candidate_arm.progress()

                    # Update values
                    num_pulls_per_arm[arm_name_candidate] += 1
                    slope_per_arm[arm_name_candidate] = num_errors_per_arm[arm_name_candidate] - result.num_errors
                    num_errors_per_arm[arm_name_candidate] = result.num_errors

                    # Notify observer
                    self._observer.on_update(arm_name_candidate, result)

                    break
