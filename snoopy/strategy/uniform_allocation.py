from typing import OrderedDict

from snoopy.nn import Arm
from snoopy.result import Observer
from .base import _notify_initial_error


class UniformAllocationAlgorithm:
    def __init__(self, arms: OrderedDict[str, Arm], observer: Observer):
        _notify_initial_error(arms, observer)

        self._arms = arms
        self._observer = observer

    def uniform_allocation(self):
        names_of_arms_left = list(self._arms.keys())

        # Until there are arms to pull
        while len(names_of_arms_left) > 0:
            names_of_arms_to_remove = []

            # For each arm
            # TODO: use arms in order of lowest error from previous iteration
            for arm_name in names_of_arms_left:
                current_arm = self._arms[arm_name]

                # If arm can be pulled, notify observer of new result
                if current_arm.can_progress():
                    result = current_arm.progress()
                    self._observer.on_update(arm_name, result)

                # Otherwise, remove it
                else:
                    names_of_arms_to_remove.append(arm_name)

            names_of_arms_left = [i for i in names_of_arms_left if i not in names_of_arms_to_remove]
