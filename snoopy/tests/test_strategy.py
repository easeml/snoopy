import unittest
from collections import OrderedDict as OrdD

import numpy as np

from snoopy.nn import Arm, ProgressResult
from snoopy.result import Observer
from snoopy.strategy import SuccessiveHalvingAlgorithm


def gen_graphs():
    import matplotlib.pyplot as plt

    num_samples = 30
    x = np.array(list(range(num_samples))) + 1

    values_all = np.empty((8, num_samples), dtype=np.int)

    for graph in range(8):
        first = int(np.random.uniform(70000, 100000))
        second = int(np.random.uniform(60000, 70000))
        values = [first, second]

        s = np.random.uniform(0.5, 0.8)

        for iteration in range(num_samples - 2):
            values.append(int(np.random.uniform(
                values[-1] - (values[-2] - values[-1]),
                values[-1] - s * (values[-2] - values[-1]))
            ))

        values_all[graph, :] = np.array(values) + 500_000
        plt.plot(x, np.array(values) + 500_000)

    np.savez("test_data", x=x, data=values_all)
    plt.show()


class ArmSimulator(Arm):
    def __init__(self, x: np.ndarray, y: np.ndarray, initial_error: int, increment: int = 1):
        assert x.size == y.size, "x and y should have same number of elements!"
        self._x = x
        self._y = y
        self._index = min(increment - 1, self._x.size - 1)
        self._initial_error = initial_error
        self._increment = increment

    @property
    def initial_error(self) -> ProgressResult:
        return ProgressResult(0, self._initial_error)

    def can_progress(self) -> bool:
        return self._index < self._x.size

    def progress(self) -> ProgressResult:
        return_value = ProgressResult(self._x[self._index], self._y[self._index])

        # When combining multiple pulls (increment > 1), let last progress report error after all possible pulls
        # instead of skipping it entirely
        if self._index + self._increment >= self._x.size and self._index != self._x.size - 1:
            self._index = self._x.size - 1
        else:
            self._index += self._increment
        return return_value


class TestSuccessiveHalving(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        all_data = np.load("test_data.npz")
        x = all_data["x"]
        values = all_data["data"]
        cls.arms = OrdD()
        for i in range(values.shape[0]):
            cls.arms[str(i)] = ArmSimulator(x, values[i, :], initial_error=600_000)

    def test_successive_halving_with_doubling(self):
        class MyObserver(Observer):
            def on_update(self, name: str, progress_result: ProgressResult):
                print(name, progress_result)

        algorithm = SuccessiveHalvingAlgorithm(self.arms, MyObserver())
        algorithm.successive_halving_with_doubling(initial_budget=24, eta=2)
        # TODO: write a test


if __name__ == "__main__":
    unittest.main()
