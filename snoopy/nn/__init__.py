from .base import Arm, KNNAlgorithm, KNNAlgorithmType, KNNProgression, ProgressResult
from .faiss_knn import FaissKNN
from .top_k import TopK


def knn_algorithm_factory(nn_algorithm_type: KNNAlgorithmType) -> KNNAlgorithm:
    if nn_algorithm_type == KNNAlgorithmType.TOP_K:
        return TopK()
    else:
        return FaissKNN()
