from typing import List

import numpy as np


def calculate_accuracy(y_pred: List[int], y: List[int]) -> float:
    y_pred, y = np.array(y_pred), np.array(y)
    return np.sum(y_pred == y) / y_pred.size
