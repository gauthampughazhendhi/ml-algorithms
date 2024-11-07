from typing import List

import numpy as np


def calculate_accuracy(y_pred: List[int], y: List[int]) -> float:
    y_pred, y = np.array(y_pred), np.array(y)
    return np.sum(y_pred == y) / y_pred.size

def calculate_rmse(y_pred: List[int], y: List[int]) -> float:
    return np.sqrt(np.mean((y - y_pred)**2))

def calculate_mse(y_pred: List[int], y: List[int]) -> float:
    return np.mean((y - y_pred)**2)

def unit_step_activation(y: np.array) -> int:
    return np.where(y > 0, 1, 0)
