"""Métricas de regressão implementadas à mão, como pede o enunciado."""
from __future__ import annotations

import numpy as np


def mse(y: np.ndarray, y_previsto: np.ndarray) -> float:
    """Erro quadrático médio."""
    y, y_previsto = np.asarray(y, float), np.asarray(y_previsto, float)
    return float(np.mean((y - y_previsto) ** 2))


def rmse(y: np.ndarray, y_previsto: np.ndarray) -> float:
    """Raiz do MSE — mesma unidade do alvo, mais fácil de interpretar."""
    return float(np.sqrt(mse(y, y_previsto)))


def r2(y: np.ndarray, y_previsto: np.ndarray) -> float:
    """Coeficiente de determinação.

        R² = 1 - SQ_res / SQ_tot

    SQ_tot usa a média do PRÓPRIO conjunto avaliado. R² negativo significa que
    o modelo é pior que prever a média — é resultado válido, não erro de conta.
    """
    y, y_previsto = np.asarray(y, float), np.asarray(y_previsto, float)
    sq_res = np.sum((y - y_previsto) ** 2)
    sq_tot = np.sum((y - y.mean()) ** 2)
    if sq_tot == 0:
        raise ValueError("alvo constante: R² não é definido")
    return float(1 - sq_res / sq_tot)


def avaliar(y: np.ndarray, y_previsto: np.ndarray) -> dict[str, float]:
    return {"mse": mse(y, y_previsto), "rmse": rmse(y, y_previsto), "r2": r2(y, y_previsto)}


def desfazer_log(y_log: np.ndarray) -> np.ndarray:
    """Volta previsões da escala log para a escala original.

    Cuidado ao comparar modelos: R² calculado em log NÃO é comparável com R²
    em dólares. Traga as previsões de volta com esta função antes de medir.
    A transformação inversa é enviesada (prevê a mediana, não a média) — vale
    uma linha no relatório.
    """
    return np.exp(np.asarray(y_log, float))
