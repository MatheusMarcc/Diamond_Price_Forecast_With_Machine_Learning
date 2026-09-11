"""Regressão Linear: equações normais e Gradient Descent.

O enunciado pede que estes dois algoritmos sejam implementados por você, com
módulos de treino e de predição separados. Por isso o encanamento está pronto
(intercepto, validações, histórico, comparação entre os dois) e o miolo de cada
um está marcado com TODO — são as poucas linhas que valem a nota.

Notação usada em todo o arquivo:
    X  matriz n x d de atributos JÁ PADRONIZADOS (sem coluna de uns)
    A  matriz n x (d+1), o X com uma coluna de uns na frente
    w  vetor (d+1), sendo w[0] o intercepto
"""
from __future__ import annotations

import numpy as np


def _com_intercepto(X: np.ndarray) -> np.ndarray:
    """Acrescenta a coluna de uns na frente de X."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X deve ser 2-D, recebi shape {X.shape}")
    return np.column_stack([np.ones(len(X)), X])


def _conferir(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if len(A) != len(y):
        raise ValueError(f"X tem {len(A)} linhas e y tem {len(y)}")
    if not np.isfinite(A).all() or not np.isfinite(y).all():
        raise ValueError("ha NaN ou infinito nos dados de entrada")
    return y


class RegressaoLinearFechada:
    """Solução exata pelas equações normais.

        AᵀA w = Aᵀy      →      w = (AᵀA)⁻¹ Aᵀy

    Serve de gabarito para validar o Gradient Descent: os dois têm que chegar
    no mesmo vetor de pesos.
    """

    def __init__(self) -> None:
        self.w_: np.ndarray | None = None

    def treinar(self, X: np.ndarray, y: np.ndarray) -> "RegressaoLinearFechada":
        A = _com_intercepto(X)
        y = _conferir(A, y)

        # TODO: resolver (AᵀA) w = Aᵀy e guardar o resultado em self.w_
        # Dica: np.linalg.solve(...) e mais estavel e mais rapido do que
        # inverter AᵀA explicitamente com np.linalg.inv.
        raise NotImplementedError("implemente as equacoes normais")

        return self

    def prever(self, X: np.ndarray) -> np.ndarray:
        if self.w_ is None:
            raise RuntimeError("chame treinar() antes de prever()")
        return _com_intercepto(X) @ self.w_

    @property
    def intercepto_(self) -> float:
        return float(self.w_[0])

    @property
    def coef_(self) -> np.ndarray:
        return self.w_[1:]


class RegressaoLinearGD:
    """Mesma solução, alcançada por descida de gradiente.

    Função de custo (erro quadrático médio):

        J(w) = (1/n)‖Aw - y‖²

    Gradiente:

        ∇J = (2/n) Aᵀ(Aw - y)

    Atualização: w ← w - taxa * ∇J

    O treino guarda o histórico de custo, norma do gradiente e pesos por época,
    que é o material das figuras de convergência.
    """

    def __init__(self, taxa: float = 0.1, epocas: int = 2000, tolerancia: float = 1e-10) -> None:
        self.taxa = float(taxa)
        self.epocas = int(epocas)
        self.tolerancia = float(tolerancia)
        self.w_: np.ndarray | None = None
        self.epocas_executadas_ = 0
        self.historico_: dict[str, list] = {}

    def custo(self, A: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        return float(np.mean((A @ w - y) ** 2))

    def treinar(self, X: np.ndarray, y: np.ndarray) -> "RegressaoLinearGD":
        A = _com_intercepto(X)
        y = _conferir(A, y)
        n, d = A.shape
        w = np.zeros(d)

        self.historico_ = {"custo": [], "norma_gradiente": [], "pesos": []}

        for epoca in range(self.epocas):
            # TODO: calcular o gradiente conforme a formula do docstring
            #       e atualizar w dando um passo de tamanho self.taxa
            raise NotImplementedError("implemente o passo do Gradient Descent")

            self.historico_["custo"].append(self.custo(A, y, w))
            self.historico_["norma_gradiente"].append(float(np.linalg.norm(gradiente)))
            self.historico_["pesos"].append(w.copy())

            if not np.isfinite(w).all():
                raise FloatingPointError(
                    f"divergiu na epoca {epoca}: taxa={self.taxa} alta demais. "
                    "Reduza a taxa ou confirme que X esta padronizado."
                )
            if np.linalg.norm(gradiente) < self.tolerancia:
                break

        self.w_ = w
        self.epocas_executadas_ = epoca + 1
        return self

    def prever(self, X: np.ndarray) -> np.ndarray:
        if self.w_ is None:
            raise RuntimeError("chame treinar() antes de prever()")
        return _com_intercepto(X) @ self.w_

    @property
    def intercepto_(self) -> float:
        return float(self.w_[0])

    @property
    def coef_(self) -> np.ndarray:
        return self.w_[1:]


# --------------------------------------------------------------------------- #
# Diagnóstico da colinearidade
# --------------------------------------------------------------------------- #
def vif(X: np.ndarray) -> np.ndarray:
    """Fator de inflação da variância de cada coluna.

        VIF_j = 1 / (1 - R²_j)

    onde R²_j vem de regredir a coluna j contra todas as outras. VIF acima de
    10 é o limiar usual de alerta: o coeficiente daquele atributo tem variância
    inflada e o sinal pode inverter com uma pequena mudança na amostra.
    """
    X = np.asarray(X, dtype=float)
    fatores = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        outros = _com_intercepto(np.delete(X, j, axis=1))
        beta, *_ = np.linalg.lstsq(outros, X[:, j], rcond=None)
        residuo = X[:, j] - outros @ beta
        total = ((X[:, j] - X[:, j].mean()) ** 2).sum()
        r2 = 1 - (residuo**2).sum() / total
        fatores[j] = np.inf if r2 >= 1 else 1 / (1 - r2)
    return fatores


def pesos_por_reamostragem(
    X: np.ndarray, y: np.ndarray, repeticoes: int = 200, semente: int = 42
) -> np.ndarray:
    """Reajusta o modelo em amostras bootstrap e devolve a matriz de pesos.

    Sem penalização, é assim que se mostra a instabilidade causada pela
    colinearidade: atributos com VIF alto produzem coeficientes que variam
    muito — e às vezes trocam de sinal — de reamostra para reamostra.

    Retorna matriz (repeticoes, d+1).
    """
    rng = np.random.default_rng(semente)
    X, y = np.asarray(X, float), np.asarray(y, float)
    n = len(y)
    saida = []
    for _ in range(repeticoes):
        sorteio = rng.integers(0, n, n)
        saida.append(RegressaoLinearFechada().treinar(X[sorteio], y[sorteio]).w_)
    return np.array(saida)


def distancia_ate_a_solucao_exata(
    gd: RegressaoLinearGD, fechada: RegressaoLinearFechada
) -> np.ndarray:
    """Norma da diferença entre os pesos do GD e a solução exata, por época.

    É a prova de que o GD converge para o lugar certo — e a resposta direta à
    pergunta do enunciado sobre o ponto em que os parâmetros param de mudar.
    """
    if fechada.w_ is None or not gd.historico_.get("pesos"):
        raise RuntimeError("treine os dois modelos antes de comparar")
    return np.array([np.linalg.norm(w - fechada.w_) for w in gd.historico_["pesos"]])
