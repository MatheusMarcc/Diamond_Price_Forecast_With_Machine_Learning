"""Carga, limpeza, codificação, divisão e padronização da base diamonds.

Toda estatística usada para transformar os dados (média, desvio) é calculada
apenas no conjunto de treino. Ajustar no conjunto completo vaza informação do
teste para o treino e infla as métricas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CAMINHO_PADRAO = Path(__file__).resolve().parent.parent / "dados" / "diamonds.csv"

ALVO = "price"
NUMERICAS = ["carat", "depth", "table", "x", "y", "z"]
CATEGORICAS = ["cut", "color", "clarity"]

# Ordem de QUALIDADE, do pior para o melhor. Não é a ordem alfabética em que os
# níveis aparecem no ARFF do OpenML — usar a ordem do arquivo inverte o sinal
# dos pesos de `color` e embaralha o de `cut`.
ORDEM = {
    "cut": ["Fair", "Good", "Very Good", "Premium", "Ideal"],
    "color": ["J", "I", "H", "G", "F", "E", "D"],
    "clarity": ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
}

# Nenhuma pedra do conjunto tem mais de 5,01 quilates; dimensões acima de 20 mm
# são erro de digitação, e dimensão zero é fisicamente impossível.
LIMITE_MM = 20.0


def carregar(caminho: str | Path = CAMINHO_PADRAO) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    faltando = ({ALVO} | set(NUMERICAS) | set(CATEGORICAS)) - set(df.columns)
    if faltando:
        raise ValueError(f"colunas ausentes no arquivo: {sorted(faltando)}")
    return df


def limpar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove linhas com dimensões impossíveis. Devolve (df limpo, relatório)."""
    dims = df[["x", "y", "z"]]
    zerada = (dims == 0).any(axis=1)
    absurda = (dims > LIMITE_MM).any(axis=1)
    descartar = zerada | absurda

    relatorio = {
        "linhas_originais": len(df),
        "dimensao_zero": int(zerada.sum()),
        "dimensao_acima_de_20mm": int(absurda.sum()),
        "removidas": int(descartar.sum()),
        "linhas_finais": int((~descartar).sum()),
    }
    return df.loc[~descartar].reset_index(drop=True), relatorio


def codificar_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """Mapeia cut/color/clarity para inteiros na ordem de qualidade (0 = pior)."""
    saida = df.copy()
    for coluna, niveis in ORDEM.items():
        codigos = pd.Categorical(saida[coluna], categories=niveis, ordered=True).codes
        if (codigos < 0).any():
            desconhecidos = sorted(set(saida.loc[codigos < 0, coluna]))
            raise ValueError(f"valores fora da ordem declarada em {coluna}: {desconhecidos}")
        saida[coluna] = codigos
    return saida


def codificar_onehot(df: pd.DataFrame, descartar_primeira: bool = True) -> pd.DataFrame:
    """Uma coluna binária por nível, na ordem de qualidade.

    Com `descartar_primeira`, o nível de pior qualidade vira a referência e o
    modelo fica identificável (sem colinearidade exata com o intercepto).
    """
    saida = df.copy()
    for coluna, niveis in ORDEM.items():
        saida[coluna] = pd.Categorical(saida[coluna], categories=niveis, ordered=True)
    return pd.get_dummies(saida, columns=CATEGORICAS, drop_first=descartar_primeira, dtype=float)


def separar_alvo(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return df.drop(columns=[ALVO]), df[ALVO].to_numpy(float)


def dividir(n: int, fracao_teste: float = 0.2, semente: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Índices de treino e teste, embaralhados com semente fixa.

    A base não é temporal — cada linha é uma pedra independente — então a
    divisão aleatória é apropriada aqui.
    """
    indices = np.random.default_rng(semente).permutation(n)
    corte = int(round(n * (1 - fracao_teste)))
    return indices[:corte], indices[corte:]


class Padronizador:
    """z = (x - média) / desvio, com os parâmetros vindos só do treino."""

    def __init__(self) -> None:
        self.media_: np.ndarray | None = None
        self.desvio_: np.ndarray | None = None

    def ajustar(self, X: np.ndarray) -> "Padronizador":
        X = np.asarray(X, dtype=float)
        self.media_ = X.mean(axis=0)
        desvio = X.std(axis=0)
        desvio[desvio == 0] = 1.0  # coluna constante não é reescalada
        self.desvio_ = desvio
        return self

    def transformar(self, X: np.ndarray) -> np.ndarray:
        if self.media_ is None or self.desvio_ is None:
            raise RuntimeError("chame ajustar() com o treino antes de transformar()")
        return (np.asarray(X, dtype=float) - self.media_) / self.desvio_

    def ajustar_transformar(self, X: np.ndarray) -> np.ndarray:
        return self.ajustar(X).transformar(X)


def preparar(
    caminho: str | Path = CAMINHO_PADRAO,
    codificacao: str = "ordinal",
    alvo_em_log: bool = False,
    fracao_teste: float = 0.2,
    semente: int = 42,
) -> dict:
    """Executa o pipeline inteiro e devolve tudo que os scripts precisam.

    Retorna um dicionário com X_treino, X_teste (já padronizados), y_treino,
    y_teste, os nomes das colunas, o relatório de limpeza e o padronizador.
    """
    bruto = carregar(caminho)
    limpo, relatorio = limpar(bruto)

    if codificacao == "ordinal":
        tabela = codificar_ordinal(limpo)
    elif codificacao == "onehot":
        tabela = codificar_onehot(limpo)
    else:
        raise ValueError("codificacao deve ser 'ordinal' ou 'onehot'")

    X_df, y = separar_alvo(tabela)
    if alvo_em_log:
        y = np.log(y)

    treino, teste = dividir(len(y), fracao_teste, semente)
    X = X_df.to_numpy(float)

    padronizador = Padronizador().ajustar(X[treino])
    return {
        "X_treino": padronizador.transformar(X[treino]),
        "X_teste": padronizador.transformar(X[teste]),
        "y_treino": y[treino],
        "y_teste": y[teste],
        "colunas": list(X_df.columns),
        "relatorio_limpeza": relatorio,
        "padronizador": padronizador,
        "alvo_em_log": alvo_em_log,
        "df_limpo": limpo,
        "indices": (treino, teste),
    }
