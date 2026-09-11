#!/usr/bin/env python3
"""Estudo introdutório da base — item 3 do enunciado.

Produz, em figuras/ e resultados/:
    01_distribuicao_alvo.png    histograma do preço em escala linear e log
    02_correlacao.png           matriz de correlação das numéricas
    03_categoricas.png          contagem por nível de cut, color e clarity
    04_tsne_sem_rotulo.png      t-SNE sem cor
    05_tsne_com_rotulo.png      t-SNE colorido por quartil de preço
    estatisticas.csv            descritivas por coluna
    correlacao.csv              matriz de correlação
    limpeza.csv                 o que foi removido e por quê

Uso:
    python 01_estudo_base.py
    python 01_estudo_base.py --amostra-tsne 3000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem janela: salva direto em arquivo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
from src import dados as D  # noqa: E402

FIGURAS = RAIZ / "figuras"
RESULTADOS = RAIZ / "resultados"

DESCRICAO = {
    "carat": "peso da pedra em quilates (1 ct = 0,2 g)",
    "cut": "qualidade da lapidação: Fair < Good < Very Good < Premium < Ideal",
    "color": "cor: J (pior) ate D (incolor, melhor)",
    "clarity": "pureza: I1 (pior) ate IF (impecavel)",
    "depth": "profundidade percentual = 2z/(x+y) x 100",
    "table": "largura da face superior como % da largura maxima",
    "price": "ALVO — preco em dolares",
    "x": "comprimento em mm",
    "y": "largura em mm",
    "z": "profundidade em mm",
}


def figura(nome: str) -> Path:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    return FIGURAS / nome


def distribuicao_do_alvo(y: pd.Series) -> None:
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4))
    eixos[0].hist(y, bins=80, color="#2F6690", edgecolor="none")
    eixos[0].set_title(f"preço em dólares — assimetria {y.skew():.3f}")
    eixos[0].set_xlabel("price")
    eixos[0].set_ylabel("frequência")

    log_y = np.log(y)
    eixos[1].hist(log_y, bins=80, color="#1F918B", edgecolor="none")
    eixos[1].set_title(f"log(preço) — assimetria {log_y.skew():.3f}")
    eixos[1].set_xlabel("log(price)")

    fig.suptitle("Desbalanceamento do alvo", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figura("01_distribuicao_alvo.png"), dpi=150)
    plt.close(fig)


def matriz_correlacao(df: pd.DataFrame) -> pd.DataFrame:
    colunas = D.NUMERICAS + [D.ALVO]
    cor = df[colunas].corr()

    fig, eixo = plt.subplots(figsize=(7.5, 6.5))
    imagem = eixo.imshow(cor, cmap="RdBu_r", vmin=-1, vmax=1)
    eixo.set_xticks(range(len(colunas)), colunas, rotation=45, ha="right")
    eixo.set_yticks(range(len(colunas)), colunas)
    for i in range(len(colunas)):
        for j in range(len(colunas)):
            valor = cor.iloc[i, j]
            eixo.text(j, i, f"{valor:.2f}", ha="center", va="center",
                      fontsize=8, color="white" if abs(valor) > 0.55 else "black")
    fig.colorbar(imagem, ax=eixo, shrink=0.8)
    eixo.set_title("Correlação de Pearson", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figura("02_correlacao.png"), dpi=150)
    plt.close(fig)
    return cor


def contagem_categoricas(df: pd.DataFrame) -> None:
    fig, eixos = plt.subplots(1, 3, figsize=(13, 3.8))
    for eixo, coluna in zip(eixos, D.CATEGORICAS):
        niveis = D.ORDEM[coluna]
        alturas = [int((df[coluna] == n).sum()) for n in niveis]
        eixo.bar(niveis, alturas, color="#2F6690")
        eixo.set_title(f"{coluna}  ({min(alturas):,} a {max(alturas):,})")
        eixo.tick_params(axis="x", rotation=45)
    fig.suptitle("Desbalanceamento das categóricas (pior → melhor)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figura("03_categoricas.png"), dpi=150)
    plt.close(fig)


def tsne(df: pd.DataFrame, tamanho: int, semente: int) -> None:
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("\n  ! scikit-learn nao instalado — t-SNE pulado.")
        print("    rode:  python -m pip install scikit-learn")
        return

    rng = np.random.default_rng(semente)
    amostra = df.iloc[rng.choice(len(df), size=min(tamanho, len(df)), replace=False)]

    X = D.codificar_ordinal(amostra).drop(columns=[D.ALVO]).to_numpy(float)
    X = D.Padronizador().ajustar_transformar(X)
    y = amostra[D.ALVO].to_numpy(float)

    print(f"  rodando t-SNE em {len(X):,} pontos...")
    Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=semente).fit_transform(X)

    fig, eixo = plt.subplots(figsize=(6.5, 6))
    eixo.scatter(Z[:, 0], Z[:, 1], s=4, alpha=0.5, color="#3B4A46")
    eixo.set_title(f"t-SNE sem rótulos (n={len(X):,})", fontweight="bold")
    eixo.set_xticks([]); eixo.set_yticks([])
    fig.tight_layout()
    fig.savefig(figura("04_tsne_sem_rotulo.png"), dpi=150)
    plt.close(fig)

    # Em regressão não existe classe para colorir: discretizamos o alvo em
    # quartis e usamos isso apenas como cor.
    quartis = pd.qcut(y, 4, labels=["Q1 barato", "Q2", "Q3", "Q4 caro"])
    cores = ["#7FCB48", "#1F918B", "#2F6690", "#3E2E62"]
    fig, eixo = plt.subplots(figsize=(6.5, 6))
    for cor, nivel in zip(cores, quartis.categories):
        m = quartis == nivel
        eixo.scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.6, color=cor, label=str(nivel))
    eixo.legend(markerscale=4, frameon=False, loc="best")
    eixo.set_title("t-SNE colorido por quartil de preço", fontweight="bold")
    eixo.set_xticks([]); eixo.set_yticks([])
    fig.tight_layout()
    fig.savefig(figura("05_tsne_com_rotulo.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--amostra-tsne", type=int, default=5000, help="pontos usados no t-SNE")
    p.add_argument("--semente", type=int, default=42)
    args = p.parse_args()

    RESULTADOS.mkdir(parents=True, exist_ok=True)

    bruto = D.carregar()
    limpo, relatorio = D.limpar(bruto)

    print("LIMPEZA")
    for chave, valor in relatorio.items():
        print(f"  {chave:26} {valor:>8,}")
    pd.Series(relatorio).to_frame("valor").to_csv(RESULTADOS / "limpeza.csv")

    print("\nATRIBUTOS")
    for coluna in bruto.columns:
        print(f"  {coluna:8} {DESCRICAO[coluna]}")

    print("\nDESCRITIVAS (apos limpeza)")
    descritivas = limpo.describe(include="all").T
    print(limpo[D.NUMERICAS + [D.ALVO]].describe().T.to_string())
    descritivas.to_csv(RESULTADOS / "estatisticas.csv")

    y = limpo[D.ALVO]
    print(f"\nassimetria do preco: {y.skew():.3f}   |   de log(preco): {np.log(y).skew():.3f}")

    distribuicao_do_alvo(y)
    cor = matriz_correlacao(limpo)
    cor.to_csv(RESULTADOS / "correlacao.csv")
    contagem_categoricas(limpo)

    print("\nCORRELACAO COM O ALVO")
    for nome, valor in cor[D.ALVO].drop(D.ALVO).sort_values(key=abs, ascending=False).items():
        print(f"  {nome:8} {valor:+.3f}")

    tsne(limpo, args.amostra_tsne, args.semente)

    print(f"\nfiguras em  {FIGURAS}")
    print(f"tabelas em  {RESULTADOS}")


if __name__ == "__main__":
    main()
