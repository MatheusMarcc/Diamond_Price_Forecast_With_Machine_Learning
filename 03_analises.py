#!/usr/bin/env python3
"""Análise dos pesos e da convergência do Gradient Descent.

Responde aos itens "faça uma análise dos pesos de cada atributo" e "mostre a
variação dos parâmetros com Gradient Descent [...] tem algum ponto que eles não
se alteram mais?".

A estabilidade dos coeficientes é medida pelo fator de inflação da variância
(VIF) e por reamostragem bootstrap.

Produz em figuras/ e resultados/:
    06_pesos_e_vif.png          coeficiente e VIF de cada atributo
    07_estabilidade.png         dispersão dos coeficientes em 200 reamostras
    08_convergencia.png         custo por época e distância até a solução exata
    09_taxa_aprendizado.png     efeito da taxa sobre a convergência
    pesos.csv                   coeficiente, VIF e desvio bootstrap por atributo

Uso:
    python 03_analises.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# O console do Windows abre em cp1252 e quebra ao imprimir setas e simbolos.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
from src import dados as D  # noqa: E402
from src.modelos import (  # noqa: E402
    RegressaoLinearFechada,
    RegressaoLinearGD,
    distancia_ate_a_solucao_exata,
    espectro,
    pesos_por_reamostragem,
    vif,
)

FIGURAS = RAIZ / "figuras"
RESULTADOS = RAIZ / "resultados"
AZUL, VERDE, ROXO, LARANJA = "#2F6690", "#1F918B", "#3E2E62", "#B4553B"


def pesos_e_vif(conjunto: dict) -> pd.DataFrame:
    modelo = RegressaoLinearFechada().treinar(conjunto["X_treino"], conjunto["y_treino"])
    fatores = vif(conjunto["X_treino"])
    tabela = pd.DataFrame(
        {"coeficiente": modelo.coef_, "vif": fatores}, index=conjunto["colunas"]
    ).sort_values("coeficiente", key=abs, ascending=False)

    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.6))
    cores = [LARANJA if c < 0 else AZUL for c in tabela.coeficiente]
    eixos[0].barh(tabela.index[::-1], tabela.coeficiente[::-1], color=cores[::-1])
    eixos[0].axvline(0, color="#3B4A46", linewidth=0.8)
    eixos[0].set_title("Coeficientes padronizados\n(laranja = sinal negativo)")
    eixos[0].set_xlabel("peso")

    eixos[1].barh(tabela.index[::-1], tabela.vif[::-1], color=VERDE)
    eixos[1].axvline(10, color=LARANJA, linestyle="--", linewidth=1, label="limiar 10")
    eixos[1].set_xscale("log")
    eixos[1].set_title("Fator de inflação da variância (VIF)")
    eixos[1].set_xlabel("VIF (escala log)")
    eixos[1].legend(frameon=False)

    fig.suptitle("Pesos e colinearidade", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURAS / "06_pesos_e_vif.png", dpi=150)
    plt.close(fig)
    return tabela


def estabilidade(conjunto: dict, repeticoes: int, semente: int) -> pd.Series:
    """Quanto cada coeficiente varia quando a amostra de treino muda."""
    pesos = pesos_por_reamostragem(
        conjunto["X_treino"], conjunto["y_treino"], repeticoes, semente
    )[:, 1:]  # descarta o intercepto
    colunas = conjunto["colunas"]

    fig, eixo = plt.subplots(figsize=(9, 5))
    # Os rotulos sao postos por set_xticks, e nao pelo argumento `labels` de
    # boxplot: esse argumento virou `tick_labels` no matplotlib 3.9 e sai de vez
    # no 3.11, e requirements.txt admite qualquer versao a partir de 3.8.
    eixo.boxplot([pesos[:, j] for j in range(pesos.shape[1])],
                 showfliers=False, medianprops={"color": ROXO})
    eixo.set_xticks(range(1, len(colunas) + 1), colunas)
    eixo.axhline(0, color="#3B4A46", linewidth=0.8, linestyle="--")
    eixo.tick_params(axis="x", rotation=45)
    eixo.set_ylabel("coeficiente padronizado")
    eixo.set_title(f"Estabilidade dos pesos em {repeticoes} reamostras bootstrap",
                   fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURAS / "07_estabilidade.png", dpi=150)
    plt.close(fig)

    return pd.Series(pesos.std(axis=0), index=colunas, name="desvio_bootstrap")


def convergencia(conjunto: dict, taxa: float, epocas: int) -> None:
    fechada = RegressaoLinearFechada().treinar(conjunto["X_treino"], conjunto["y_treino"])
    gd = RegressaoLinearGD(taxa=taxa, epocas=epocas).treinar(
        conjunto["X_treino"], conjunto["y_treino"]
    )
    distancia = distancia_ate_a_solucao_exata(gd, fechada)

    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.2))
    eixos[0].plot(gd.historico_["custo"], color=AZUL)
    eixos[0].set_yscale("log")
    eixos[0].set_xlabel("época"); eixos[0].set_ylabel("J(w)")
    eixos[0].set_title("Custo por época")

    eixos[1].plot(distancia, color=VERDE)
    eixos[1].set_yscale("log")
    eixos[1].set_xlabel("época"); eixos[1].set_ylabel("‖w − w*‖")
    eixos[1].set_title("Distância até a solução exata")

    fig.suptitle(f"Convergência do Gradient Descent (taxa={taxa:g})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURAS / "08_convergencia.png", dpi=150)
    plt.close(fig)

    limiar = 1e-6
    abaixo = distancia < limiar
    print(f"  custo inicial {gd.historico_['custo'][0]:.4e} → final {gd.historico_['custo'][-1]:.4e}")
    print(f"  distancia final ate a solucao exata: {distancia[-1]:.3e}")
    if abaixo.any():
        print(f"  os pesos estabilizam (‖w−w*‖ < {limiar:g}) por volta da epoca {int(np.argmax(abaixo)):,}")
    else:
        print(f"  nao atingiu ‖w−w*‖ < {limiar:g} em {len(distancia):,} epocas — "
              "aumente as epocas ou a taxa")


def varredura_taxa(conjunto: dict, taxas: list[float], epocas: int) -> None:
    esp = espectro(conjunto["X_treino"])
    print(f"  taxa maxima estavel = 2/lambda_max = {esp['taxa_maxima']:.4f}")
    print(f"  numero de condicao da hessiana = {esp['kappa']:,.0f}")
    cores = [AZUL, VERDE, ROXO, LARANJA, "#6A7A74"]
    fig, eixo = plt.subplots(figsize=(7.5, 4.6))
    for cor, taxa in zip(cores, taxas):
        try:
            gd = RegressaoLinearGD(taxa=taxa, epocas=epocas).treinar(
                conjunto["X_treino"], conjunto["y_treino"]
            )
            eixo.plot(gd.historico_["custo"], color=cor, label=f"taxa = {taxa:g}")
        except FloatingPointError:
            print(f"  taxa {taxa:g} divergiu")
            eixo.plot([], [], color=cor, label=f"taxa = {taxa:g} (divergiu)")
    eixo.set_yscale("log")
    eixo.set_xlabel("época"); eixo.set_ylabel("J(w)")
    eixo.set_title("Efeito da taxa de aprendizado", fontweight="bold")
    eixo.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURAS / "09_taxa_aprendizado.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--taxa", type=float, default=0.2)
    p.add_argument("--epocas", type=int, default=50_000)
    p.add_argument("--reamostras", type=int, default=200)
    p.add_argument("--semente", type=int, default=42)
    args = p.parse_args()

    FIGURAS.mkdir(parents=True, exist_ok=True)
    RESULTADOS.mkdir(parents=True, exist_ok=True)

    conjunto = D.preparar(codificacao="ordinal", alvo_em_log=False, semente=args.semente)

    print("PESOS E COLINEARIDADE")
    tabela = pesos_e_vif(conjunto)
    print(tabela.to_string(float_format=lambda v: f"{v:10.3f}"))

    print(f"\nESTABILIDADE ({args.reamostras} reamostras)")
    desvios = estabilidade(conjunto, args.reamostras, args.semente)
    tabela = tabela.join(desvios)
    tabela["coef_sobre_desvio"] = tabela.coeficiente / tabela.desvio_bootstrap
    tabela.to_csv(RESULTADOS / "pesos.csv")
    print(desvios.to_string(float_format=lambda v: f"{v:10.3f}"))

    print("\nCONVERGENCIA")
    convergencia(conjunto, args.taxa, args.epocas)

    print("\nVARREDURA DE TAXA")
    varredura_taxa(conjunto, [0.001, 0.01, 0.1, 0.2, 0.3], epocas=400)

    print(f"\nfiguras em {FIGURAS}")
    print(f"tabela  em {RESULTADOS / 'pesos.csv'}")


if __name__ == "__main__":
    main()
