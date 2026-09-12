#!/usr/bin/env python3
"""Treino e comparação entre configurações e entre os dois algoritmos.

Responde aos itens "avalie os dados com R2 e MSE" e "faça uma análise de
performance dos tipos de regressão".

Compara quatro configurações — codificação ordinal ou one-hot, alvo em dólares
ou em log — cada uma treinada pelas equações normais e por Gradient Descent.
Quando o alvo é log, as previsões voltam para dólares antes de medir; senão as
métricas não são comparáveis entre as linhas da tabela.

O Gradient Descent roda 50.000 épocas com taxa 0,2 por padrão. Não é exagero:
a colinearidade entre carat, x, y e z deixa o vale do custo muito alongado, e
com menos épocas o GD acerta as métricas mas ainda não chegou aos mesmos pesos
da solução exata. A análise disso está em 03_analises.py.

Uso:
    python 02_treino.py
    python 02_treino.py --taxa 0.1 --epocas 100000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# O console do Windows abre em cp1252 e quebra ao imprimir setas e simbolos.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
from src import dados as D  # noqa: E402
from src import metricas as M  # noqa: E402
from src.modelos import RegressaoLinearFechada, RegressaoLinearGD, espectro  # noqa: E402

RESULTADOS = RAIZ / "resultados"


def medir(modelo, conjunto: dict, particao: str) -> dict:
    """Avalia sempre na escala de dólares, mesmo se o treino foi em log."""
    X, y = conjunto[f"X_{particao}"], conjunto[f"y_{particao}"]
    previsto = modelo.prever(X)
    if conjunto["alvo_em_log"]:
        y, previsto = M.desfazer_log(y), M.desfazer_log(previsto)
    return M.avaliar(y, previsto)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--taxa", type=float, default=0.2, help="taxa de aprendizado do GD")
    p.add_argument("--epocas", type=int, default=50_000)
    p.add_argument("--semente", type=int, default=42)
    args = p.parse_args()

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    linhas = []

    for codificacao in ("ordinal", "onehot"):
        for alvo_em_log in (False, True):
            conjunto = D.preparar(
                codificacao=codificacao, alvo_em_log=alvo_em_log, semente=args.semente
            )
            rotulo = f"{codificacao}/{'log' if alvo_em_log else 'USD'}"
            print(f"\n{rotulo}  —  {conjunto['X_treino'].shape[1]} preditoras, "
                  f"{len(conjunto['y_treino']):,} treino / {len(conjunto['y_teste']):,} teste")

            for nome, modelo in (
                ("equacoes normais", RegressaoLinearFechada()),
                ("gradient descent", RegressaoLinearGD(taxa=args.taxa, epocas=args.epocas)),
            ):
                inicio = time.perf_counter()
                modelo.treinar(conjunto["X_treino"], conjunto["y_treino"])
                duracao = time.perf_counter() - inicio

                treino, teste = medir(modelo, conjunto, "treino"), medir(modelo, conjunto, "teste")
                linhas.append({
                    "codificacao": codificacao,
                    "alvo": "log" if alvo_em_log else "USD",
                    "metodo": nome,
                    "r2_treino": treino["r2"], "r2_teste": teste["r2"],
                    "mse_teste": teste["mse"], "rmse_teste": teste["rmse"],
                    "segundos": duracao,
                    "epocas": getattr(modelo, "epocas_executadas_", None),
                })
                print(f"  {nome:18} R²treino={treino['r2']:.4f}  R²teste={teste['r2']:.4f}  "
                      f"RMSE={teste['rmse']:9.2f} USD  ({duracao:.3f}s)")

    tabela = pd.DataFrame(linhas)
    tabela.to_csv(RESULTADOS / "metricas.csv", index=False)

    melhor = tabela.loc[tabela.r2_teste.idxmax()]
    print(f"\nmelhor configuracao: {melhor.codificacao}/{melhor.alvo} por {melhor.metodo}"
          f"  →  R² teste {melhor.r2_teste:.4f}, RMSE {melhor.rmse_teste:.2f} USD")

    # Os dois algoritmos resolvem o mesmo problema: têm que chegar no mesmo w.
    print("\nCONFERENCIA: Gradient Descent contra a solucao exata")
    conjunto = D.preparar(codificacao="ordinal", alvo_em_log=False, semente=args.semente)
    fechada = RegressaoLinearFechada().treinar(conjunto["X_treino"], conjunto["y_treino"])
    gd = RegressaoLinearGD(taxa=args.taxa, epocas=args.epocas).treinar(
        conjunto["X_treino"], conjunto["y_treino"]
    )
    print(f"  epocas executadas            {gd.epocas_executadas_:,}")
    print(f"  maior diferenca entre pesos  {np.max(np.abs(gd.w_ - fechada.w_)):.3e}")
    print(f"  R² teste (equacoes normais)  {medir(fechada, conjunto, 'teste')['r2']:.6f}")
    print(f"  R² teste (gradient descent)  {medir(gd, conjunto, 'teste')['r2']:.6f}")

    print(f"\ntabela salva em {RESULTADOS / 'metricas.csv'}")


if __name__ == "__main__":
    main()
