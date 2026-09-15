#!/usr/bin/env python3
"""Conferencia: o Gradient Descent chega mesmo na solucao das equacoes normais?

Os dois algoritmos minimizam a MESMA funcao de custo, que e convexa e quadratica
e por isso tem um unico minimo. Entao eles tem que chegar no mesmo vetor de
pesos. Este script mostra isso acontecendo, em vez de so afirmar:

    1. treina os dois na mesma configuracao;
    2. imprime os pesos lado a lado, atributo por atributo;
    3. mostra a trajetoria do GD — a que distancia da solucao exata ele estava
       na epoca 10, 100, 1.000, 10.000 e assim por diante.

A conclusao que se le na tabela: as epocas controlam a PRECISAO; quem decide o
DESTINO e a convexidade do custo.

Uso:
    python conferencia_gd.py
    python conferencia_gd.py --codificacao onehot --alvo-log
    python conferencia_gd.py --taxa 0.1 --epocas 20000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# O console do Windows abre em cp1252 e quebra ao imprimir setas e simbolos.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
from src import dados as D  # noqa: E402
from src import metricas as M  # noqa: E402
from src.modelos import (  # noqa: E402
    RegressaoLinearFechada,
    RegressaoLinearGD,
    distancia_ate_a_solucao_exata,
    espectro,
)


def medir(modelo, conjunto: dict) -> dict:
    """R2 e MSE no teste, sempre em dolares."""
    y, previsto = conjunto["y_teste"], modelo.prever(conjunto["X_teste"])
    if conjunto["alvo_em_log"]:
        y, previsto = M.desfazer_log(y), M.desfazer_log(previsto)
    return M.avaliar(y, previsto)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--codificacao", choices=("ordinal", "onehot"), default="ordinal")
    p.add_argument("--alvo-log", action="store_true", help="treinar sobre log(price)")
    p.add_argument("--taxa", type=float, default=0.2)
    p.add_argument("--epocas", type=int, default=50_000)
    p.add_argument("--semente", type=int, default=42)
    args = p.parse_args()

    rotulo = f"{args.codificacao}/{'log' if args.alvo_log else 'USD'}"
    conjunto = D.preparar(
        codificacao=args.codificacao, alvo_em_log=args.alvo_log, semente=args.semente
    )
    X, y = conjunto["X_treino"], conjunto["y_treino"]

    esp = espectro(X)
    print(f"\nCONFIGURACAO {rotulo}  —  {X.shape[1]} preditoras, {len(y):,} linhas de treino")
    print(f"  taxa = {args.taxa:g}   (limite estavel 2/lambda_max = {esp['taxa_maxima']:.4f})")
    print(f"  numero de condicao da hessiana: kappa = {esp['kappa']:,.1f}")

    fechada = RegressaoLinearFechada().treinar(X, y)
    print(f"\n  rodando {args.epocas:,} epocas de gradient descent...")
    gd = RegressaoLinearGD(taxa=args.taxa, epocas=args.epocas).treinar(X, y)
    print(f"  parou na epoca {gd.epocas_executadas_:,}", end="")
    print(" (atingiu a tolerancia do gradiente)" if gd.epocas_executadas_ < args.epocas
          else " (bateu no teto de epocas)")

    # ------------------------------------------------------------------ #
    print("\n1) OS PESOS, LADO A LADO")
    nomes = ["intercepto"] + list(conjunto["colunas"])
    print(f"  {'atributo':<16}{'eq. normais':>16}{'grad. descent':>16}{'diferenca':>14}")
    print("  " + "-" * 62)
    for nome, wf, wg in zip(nomes, fechada.w_, gd.w_):
        print(f"  {nome:<16}{wf:>16.6f}{wg:>16.6f}{wg - wf:>14.2e}")
    print("  " + "-" * 62)
    print(f"  maior diferenca absoluta: {np.max(np.abs(gd.w_ - fechada.w_)):.3e}")
    print(f"  norma da diferenca:       {np.linalg.norm(gd.w_ - fechada.w_):.3e}")

    # ------------------------------------------------------------------ #
    print("\n2) A TRAJETORIA — a que distancia o GD estava, epoca a epoca")
    distancia = distancia_ate_a_solucao_exata(gd, fechada)
    custo = np.array(gd.historico_["custo"])
    custo_final = custo[-1]

    marcos = [m for m in (1, 10, 100, 331, 1_000, 5_000, 10_000, 20_000, 30_000, 40_000)
              if m <= len(distancia)] + [len(distancia)]
    print(f"  {'epoca':>8}{'||w - w*||':>16}{'custo J(w)':>18}{'custo vs final':>16}")
    print("  " + "-" * 58)
    for m in marcos:
        i = m - 1
        print(f"  {m:>8,}{distancia[i]:>16.3e}{custo[i]:>18,.2f}"
              f"{custo[i] / custo_final - 1:>15.2%}")

    # ------------------------------------------------------------------ #
    print("\n3) O QUE ISSO MOSTRA")
    dentro_1pct = int(np.argmax(custo <= custo_final * 1.01))
    abaixo = distancia < 1e-6
    print(f"  o CUSTO entra em 1% do valor final na epoca {dentro_1pct:,}")
    if abaixo.any():
        print(f"  os PESOS entram em ||w - w*|| < 1e-6 na epoca {int(np.argmax(abaixo)):,}")
    else:
        print(f"  os PESOS nao atingiram ||w - w*|| < 1e-6 em {len(distancia):,} epocas")
    print("  ou seja: o custo satura MUITO antes dos parametros pararem de mudar.")

    mf, mg = medir(fechada, conjunto), medir(gd, conjunto)
    print(f"\n  R2  de teste: {mf['r2']:.15f}  (eq. normais)")
    print(f"                {mg['r2']:.15f}  (gradient descent)")
    print(f"  MSE de teste: {mf['mse']:,.6f}  (eq. normais)")
    print(f"                {mg['mse']:,.6f}  (gradient descent)")
    print(f"\n  As metricas coincidem, mas nao sao bit a bit identicas — a diferenca"
          f"\n  de {abs(mg['r2'] - mf['r2']):.1e} em R2 e o residuo de duas contas de fato distintas."
          f"\n  Diferenca EXATAMENTE zero e que seria suspeito.\n")


if __name__ == "__main__":
    main()
