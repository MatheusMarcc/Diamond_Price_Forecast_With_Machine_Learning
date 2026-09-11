#!/usr/bin/env python3
"""Coleta séries macroeconômicas (SGS/Banco Central) e de mercado (Yahoo Finance).

Usa apenas a biblioteca padrão — não precisa instalar nada para rodar.
Gera CSVs em ./dados, incluindo um painel mensal alinhado pronto para regressão.

Exemplos:
    python coleta_dados.py
    python coleta_dados.py --inicio 2010-01-01 --tickers ^BVSP PETR4.SA VALE3.SA
    python coleta_dados.py --so-sgs
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

# O console do Windows abre em cp1252 e quebra ao imprimir acento.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CABECALHO = {"User-Agent": "Mozilla/5.0 (projeto academico; coleta de dados)"}

# Códigos verificados na API do SGS em 08/09/2026.
# Catálogo completo: https://www3.bcb.gov.br/sgspub/  (busque pelo nome da série)
SERIES_SGS: dict[str, tuple[int, str, str]] = {
    "selic_meta":    (432,   "% a.a.", "Meta Selic definida pelo Copom"),
    "selic_diaria":  (11,    "% a.d.", "Taxa Selic efetiva diária"),
    "cdi_anual":     (4389,  "% a.a.", "CDI anualizado, base 252"),
    "cambio_usd":    (1,     "R$/US$", "Dólar comercial, venda, fechamento"),
    "cambio_eur":    (21619, "R$/EUR", "Euro, venda"),
    "ipca_mes":      (433,   "% a.m.", "IPCA, variação mensal"),
    "ipca_12m":      (13522, "% a.a.", "IPCA acumulado em 12 meses"),
    "inpc_mes":      (188,   "% a.m.", "INPC, variação mensal"),
    "igpm_mes":      (189,   "% a.m.", "IGP-M, variação mensal"),
    "ibcbr":         (24363, "índice", "IBC-Br, atividade econômica"),
    "ibcbr_dessaz":  (24364, "índice", "IBC-Br dessazonalizado"),
    "pib_mensal":    (4380,  "R$ mi",  "PIB mensal, valores correntes"),
}


# --------------------------------------------------------------------------- #
# Acesso à rede
# --------------------------------------------------------------------------- #
def _baixar_json(url: str, timeout: int = 60, tentativas: int = 4):
    """GET com retentativa.

    A API do SGS é instável sob carga: devolve corpo vazio ou 5xx de forma
    intermitente. Sem retentativa, uma coleta de 12 séries falha quase sempre.
    """
    req = urllib.request.Request(url, headers=CABECALHO)
    for tentativa in range(tentativas):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                corpo = resp.read().decode("utf-8").strip()
            if corpo:
                return json.loads(corpo)
            erro: Exception = ValueError("resposta vazia")
        except urllib.error.HTTPError as e:
            if e.code == 404:  # janela sem observação: não adianta insistir
                raise
            erro = e
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            erro = e
        if tentativa == tentativas - 1:
            raise erro
        time.sleep(1.5 * (tentativa + 1))
    raise RuntimeError("inalcançável")


def fetch_sgs(codigo: int, inicio: date, fim: date) -> list[tuple[date, float]]:
    """Baixa uma série do SGS.

    A API rejeita janelas maiores que 10 anos em séries diárias, então a consulta
    é quebrada em blocos. Janela sem observação devolve HTTP 404, que é ignorado.
    """
    dados: list[tuple[date, float]] = []
    cursor = inicio
    while cursor <= fim:
        parada = min(date(cursor.year + 9, 12, 31), fim)
        url = (
            f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"
            f"?formato=json&dataInicial={cursor:%d/%m/%Y}&dataFinal={parada:%d/%m/%Y}"
        )
        try:
            for linha in _baixar_json(url):
                dados.append(
                    (datetime.strptime(linha["data"], "%d/%m/%Y").date(), float(linha["valor"]))
                )
        except urllib.error.HTTPError as erro:
            if erro.code != 404:
                raise
        cursor = date(parada.year + 1, 1, 1)
    dados.sort()
    return dados


def fetch_yahoo(ticker: str, inicio: date, fim: date) -> list[dict]:
    """Baixa candles diários do Yahoo Finance."""
    p1 = int(datetime(inicio.year, inicio.month, inicio.day, tzinfo=timezone.utc).timestamp())
    p2 = int(datetime(fim.year, fim.month, fim.day, tzinfo=timezone.utc).timestamp()) + 86400
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}"
        f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit"
    )
    resultado = _baixar_json(url)["chart"]["result"][0]
    offset = resultado["meta"].get("gmtoffset", 0)
    cotacoes = resultado["indicators"]["quote"][0]
    ajustado = resultado["indicators"].get("adjclose", [{}])[0].get("adjclose")

    linhas = []
    for i, epoch in enumerate(resultado["timestamp"]):
        fechamento = cotacoes["close"][i]
        if fechamento is None:  # pregão sem negócio no ativo
            continue
        linhas.append(
            {
                "data": datetime.fromtimestamp(epoch + offset, tz=timezone.utc).date(),
                "abertura": cotacoes["open"][i],
                "maxima": cotacoes["high"][i],
                "minima": cotacoes["low"][i],
                "fechamento": fechamento,
                "fechamento_ajustado": ajustado[i] if ajustado else fechamento,
                "volume": cotacoes["volume"][i],
            }
        )
    return linhas


# --------------------------------------------------------------------------- #
# Transformação
# --------------------------------------------------------------------------- #
def ultimo_do_mes(serie: list[tuple[date, float]]) -> dict[str, float]:
    """Colapsa uma série para frequência mensal, guardando a última observação.

    Séries mensais têm uma observação por mês e passam intactas; séries diárias
    viram o valor de fechamento do mês.
    """
    por_mes: dict[str, float] = {}
    for dia, valor in serie:  # já ordenada, então o último a escrever vence
        por_mes[f"{dia:%Y-%m}"] = valor
    return por_mes


def resumo_mensal_ativo(candles: list[dict]) -> dict[str, dict[str, float]]:
    """Fechamento do mês, retorno e volatilidade realizada anualizada."""
    meses: dict[str, list[dict]] = {}
    for linha in candles:
        meses.setdefault(f"{linha['data']:%Y-%m}", []).append(linha)

    saida: dict[str, dict[str, float]] = {}
    anterior = None
    for mes in sorted(meses):
        dias = meses[mes]
        fechamentos = [d["fechamento_ajustado"] for d in dias]
        retornos_diarios = [
            fechamentos[i] / fechamentos[i - 1] - 1 for i in range(1, len(fechamentos))
        ]
        registro = {
            "fechamento": round(fechamentos[-1], 4),
            "retorno_mensal": round(fechamentos[-1] / anterior - 1, 6) if anterior else "",
            "vol_anualizada": (
                round(statistics.stdev(retornos_diarios) * (252 ** 0.5), 6)
                if len(retornos_diarios) > 1
                else ""
            ),
            "pregoes": len(dias),
        }
        saida[mes] = registro
        anterior = fechamentos[-1]
    return saida


def gravar_csv(caminho: Path, colunas: list[str], linhas: list[dict]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as arq:
        escritor = csv.DictWriter(arq, fieldnames=colunas)
        escritor.writeheader()
        escritor.writerows(linhas)


# --------------------------------------------------------------------------- #
# Execução
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inicio", default="2010-01-01", help="data inicial (AAAA-MM-DD)")
    p.add_argument("--fim", default=date.today().isoformat(), help="data final (AAAA-MM-DD)")
    p.add_argument("--saida", default="dados", help="diretório de saída")
    p.add_argument(
        "--tickers",
        nargs="*",
        default=["^BVSP", "USDBRL=X", "PETR4.SA", "VALE3.SA", "ITUB4.SA"],
        help="tickers do Yahoo Finance (ações brasileiras terminam em .SA)",
    )
    p.add_argument("--so-sgs", action="store_true", help="pula a coleta de mercado")
    args = p.parse_args()

    inicio = date.fromisoformat(args.inicio)
    fim = date.fromisoformat(args.fim)
    saida = Path(args.saida)

    painel: dict[str, dict[str, object]] = {}
    colunas = ["mes"]

    print(f"SGS - {len(SERIES_SGS)} series, {inicio} a {fim}")
    for nome, (codigo, unidade, descricao) in SERIES_SGS.items():
        try:
            serie = fetch_sgs(codigo, inicio, fim)
        except Exception as erro:  # uma série fora do ar não derruba a coleta
            print(f"  !  {nome:<14} (codigo {codigo}) falhou: {erro}")
            continue
        gravar_csv(
            saida / f"sgs_{nome}.csv",
            ["data", "valor"],
            [{"data": d.isoformat(), "valor": v} for d, v in serie],
        )
        colunas.append(nome)
        for mes, valor in ultimo_do_mes(serie).items():
            painel.setdefault(mes, {})[nome] = valor
        faixa = f"{serie[0][0]} a {serie[-1][0]}" if serie else "vazia"
        print(f"  -  {nome:<14} {len(serie):>6} obs  {faixa:<25} [{unidade}] {descricao}")

    if not args.so_sgs:
        print(f"\nMercado — {len(args.tickers)} ativos")
        for ticker in args.tickers:
            try:
                candles = fetch_yahoo(ticker, inicio, fim)
            except Exception as erro:
                print(f"  !  {ticker:<12} falhou: {erro}")
                continue
            gravar_csv(
                saida / f"mercado_{ticker.replace('^', '').replace('=', '_')}.csv",
                ["data", "abertura", "maxima", "minima", "fechamento", "fechamento_ajustado", "volume"],
                [{**c, "data": c["data"].isoformat()} for c in candles],
            )
            base = ticker.replace("^", "").replace("=X", "").replace(".SA", "").lower()
            for sufixo in ("fechamento", "retorno_mensal", "vol_anualizada"):
                colunas.append(f"{base}_{sufixo}")
            for mes, registro in resumo_mensal_ativo(candles).items():
                alvo = painel.setdefault(mes, {})
                for sufixo in ("fechamento", "retorno_mensal", "vol_anualizada"):
                    alvo[f"{base}_{sufixo}"] = registro[sufixo]
            print(f"  -  {ticker:<12} {len(candles):>6} pregoes  {candles[0]['data']} a {candles[-1]['data']}")

    linhas = [{"mes": mes, **painel[mes]} for mes in sorted(painel)]
    gravar_csv(saida / "painel_mensal.csv", colunas, linhas)
    print(f"\npainel_mensal.csv: {len(linhas)} meses x {len(colunas)} colunas → {saida.resolve()}")
    print(
        "\nATENÇÃO ao montar X e y: toda coluna macro precisa entrar defasada.\n"
        "  O IPCA de julho só é publicado em agosto — usá-lo para explicar julho é vazamento.\n"
        "  df[preditoras] = df[preditoras].shift(1)  antes de treinar."
    )


if __name__ == "__main__":
    main()
