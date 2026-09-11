#!/usr/bin/env python3
"""Baixa o dataset diamonds do OpenML em ARFF e converte para CSV.

Fonte oficial: OpenML dataset 42225 (https://www.openml.org/d/42225).
Usa apenas a biblioteca padrão — nenhuma dependência externa.

O parser de ARFF é escrito aqui de propósito: o formato é simples e ter o
código à vista permite descrever a conversão no relatório.

Exemplos:
    python arff_para_csv.py
    python arff_para_csv.py --sep ";"
    python arff_para_csv.py --arff dados/diamonds.arff --saida dados/diamonds.txt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URL_ARFF = "https://openml.org/data/v1/download/21792853/diamonds.arff"
CABECALHO = {"User-Agent": "Mozilla/5.0 (projeto academico; conversao arff)"}

TIPOS_NUMERICOS = {"NUMERIC", "REAL", "INTEGER"}


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def baixar(url: str, destino: Path, tentativas: int = 3) -> Path:
    """Baixa o arquivo, a menos que ele já exista."""
    if destino.exists() and destino.stat().st_size > 0:
        print(f"ARFF ja existe, mantendo: {destino}  ({destino.stat().st_size:,} bytes)")
        return destino

    destino.parent.mkdir(parents=True, exist_ok=True)
    for tentativa in range(tentativas):
        try:
            req = urllib.request.Request(url, headers=CABECALHO)
            with urllib.request.urlopen(req, timeout=120) as resp:
                conteudo = resp.read()
            if not conteudo:
                raise ValueError("resposta vazia")
            destino.write_bytes(conteudo)
            print(f"baixado: {destino}  ({len(conteudo):,} bytes)")
            return destino
        except (urllib.error.URLError, ValueError, TimeoutError) as erro:
            if tentativa == tentativas - 1:
                raise
            print(f"  falhou ({erro}), tentando de novo...")
            time.sleep(2 * (tentativa + 1))
    raise RuntimeError("inalcancavel")


# --------------------------------------------------------------------------- #
# Parser de ARFF
# --------------------------------------------------------------------------- #
def _sem_aspas(texto: str) -> str:
    texto = texto.strip()
    if len(texto) >= 2 and texto[0] == texto[-1] and texto[0] in "'\"":
        return texto[1:-1]
    return texto


def _parse_atributo(linha: str) -> dict:
    """Interpreta uma linha @attribute.

    Duas formas possíveis:
        @ATTRIBUTE carat REAL
        @ATTRIBUTE cut {Fair, Good, Ideal, Premium, 'Very Good'}
    """
    corpo = linha.split(None, 1)[1].strip()  # remove o "@attribute"

    if corpo.startswith("'") or corpo.startswith('"'):  # nome entre aspas
        aspa = corpo[0]
        fim = corpo.index(aspa, 1)
        nome, resto = corpo[1:fim], corpo[fim + 1 :].strip()
    else:
        nome, _, resto = corpo.partition(" ")
        resto = resto.strip()

    if resto.startswith("{"):
        if not resto.endswith("}"):
            raise ValueError(f"conjunto nominal nao fechado em: {linha!r}")
        niveis = [_sem_aspas(v) for v in next(csv.reader([resto[1:-1]], quotechar="'", skipinitialspace=True))]
        return {"nome": nome, "tipo": "nominal", "niveis": niveis}

    return {"nome": nome, "tipo": resto.split()[0].upper(), "niveis": None}


def ler_arff(caminho: Path) -> tuple[list[dict], list[list[str]], str]:
    """Devolve (atributos, linhas de dados, nome da relacao).

    Os valores são mantidos como texto original. Converter para float e
    escrever de volta pode alterar a formatação dos números; preservar a
    string garante que o CSV reproduza o ARFF caractere por caractere.
    """
    atributos: list[dict] = []
    relacao = ""
    dados_brutos: list[str] = []
    dentro_dos_dados = False

    with caminho.open(encoding="utf-8", errors="replace") as arq:
        for numero, linha in enumerate(arq, 1):
            bruta = linha.rstrip("\n\r")
            nua = bruta.strip()

            if dentro_dos_dados:
                if nua and not nua.startswith("%"):
                    if nua.startswith("{"):
                        raise NotImplementedError(
                            f"linha {numero}: ARFF esparso nao e suportado por este parser"
                        )
                    dados_brutos.append(nua)
                continue

            if not nua or nua.startswith("%"):
                continue

            chave = nua.split(None, 1)[0].lower()
            if chave == "@relation":
                relacao = _sem_aspas(nua.split(None, 1)[1]) if " " in nua else ""
            elif chave == "@attribute":
                atributos.append(_parse_atributo(nua))
            elif chave == "@data":
                dentro_dos_dados = True

    if not atributos:
        raise ValueError("nenhum @attribute encontrado — o arquivo e mesmo ARFF?")
    if not dentro_dos_dados:
        raise ValueError("marcador @data nao encontrado")

    leitor = csv.reader(io.StringIO("\n".join(dados_brutos)), quotechar="'", skipinitialspace=True)
    linhas = [[c.strip() for c in linha] for linha in leitor]
    return atributos, linhas, relacao


def validar(atributos: list[dict], linhas: list[list[str]]) -> dict:
    """Confere largura, tipos e níveis. Devolve contagens por coluna."""
    n = len(atributos)
    resumo = {a["nome"]: {"faltantes": 0, "distintos": set()} for a in atributos}

    for i, linha in enumerate(linhas):
        if len(linha) != n:
            raise ValueError(f"linha {i + 1} tem {len(linha)} campos, esperado {n}")
        for atributo, valor in zip(atributos, linha):
            info = resumo[atributo["nome"]]
            if valor in ("?", ""):  # marcador de faltante do ARFF
                info["faltantes"] += 1
                continue
            if atributo["tipo"] in TIPOS_NUMERICOS:
                try:
                    float(valor)
                except ValueError as erro:
                    raise ValueError(
                        f"linha {i + 1}, coluna {atributo['nome']}: {valor!r} nao e numero"
                    ) from erro
            elif atributo["tipo"] == "nominal" and valor not in atributo["niveis"]:
                raise ValueError(
                    f"linha {i + 1}, coluna {atributo['nome']}: {valor!r} "
                    f"fora dos niveis declarados {atributo['niveis']}"
                )
            info["distintos"].add(valor)

    return {k: {"faltantes": v["faltantes"], "distintos": len(v["distintos"])} for k, v in resumo.items()}


# --------------------------------------------------------------------------- #
# Saída
# --------------------------------------------------------------------------- #
def escrever(caminho: Path, atributos: list[dict], linhas: list[list[str]], sep: str) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as arq:
        escritor = csv.writer(arq, delimiter=sep, lineterminator="\n")
        escritor.writerow([a["nome"] for a in atributos])
        escritor.writerows(linhas)


def sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as arq:
        for bloco in iter(lambda: arq.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=URL_ARFF, help="URL do ARFF")
    p.add_argument("--arff", default="dados/diamonds.arff", help="onde salvar/ler o ARFF")
    p.add_argument("--saida", default="dados/diamonds.csv", help="arquivo de saida")
    p.add_argument("--sep", default=",", help="separador da saida (use '\\t' para TSV)")
    args = p.parse_args()

    sep = "\t" if args.sep in ("\\t", "tab") else args.sep
    arff = baixar(args.url, Path(args.arff))

    atributos, linhas, relacao = ler_arff(arff)
    print(f"\nrelacao: {relacao}")
    print(f"{len(linhas):,} linhas x {len(atributos)} atributos\n")

    resumo = validar(atributos, linhas)
    for a in atributos:
        info = resumo[a["nome"]]
        tipo = a["tipo"] if a["tipo"] != "nominal" else f"nominal{a['niveis']}"
        print(f"  {a['nome']:10} {tipo:<70.70} distintos={info['distintos']:>5} faltantes={info['faltantes']}")

    saida = Path(args.saida)
    escrever(saida, atributos, linhas, sep)

    metadados = {
        "fonte": args.url,
        "openml_id": 42225,
        "relacao": relacao,
        "extraido_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "linhas": len(linhas),
        "atributos": atributos,
        "sha256_arff": sha256(arff),
        "sha256_csv": sha256(saida),
    }
    caminho_meta = saida.with_name(saida.stem + "_metadados.json")
    caminho_meta.write_text(json.dumps(metadados, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nCSV        : {saida}  ({saida.stat().st_size:,} bytes)")
    print(f"metadados  : {caminho_meta}")
    print(f"sha256 CSV : {metadados['sha256_csv']}")
    print(
        "\nATENCAO: a ordem dos niveis no ARFF e alfabetica, nao a ordem de qualidade.\n"
        "  cut     : Fair < Good < Very Good < Premium < Ideal\n"
        "  color   : J (pior) < I < H < G < F < E < D (melhor)\n"
        "  clarity : I1 < SI2 < SI1 < VS2 < VS1 < VVS2 < VVS1 < IF\n"
        "  Codificar na ordem do arquivo inverte o sinal dos pesos."
    )


if __name__ == "__main__":
    main()
