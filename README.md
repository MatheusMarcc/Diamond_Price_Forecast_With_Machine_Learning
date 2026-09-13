# Projeto 1 — Regressão

Predição do preço de diamantes com Regressão Linear, implementada do zero por
**equações normais** e por **Gradient Descent**. Nenhuma biblioteca de machine
learning é usada nos modelos — o scikit-learn entra apenas no t-SNE.

**Base**: diamonds — 53.940 pedras, 9 preditoras, alvo `price` em dólares.
Fonte: OpenML dataset 42225, baixado em ARFF e convertido para CSV por
`arff_para_csv.py`. Proveniência (URL, data e SHA-256) em
`dados/diamonds_metadados.json`.

## Como rodar

```
python -m pip install -r requirements.txt
python arff_para_csv.py     # só na primeira vez: baixa e converte a base
python 01_estudo_base.py    # figuras 01 a 05
python 02_treino.py         # tabela de métricas  (~2,5 min)
python 03_analises.py       # figuras 06 a 09     (~1,5 min)
```

## Estrutura

```
dados/       ARFF original, CSV convertido e metadados de proveniência
src/         módulos: dados, modelos, metricas
figuras/     as 9 figuras do relatório
resultados/  tabelas em csv e resumo.json com todos os números
```

| arquivo | o que faz |
|---|---|
| `arff_para_csv.py` | baixa o ARFF do OpenML, valida e converte para CSV |
| `src/dados.py` | carga, limpeza, codificação, divisão treino/teste, padronização |
| `src/modelos.py` | os dois algoritmos, mais VIF, bootstrap e espectro da hessiana |
| `src/metricas.py` | MSE, RMSE e R² |
| `01_estudo_base.py` | correlação, desbalanceamento e t-SNE |
| `02_treino.py` | 4 configurações × 2 algoritmos, com R² e MSE |
| `03_analises.py` | pesos, VIF, estabilidade e convergência do GD |

## Resultados

| codificação | alvo | R² teste | RMSE teste |
|---|---|---|---|
| ordinal | USD | 0,9100 | US$ 1.186,85 |
| ordinal | log | 0,9524 | US$ 863,45 |
| one-hot | USD | 0,9218 | US$ 1.106,21 |
| **one-hot** | **log** | **0,9599** | **US$ 792,50** |

Equações normais e Gradient Descent chegam ao mesmo vetor de pesos (diferença
máxima de 7,2 × 10⁻⁸), com custo computacional muito diferente: milissegundos
contra dezenas de segundos.

## Decisões registradas

- **Limpeza**: removidas 23 linhas com `x`, `y` ou `z` iguais a zero ou acima de
  20 mm (dimensões fisicamente impossíveis). Restam 53.917.
- **Divisão**: 80/20 aleatória com semente fixa. A base não é temporal, então
  embaralhar é apropriado. Média e desvio da padronização vêm só do treino.
- **Ordem das categóricas**: o ARFF declara os níveis em ordem alfabética, que
  não é a ordem de qualidade. A ordem correta está em `src/dados.ORDEM`.
- **Alvo em log**: as previsões voltam para dólares com `exp` antes de calcular
  as métricas, senão as linhas da tabela não seriam comparáveis.
- **Colinearidade**: `x`, `y` e `z` têm VIF acima de 480, e `x` e `z` recebem
  peso negativo. O diagnóstico está em `03_analises.py` (VIF e bootstrap).
- **Convergência**: o custo entra em 1% do valor final na época 331, mas os
  parâmetros só estabilizam por volta da época 45.776 — o número de condição da
  hessiana é 3.580.
