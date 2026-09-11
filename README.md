# Projeto 1 — Regressão

Predição do preço de diamantes com Regressão Linear, implementada por
equações normais e por Gradient Descent.

**Base**: diamonds — 53.940 pedras, 9 preditoras, alvo `price` em dólares.
Fonte: OpenML dataset 42225, baixado em ARFF e convertido para CSV por
`arff_para_csv.py`. Proveniência (URL, data e SHA-256) em
`dados/diamonds_metadados.json`.

## Estrutura

```
dados/       ARFF original, CSV convertido e metadados
src/         módulos: dados, modelos, metricas
figuras/     saída dos scripts (png)
resultados/  tabelas de métricas e estatísticas (csv)
```

| arquivo | o que faz |
|---|---|
| `arff_para_csv.py` | baixa o ARFF do OpenML, valida e converte para CSV |
| `src/dados.py` | carga, limpeza, codificação, divisão treino/teste, padronização |
| `src/modelos.py` | Regressão Linear por equações normais e por Gradient Descent, mais VIF e bootstrap |
| `src/metricas.py` | MSE, RMSE e R² |
| `01_estudo_base.py` | correlação, desbalanceamento e t-SNE |
| `02_treino.py` | comparação entre codificações, escalas do alvo e os dois algoritmos |
| `03_analises.py` | pesos, VIF, estabilidade por bootstrap e convergência do GD |

## Como rodar

```
python -m pip install -r requirements.txt
python arff_para_csv.py     # só na primeira vez
python 01_estudo_base.py
python 02_treino.py
python 03_analises.py
```

## Decisões registradas

- **Limpeza**: removidas 23 linhas com `x`, `y` ou `z` iguais a zero ou acima de
  20 mm (dimensões fisicamente impossíveis). Restam 53.917.
- **Divisão**: 80/20 aleatória com semente fixa. A base não é temporal, então
  embaralhar é apropriado. Média e desvio da padronização vêm só do treino.
- **Ordem das categóricas**: o ARFF declara os níveis em ordem alfabética, que
  não é a ordem de qualidade. A ordem correta está em `src/dados.ORDEM`.
- **Intercepto**: estimado junto, via coluna de uns.
- **Colinearidade**: sem penalização, é diagnosticada por VIF e por
  reamostragem bootstrap dos coeficientes. VIF de `x`, `y` e `z` passa de 480.
