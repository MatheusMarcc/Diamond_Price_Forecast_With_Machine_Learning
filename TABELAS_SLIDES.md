# Tabelas para os slides

Base `diamonds` (OpenML 42225) · 53.917 instâncias · divisão 43.134 treino / 10.783 teste, semente 42.
Quando o treino é em log, a previsão volta para dólares antes da medição — todas as métricas, de treino e de teste, estão na mesma unidade.

---

## Slide — Desempenho das quatro configurações

Os dois algoritmos produzem métricas idênticas, então cada configuração ocupa **uma** linha, com o tempo de cada algoritmo em colunas separadas.

| Codificação | Alvo | R² treino | R² teste | MSE treino (USD²) | MSE teste (USD²) | RMSE teste (USD) | Eq. normais | Gradient Descent |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ordinal | USD | 0,9075 | 0,9100 | 1.475.327 | 1.408.606 | 1.186,85 | 5,1 ms | 25,8 s · 50.000 ép. |
| ordinal | log | 0,9496 | 0,9524 | 803.543 | 745.544 | 863,45 | 2,9 ms | 18,1 s · 31.191 ép. |
| one-hot | USD | 0,9202 | 0,9218 | 1.272.976 | 1.223.692 | 1.106,21 | 8,1 ms | 31,4 s · 50.000 ép. |
| **one-hot** | **log** | **0,9587** | **0,9599** | **659.053** | **628.052** | **792,50** | 6,9 ms | 25,1 s · 38.845 ép. |

> **Melhor configuração:** one-hot com alvo em log — erro típico de US$ 792,50, contra US$ 1.186,85 da pior. A escala do alvo pesa de 3 a 5 vezes mais que a codificação: o ganho veio do pré-processamento, não do otimizador.

---

## Slide — Treino contra teste: não há sobreajuste

| Configuração | MSE treino | MSE teste | teste ÷ treino |
|---|---:|---:|---:|
| ordinal · USD | 1.475.327 | 1.408.606 | **0,955** |
| ordinal · log | 803.543 | 745.544 | **0,928** |
| one-hot · USD | 1.272.976 | 1.223.692 | **0,961** |
| one-hot · log | 659.053 | 628.052 | **0,953** |

> Nas quatro configurações o erro de **teste é menor que o de treino** — razão entre 0,93 e 0,96. Sobreajuste produziria o contrário, com a razão acima de 1.
>
> O resultado é esperado: são 43.134 observações de treino para no máximo 24 parâmetros, ou **1.797 observações por parâmetro**. Um modelo linear não tem graus de liberdade para memorizar a amostra. A partição de teste apenas calhou de ser marginalmente mais fácil.

> **Amarrando com a convergência:** o MSE de treino de ordinal/USD é 1.475.327,47 — exatamente o valor final de J(w) na figura de convergência do Gradient Descent. Não é coincidência: a função de custo que o algoritmo minimiza **é** o MSE de treino.

---

## Slide — Custo computacional

| | Equações normais | Gradient Descent |
|---|---|---|
| Tempo por ajuste | **2,9 a 8,1 milissegundos** | 18,1 a 31,4 segundos |
| Iterativo | não — um passo direto | sim — dezenas de milhares de épocas |
| Hiperparâmetros | nenhum | taxa, épocas, tolerância |
| Precisa de padronização | não | sim |

> Mesma resposta, custo separado por **três a quatro ordens de grandeza**. Com 9 ou 23 atributos, a forma fechada é a escolha certa. O Gradient Descent é o algoritmo que continuaria viável se montar AᵀA ficasse caro — esta base não chega lá.

---

## Slide — Análise de performance: o ganho de cada decisão

| Mudança | R² teste | Δ R² | MSE treino | Δ MSE treino | MSE teste | Δ MSE teste |
|---|---|---:|---|---:|---|---:|
| ordinal → one-hot, alvo em USD | 0,9100 → 0,9218 | +0,0118 | 1.475.327 → 1.272.976 | −13,7% | 1.408.606 → 1.223.692 | −13,1% |
| ordinal → one-hot, alvo em log | 0,9524 → 0,9599 | +0,0075 | 803.543 → 659.053 | −18,0% | 745.544 → 628.052 | −15,8% |
| USD → log, codificação ordinal | 0,9100 → 0,9524 | +0,0424 | 1.475.327 → 803.543 | −45,5% | 1.408.606 → 745.544 | −47,1% |
| USD → log, codificação one-hot | 0,9218 → 0,9599 | +0,0380 | 1.272.976 → 659.053 | −48,2% | 1.223.692 → 628.052 | −48,7% |
| **combinadas** (ordinal/USD → one-hot/log) | 0,9100 → 0,9599 | **+0,0499** | 1.475.327 → 659.053 | **−55,3%** | 1.408.606 → 628.052 | **−55,4%** |

> **Os cortes de treino e de teste andam juntos** — −55,3% contra −55,4% no total. Isso reforça a leitura de que o ganho é real e não memorização da amostra de treino.
>
> **Cada correção corta uma fatia maior quando a outra já foi aplicada** — o one-hot passa de −13,1% para −15,8% no teste, a escala de −47,1% para −48,7%. Se as duas atacassem o mesmo problema, esses números cairiam. Eles sobem: os efeitos são praticamente independentes.

---

## Slide — Conferência: os dois algoritmos chegam no mesmo ponto

Configuração ordinal/USD, taxa 0,2, 50.000 épocas. Pesos padronizados, em dólares por desvio padrão.

| Atributo | Equações normais | Gradient Descent | Diferença |
|---|---:|---:|---:|
| intercepto | 3939,931794 | 3939,931794 | 9,1 × 10⁻¹³ |
| carat | 5150,264876 | 5150,264876 | −1,8 × 10⁻⁹ |
| cut | 145,340925 | 145,340925 | −4,7 × 10⁻¹⁰ |
| color | 548,880023 | 548,880023 | 9,8 × 10⁻¹² |
| clarity | 816,242238 | 816,242238 | 4,4 × 10⁻¹⁰ |
| depth | 125,623079 | 125,623079 | −9,3 × 10⁻¹⁰ |
| table | −39,934129 | −39,934129 | −7,6 × 10⁻¹⁰ |
| x | −2007,214651 | −2007,214650 | 7,3 × 10⁻⁸ |
| y | 2922,116199 | 2922,116199 | −7,5 × 10⁻⁸ |
| z | −1934,830654 | −1934,830654 | 4,2 × 10⁻⁹ |

> **Maior diferença absoluta: 7,5 × 10⁻⁸.** Os pesos coincidem em 6 casas decimais garantidas, ou de 10 a 15 algarismos significativos — mas não são bit a bit idênticos, e a diferença varia de atributo para atributo. É esse resíduo que comprova que duas computações distintas rodaram. Diferença exatamente zero é que seria suspeito.

*Gerado por `conferencia_gd.py`. A coluna de diferença varia nos últimos dígitos conforme a máquina.*

---

## Versão completa — as oito execuções

Para o relatório ou slide reserva.

| Codificação | Alvo | Algoritmo | R² treino | R² teste | MSE treino (USD²) | MSE teste (USD²) | RMSE teste (USD) | Tempo | Épocas |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ordinal | USD | equações normais | 0,9075 | 0,9100 | 1.475.327,47 | 1.408.606,43 | 1.186,85 | 5,1 ms | — |
| ordinal | USD | gradient descent | 0,9075 | 0,9100 | 1.475.327,47 | 1.408.606,43 | 1.186,85 | 25,83 s | 50.000 |
| ordinal | log | equações normais | 0,9496 | 0,9524 | 803.542,74 | 745.543,78 | 863,45 | 2,9 ms | — |
| ordinal | log | gradient descent | 0,9496 | 0,9524 | 803.542,74 | 745.543,78 | 863,45 | 18,10 s | 31.191 |
| one-hot | USD | equações normais | 0,9202 | 0,9218 | 1.272.976,28 | 1.223.692,48 | 1.106,21 | 8,1 ms | — |
| one-hot | USD | gradient descent | 0,9202 | 0,9218 | 1.272.976,28 | 1.223.692,48 | 1.106,21 | 31,44 s | 50.000 |
| one-hot | log | equações normais | 0,9587 | 0,9599 | 659.053,49 | 628.051,98 | 792,50 | 6,9 ms | — |
| one-hot | log | gradient descent | 0,9587 | 0,9599 | 659.053,49 | 628.051,97 | 792,50 | 25,09 s | 38.845 |

**Sobre os tempos:** dependem da máquina e mudam a cada execução. O que se sustenta em qualquer ambiente é a ordem de grandeza — milissegundos contra dezenas de segundos.
