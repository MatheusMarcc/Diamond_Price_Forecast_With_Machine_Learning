# Tabelas para os slides

Base `diamonds` (OpenML 42225) · 53.917 instâncias · divisão 43.134 treino / 10.783 teste, semente 42.
Quando o treino é em log, a previsão volta para dólares antes da medição — todas as métricas estão na mesma unidade.

---

## Slide — Desempenho das quatro configurações

Os dois algoritmos produzem métricas idênticas, então cada configuração ocupa **uma** linha, com o tempo de cada algoritmo em colunas separadas.

| Codificação | Alvo | R² teste | MSE teste (USD²) | RMSE teste (USD) | Eq. normais | Gradient Descent |
|---|---|---:|---:|---:|---:|---:|
| ordinal | USD | 0,9100 | 1.408.606 | 1.186,85 | 5,1 ms | 25,8 s · 50.000 ép. |
| ordinal | log | 0,9524 | 745.544 | 863,45 | 2,9 ms | 18,1 s · 31.191 ép. |
| one-hot | USD | 0,9218 | 1.223.692 | 1.106,21 | 8,1 ms | 31,4 s · 50.000 ép. |
| **one-hot** | **log** | **0,9599** | **628.052** | **792,50** | 6,9 ms | 25,1 s · 38.845 ép. |

> **Melhor configuração:** one-hot com alvo em log — R² de teste 0,9599 e erro típico de US$ 792,50, contra US$ 1.186,85 da pior. A escala do alvo pesa de 3 a 5 vezes mais que a codificação: o ganho veio do pré-processamento, não do otimizador.

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

> **Maior diferença absoluta: 7,5 × 10⁻⁸.** Os pesos coincidem até a sexta casa decimal — mas não são bit a bit idênticos, e a diferença varia de atributo para atributo. É esse resíduo que comprova que duas computações genuinamente distintas rodaram. Diferença exatamente zero é que seria suspeito.

*Gerado por `conferencia_gd.py`. A coluna de diferença varia nos últimos dígitos conforme a máquina.*

---

## Versão completa — as oito execuções

Para o relatório ou slide reserva.

| Codificação | Alvo | Algoritmo | R² treino | R² teste | MSE teste (USD²) | RMSE teste (USD) | Tempo | Épocas |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ordinal | USD | equações normais | 0,9075 | 0,9100 | 1.408.606,43 | 1.186,85 | 5,1 ms | — |
| ordinal | USD | gradient descent | 0,9075 | 0,9100 | 1.408.606,43 | 1.186,85 | 25,83 s | 50.000 |
| ordinal | log | equações normais | 0,9496 | 0,9524 | 745.543,78 | 863,45 | 2,9 ms | — |
| ordinal | log | gradient descent | 0,9496 | 0,9524 | 745.543,78 | 863,45 | 18,10 s | 31.191 |
| one-hot | USD | equações normais | 0,9202 | 0,9218 | 1.223.692,48 | 1.106,21 | 8,1 ms | — |
| one-hot | USD | gradient descent | 0,9202 | 0,9218 | 1.223.692,48 | 1.106,21 | 31,44 s | 50.000 |
| one-hot | log | equações normais | 0,9587 | 0,9599 | 628.051,98 | 792,50 | 6,9 ms | — |
| one-hot | log | gradient descent | 0,9587 | 0,9599 | 628.051,97 | 792,50 | 25,09 s | 38.845 |

**Ausência de sobreajuste:** nas quatro configurações o R² de teste é *maior* que o de treino, com diferença máxima de 0,0027 — são 43.134 observações de treino para no máximo 24 parâmetros, ou 1.797 observações por parâmetro.

**Sobre os tempos:** dependem da máquina e mudam a cada execução. O que se sustenta em qualquer ambiente é a ordem de grandeza — milissegundos contra dezenas de segundos.
