# O modelo final

O modelo escolhido é o vencedor da análise de desempenho: **codificação one-hot com o alvo em log**, resolvido pelas equações normais. R² de teste **0,9599**, RMSE **US$ 792,50**.

## Antes das equações: desfazendo a padronização

Os coeficientes da tabela de pesos (`carat` 5.150,26, etc.) estão em **escala padronizada** — valem "dólares por desvio padrão", não "dólares por quilate". Para escrever a equação usável é preciso convertê-los, e a conversão é apenas substituição algébrica.

O modelo ajustado é

$$\hat{y} = w_0 + \sum_j w_j z_j, \qquad \text{com} \qquad z_j = \frac{x_j - \mu_j}{\sigma_j}$$

Substituindo $z_j$ e distribuindo a soma:

$$\hat{y} = w_0 + \sum_j w_j \frac{x_j - \mu_j}{\sigma_j}
= w_0 + \sum_j \frac{w_j}{\sigma_j}x_j - \sum_j \frac{w_j \mu_j}{\sigma_j}$$

Agrupando os dois termos que não dependem de $x$, sobra uma equação linear nas unidades originais, $\hat{y} = a_0 + \sum_j a_j x_j$, com

$$\boxed{\;a_j = \frac{w_j}{\sigma_j} \qquad\qquad a_0 = w_0 - \sum_j \frac{w_j\,\mu_j}{\sigma_j}\;}$$

A leitura é intuitiva: dividir por $\sigma_j$ converte "por desvio padrão" em "por unidade do atributo", e o somatório subtraído do intercepto reabsorve as médias que a padronização havia removido.

Os $\mu_j$ e $\sigma_j$ usados são os do `Padronizador` — estimados **apenas no treino**, os mesmos aplicados ao teste. Tudo abaixo já está convertido, e a conversão foi conferida numericamente: as previsões da equação em unidades originais batem com as do modelo padronizado, com diferença máxima de **3,6 × 10⁻¹⁵**.

---

## A equação, em unidades originais

Como o treino foi em log, a previsão sai de uma exponencial — ou seja, o modelo é **multiplicativo** no preço:

$$\widehat{\text{preço}} = 0{,}037658 \times e^{\,-1{,}0323\,\text{carat}\;+\;0{,}03424\,\text{depth}\;+\;0{,}009686\,\text{table}\;+\;0{,}72964\,x\;+\;0{,}36790\,y\;+\;0{,}50131\,z} \times M_{\text{cut}} \times M_{\text{color}} \times M_{\text{clarity}}$$

onde 0,037658 = e^(−3,27921) é o intercepto, `carat` está em quilates, `depth` e `table` em porcentagem, `x`, `y` e `z` em milímetros, e os três *M* são os multiplicadores de qualidade da tabela abaixo.

## Os multiplicadores de qualidade

Cada nível multiplica o preço pelo fator indicado, **em relação ao pior nível** de cada atributo, que é a referência absorvida pelo intercepto.

| `cut` | × | | `color` | × | | `clarity` | × |
|---|---:|---|---|---:|---|---|---:|
| Fair | 1,000 | | J | 1,000 | | I1 | 1,000 |
| Good | 1,077 | | I | 1,148 | | SI2 | 1,509 |
| Very Good | 1,113 | | H | 1,295 | | SI1 | 1,781 |
| Premium | 1,125 | | G | 1,422 | | VS2 | 2,060 |
| Ideal | 1,158 | | F | 1,520 | | VS1 | 2,205 |
| | | | E | 1,580 | | VVS2 | 2,510 |
| | | | D | 1,671 | | VVS1 | 2,698 |
| | | | | | | IF | 2,952 |

> **O achado que vale citar:** a codificação one-hot **joga fora a ordem** dos níveis — para ela, `IF` e `I1` são apenas duas colunas binárias sem relação. Mesmo assim, os multiplicadores saíram **perfeitamente monotônicos** nos três atributos. O modelo recuperou sozinho a escala de qualidade GIA a partir dos dados. É a validação mais forte de que a ordem que declaramos em `src/dados.ORDEM` corresponde à realidade do mercado.
>
> Lendo os números: sair de `I1` para `IF` quase **triplica** o preço (×2,95). Sair de `J` para `D` acrescenta 67%. E a lapidação, de `Fair` para `Ideal`, só 16% — o mercado paga muito mais por pureza do que por lapidação.

## Exemplo numérico

Primeira pedra do conjunto de teste: 1,01 ct · Good · G · VS2 · depth 63,6 · table 60,0 · 6,30 × 6,35 × 4,02 mm.

$$\widehat{\text{preço}} = e^{8{,}5350} = \textbf{US\$ 5.089{,}88} \qquad \text{(preço real: US\$ 5.599,00)}$$

---

## A versão simples: ordinal com alvo em dólares

Se preferir uma equação de nove termos, aditiva e direta em dólares — é a configuração usada na análise dos pesos, com R² de teste 0,9100:

$$\widehat{\text{preço}} = -7.004{,}79 + 10.840{,}08\,\text{carat} + 130{,}21\,\text{cut} + 322{,}37\,\text{color} + 495{,}56\,\text{clarity}$$
$$+\; 87{,}89\,\text{depth} - 17{,}87\,\text{table} - 1.790{,}40\,x + 2.625{,}77\,y - 2.795{,}04\,z$$

Aqui `cut`, `color` e `clarity` entram como inteiros na ordem de qualidade, começando em 0 para o pior nível. A leitura é direta: **cada quilate a mais vale US$ 10.840**, e **cada degrau de pureza vale US$ 496**.

---

## Duas ressalvas honestas

**Os coeficientes de `x`, `y` e `z` não são interpretáveis isoladamente.** Aparecem com sinais alternados nas duas equações — e no modelo em log o próprio `carat` sai negativo (−1,03). É a colinearidade descrita na análise dos pesos: as quatro colunas de tamanho medem a mesma coisa e o mínimo quadrado reparte o crédito entre elas de forma arbitrária. O modelo prevê corretamente porque usa a **combinação** delas; ler qualquer uma sozinha leva a conclusões absurdas.

**A volta de log para dólares é enviesada.** Por Jensen, exp da previsão em log estima a **mediana** condicional, não a média — as previsões do modelo vencedor são levemente subestimadas. Nenhuma correção (fator de Duan ou similar) foi aplicada, e o mesmo critério vale para as quatro configurações, então a comparação entre elas permanece justa.
