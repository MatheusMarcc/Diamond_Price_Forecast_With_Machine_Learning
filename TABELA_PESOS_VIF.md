# Análise dos pesos — tabelas para os slides

Configuração **ordinal com alvo em USD**, solução por equações normais.
Atributos padronizados (z = (x − μ)/σ, com μ e σ estimados só no treino), então cada peso está em **dólares por desvio padrão do atributo** — é isso que permite comparar quilates, milímetros e níveis de qualidade na mesma escala. Intercepto descartado.

---

## Slide — Os pesos

| Atributo | Coeficiente (US$ / σ) | VIF | Desvio bootstrap | Coef. / desvio |
|---|---:|---:|---:|---:|
| carat | 5.150,26 | 24,87 | 112,01 | 46,0 |
| y | 2.922,12 | 539,53 | 427,87 | 6,8 |
| x | **−2.007,21** | 537,19 | 396,70 | −5,1 |
| z | **−1.934,83** | 489,70 | 761,97 | −2,5 |
| clarity | 816,24 | 1,24 | 8,82 | 92,5 |
| color | 548,88 | 1,12 | 7,44 | 73,8 |
| cut | 145,34 | 1,50 | 7,43 | 19,5 |
| depth | 125,62 | 8,87 | 93,05 | 1,4 |
| table | −39,93 | 1,63 | 8,22 | −4,9 |

> **Coerente:** `carat` domina, e as três qualidades saem positivas e na ordem que um catálogo de joalheria prevê — `clarity` > `color` > `cut`.
>
> **Fisicamente absurdo:** `x` e `z` saem **negativos**, apesar de correlacionarem +0,89 e +0,88 com o preço. Não é erro de sinal — é colinearidade.

---

## Slide — O que o VIF está medindo

$$\mathrm{VIF}_j = \frac{1}{1 - R^2_j}$$

onde $R^2_j$ vem de regredir o atributo *j* contra **todos os outros atributos** — o preço não entra na conta. A pergunta que ele responde: *quanto desta coluna já está contido nas demais?*

| Atributo | VIF | R²_j | √VIF | Leitura |
|---|---:|---:|---:|---|
| y | 539,53 | 0,9981 | 23,2 | 99,8% da largura é previsível pelas outras colunas |
| x | 537,19 | 0,9981 | 23,2 | idem para o comprimento |
| z | 489,70 | 0,9980 | 22,1 | idem para a altura |
| carat | 24,87 | 0,9598 | 5,0 | **redundante**: 96% previsível — massa ≈ densidade × volume ≈ x·y·z |
| *— limiar de alerta: VIF 10 —* | | | | |
| depth | 8,87 | 0,8873 | 3,0 | 88,7% previsível — é a razão 2z/(x+y) × 100 |
| table | 1,63 | 0,386 | 1,3 | independente — só 39% previsível |
| cut | 1,50 | 0,333 | 1,2 | independente |
| clarity | 1,24 | 0,194 | 1,1 | independente |
| color | 1,12 | 0,107 | 1,1 | praticamente ortogonal ao resto |

**Limiar de alerta: VIF = 10**, que corresponde a R² = 0,90 — noventa por cento da coluna redundante.

**De onde vem o nome:** a variância do coeficiente é multiplicada pelo VIF, então o erro-padrão é multiplicado por **√VIF**. Com VIF 537, o erro-padrão de `x` está inflado **23 vezes**. É esse fator que abre espaço para um coeficiente trocar de sinal sem que o ajuste perca qualidade.

**Cuidado — VIF alto não condena o coeficiente sozinho.** `carat` tem VIF 24,87 e mesmo assim é legível, porque o efeito dele (5.150 US$/σ) é grande o bastante para sobreviver à inflação: fica a 46 desvios de zero. Já `depth` tem VIF 8,87, *abaixo* do limiar, e é o coeficiente mais frágil da tabela, a 1,4 desvios. Quem decide é a razão coeficiente/desvio, não o VIF isolado.

---

## Slide — A conclusão

| | Atributos | Por quê |
|---|---|---|
| **Interpretáveis** | `clarity`, `color`, `cut`, e `carat` com ressalva | VIF entre 1,1 e 1,5; coeficiente a 19,5–92,5 desvios de zero |
| **Não interpretáveis isoladamente** | `x`, `y`, `z`, `depth` | VIF de 8,9 a 539,5; `z` a apenas 2,5 desvios, `depth` a 1,4 |

Somando os quatro atributos de tamanho:

$$5.150{,}26 + 2.922{,}12 - 2.007{,}21 - 1.934{,}83 = \mathbf{4.130{,}34}\ \text{US\$ por desvio padrão}$$

> **O total tem significado; a divisão entre as parcelas não.** Um diamante maior é maior em todas as medidas ao mesmo tempo, e é assim que as quatro colunas se movem na realidade. O mínimo quadrado não tem como repartir o crédito do "tamanho" entre elas, e resolve o empate com pesos grandes de sinais opostos que se cancelam.
>
> **Isso não degrada a predição** — o R² de 0,91 desta configuração continua valendo, porque o modelo usa a *combinação* das colunas, e ela é estável. Degrada a *interpretação* atributo por atributo. Predizer bem e explicar bem são objetivos diferentes, e falham de formas diferentes.

---

## Curiosidade que rende ponto

Se `depth` é função exata de `x`, `y` e `z` — a §3.1 mostra que a reconstrução bate com erro mediano de 0,027 ponto percentual —, por que o VIF dele é 8,87 e não infinito?

Porque `depth` é uma **razão**, não uma combinação linear, e o VIF mede previsibilidade **linear**. Uma função não linear de colunas presentes deixa resíduo linear, e por isso o R² dá 0,887 em vez de 1.
