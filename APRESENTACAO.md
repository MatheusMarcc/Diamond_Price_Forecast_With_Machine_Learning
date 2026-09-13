# Roteiro da apresentação (10 minutos)

Divisão de tempo: abertura 0,5 min + Experiência 2,5 min + Tarefa 2,0 min + Desempenho 2,0 min + Análise 2,5 min + fechamento 0,5 min = **10,0 min**. São 15 slides; a média fica em 40 segundos por slide, então cada bullet é uma frase falada, não um parágrafo lido.

---

## Abertura — 0,5 min

### Slide 1 — Prever o preço de um diamante: E, T e P

- Mitchell define um problema de aprendizado por três componentes: **Experiência (E)**, **Tarefa (T)** e **Medida de desempenho (P)**. A apresentação segue exatamente essa divisão, com uma quarta parte de análise.
- E = 53.917 diamantes com 9 atributos estruturados; T = prever `price` em dólares; P = $R^2$ e MSE no conjunto de teste.
- Dois algoritmos implementados do zero: equações normais e Gradient Descent.

Figura: nenhuma — só o título e os quatro blocos (E, T, P, Análise).

*O que falar:* dizer que a estrutura da apresentação não é arbitrária, é a formalização de Mitchell de 1997, e que a quarta parte existe porque o enunciado pede análise de pesos e de convergência, que não cabem dentro de P.

---

## Parte 1 — Experiência: os dados — 2,5 min

### Slide 2 — A base: diamonds (OpenML 42225)

- 53.940 linhas originais, 9 preditoras e o alvo `price` em dólares; **zero valores faltantes**.
- Limpeza: 20 linhas com alguma dimensão igual a zero (fisicamente impossível) e 3 com dimensão acima de 20 mm (erro de digitação) → 23 removidas, restam **53.917 linhas**.
- Divisão aleatória com semente 42: 43.134 no treino e 10.783 no teste (20%).
- Padronização $z=(x-\mu)/\sigma$ com $\mu$ e $\sigma$ calculados **só no treino**.

Figura: nenhuma — tabela de limpeza (`resultados/limpeza.csv`) e a lista dos 9 atributos com o significado de cada um.

*O que falar:* explicar por que a padronização é ajustada apenas no treino (ajustar na base completa vaza informação do teste e infla a métrica) e que aqui ela não é opcional: sem padronizar, o Gradient Descent não converge com taxa utilizável.

### Slide 3 — O alvo é desbalanceado, e as categóricas também

- `price` vai de 326 a 18.823 dólares, com mediana 2.401 e média 3.930,91 — média bem acima da mediana, assimetria **1,618**.
- Aplicando $\log$: a assimetria cai para **0,115**, praticamente simétrica. Isso motiva uma das quatro configurações testadas.
- Categóricas desbalanceadas: `cut` tem 21.547 pedras `Ideal` contra 1.609 `Fair` (13,4 vezes); `clarity` tem 13.063 `SI1` contra 738 `I1`.

Figuras: `01_distribuicao_alvo.png` (histograma linear e log) e `03_categoricas.png`.

*O que falar:* em regressão, desbalanceamento do alvo significa que o MSE é dominado pelas pedras caras — um erro de 10% numa pedra de 18 mil dólares pesa no custo o equivalente a cerca de **2.600** erros de 10% numa pedra de 350 dólares, porque o custo é quadrático: $(1.800/35)^2\approx 2.645$. É por isso que o log ajuda.

### Slide 4 — Correlação: quatro atributos dizem quase a mesma coisa

- Correlação com o preço: `carat` **0,922**, `y` 0,889, `x` 0,887, `z` 0,882. `table` fica em 0,127 e `depth` em **-0,011** (praticamente zero).
- O problema não é a correlação com o alvo, é a correlação **entre** as preditoras: `x` com `y` = 0,999 e `carat` com `x` = 0,978.
- Consequência antecipada: quatro atributos quase redundantes, um dos quais (`carat`) explica sozinho 85% da variância do preço ($0{,}922^2$).

Figura: `02_correlacao.png`.

*O que falar:* dar a intuição física — `carat` é peso, e `x`, `y`, `z` são as três dimensões; volume e peso medem a mesma coisa, então a redundância era esperada. Avisar que essa é a causa do problema que aparece na Parte 4.

### Slide 5 — t-SNE: não existem grupos, existe um gradiente — 1,0 min

- t-SNE sobre as **53.917 pedras** da base limpa (sem subamostragem), perplexidade 30, inicialização por PCA, sobre os 9 atributos padronizados.
- Sem rótulo (figura 04): a nuvem mostra estrutura local, mas não se parte em classes disjuntas com fronteiras vazias.
- Com rótulo (figura 05): não há classe para colorir em regressão, então o preço foi discretizado em quartis **apenas como cor** — Q1 barato a Q4 caro. Os quartis aparecem organizados como transição contínua da esquerda para a direita, não como ilhas separadas: o primeiro eixo da projeção correlaciona, em módulo, **0,845** com `carat` e **0,781** com $\log(price)$. Agrupando a projeção em 8 grupos por k-médias, a informação mútua ajustada é de **0,428** com `cut` e **0,322** com a faixa de `carat`, contra **0,068** com `clarity` e **0,090** com `color` — os blocos visíveis são lapidação e tamanho, não pureza nem cor. (Números em `resultados/tsne_diagnostico.json`.)
- Leitura para a modelagem: se o preço variasse em degraus por grupo, um modelo linear único seria inadequado e o caminho natural seria clusterizar e ajustar um modelo por cluster. Como a variação é contínua e monotônica, **um único modelo linear global é a escolha correta** — e o $R^2$ de 0,96 na Parte 3 confirma isso.

Figuras: `04_tsne_sem_rotulo.png` e `05_tsne_com_rotulo.png`, lado a lado.

*O que falar:* dizer explicitamente que o t-SNE aqui é ferramenta de decisão, não enfeite: ele foi consultado **antes** de escolher o modelo, e foi ele que descartou a hipótese de mistura de regimes. Lembrar que distâncias globais no t-SNE não são interpretáveis — só a vizinhança local é.

---

## Parte 2 — Tarefa: o problema e os dois algoritmos — 2,0 min

### Slide 6 — A tarefa formalizada

- Regressão: aprender $f:\mathbb{R}^{9}\to\mathbb{R}$ que prevê o preço a partir de peso, lapidação, cor, pureza e três dimensões.
- Hipótese linear com intercepto: $\hat{y}=Aw$, onde $A$ é $X$ padronizado com uma coluna de uns na frente.
- Função de custo, erro quadrático médio: $$J(w)=\frac{1}{n}\lVert Aw-y\rVert^{2}$$
- $J$ é convexa e quadrática: existe um único mínimo global, e os dois algoritmos têm que chegar nele.

Figura: nenhuma — só as três equações.

*O que falar:* enfatizar a convexidade, porque ela é a garantia que sustenta toda a Parte 4: não existe mínimo local onde o Gradient Descent possa ficar preso, então qualquer diferença entre os dois métodos é diferença de *quantas iterações*, nunca de *qual solução*.

### Slide 7 — Dois caminhos para o mesmo mínimo

- **Equações normais** (solução exata): $A^{\top}A\,w = A^{\top}y$, resolvida com `np.linalg.solve` — um passo, sem hiperparâmetro.
- **Gradient Descent**: $\nabla J=\frac{2}{n}A^{\top}(Aw-y)$, atualização $w \leftarrow w-\eta\,\nabla J$, partindo de $w=0$.
- Critério de parada do GD: $\lVert\nabla J\rVert<10^{-10}$ ou o teto de 50.000 épocas. Guarda de divergência: se o custo subir acima do custo inicial, o treino aborta com erro explícito.
- Treino e predição são módulos separados em `src/modelos.py`; as métricas $R^2$, MSE e RMSE estão implementadas à mão em `src/metricas.py`, sem scikit-learn.

Figura: nenhuma — as duas equações em colunas paralelas.

*O que falar:* explicar por que a solução exata serve de gabarito: ela é o alvo que o GD tem que alcançar, e é isso que permite medir $\lVert w - w^{*}\rVert$ época por época no slide 14. Mencionar que resolver o sistema é mais estável numericamente do que inverter $A^{\top}A$.

### Slide 8 — Quatro configurações, oito execuções

- Duas codificações das categóricas: **ordinal** na ordem de qualidade (9 colunas) e **one-hot** com o pior nível como referência (23 colunas).
- Duas escalas do alvo: **dólares** e **$\log$ de dólares**; no caso log as previsões voltam com $\exp$ antes de medir.
- 4 configurações × 2 algoritmos = **8 execuções**, todas com a mesma divisão e a mesma semente.
- Detalhe que muda o resultado: a ordem ordinal é a de *qualidade* (`J`→`D` em cor, `I1`→`IF` em pureza), não a ordem alfabética do arquivo — usar a ordem do ARFF inverte o sinal do peso de `color`.

Figura: nenhuma — diagrama do pipeline: carregar → limpar → codificar → dividir → padronizar → treinar → avaliar.

*O que falar:* avisar que $R^2$ medido em escala log **não** é comparável com $R^2$ em dólares, e que por isso toda métrica da tabela do slide 10 está em dólares; e admitir a limitação: a volta por $\exp$ é enviesada, prevê a mediana condicional e não a média.

---

## Parte 3 — Desempenho: $R^2$ e MSE — 2,0 min

### Slide 9 — Como medimos

- $R^2=1-\mathrm{SQ_{res}}/\mathrm{SQ_{tot}}$: fração da variância do preço explicada pelo modelo. Referência: prever sempre a média dá $R^2=0$.
- $\mathrm{MSE}=\frac{1}{n}\sum(y-\hat{y})^{2}$, em dólares ao quadrado; reportamos também o RMSE, em dólares, porque é interpretável.
- Tudo medido nas **10.783 pedras de teste**, que nenhum modelo viu no treino.

Figura: nenhuma — as duas fórmulas.

*O que falar:* justificar reportar as duas métricas: $R^2$ é adimensional e permite comparar entre escalas do alvo, o RMSE diz quanto o modelo erra em dólares — e um erro médio de 792 dólares numa base cuja mediana é 2.401 é o número que o cliente entende.

### Slide 10 — A tabela comparativa — 1,0 min

| codificação | alvo | $R^2$ teste (eq. normais) | $R^2$ teste (GD) | MSE teste (USD²) | RMSE (USD) |
|---|---|---|---|---|---|
| ordinal | USD | 0,9100 | 0,9100 | 1.408.606,43 | 1.186,85 |
| ordinal | log | 0,9524 | 0,9524 | 745.543,78 | 863,45 |
| one-hot | USD | 0,9218 | 0,9218 | 1.223.692,48 | 1.106,21 |
| **one-hot** | **log** | **0,9599** | **0,9599** | **628.051,98** | **792,50** |

- Melhor configuração: one-hot + alvo em log, $R^2=0{,}9599$ e RMSE de 792,50 dólares.
- Da pior para a melhor: $R^2$ sobe 0,0499 (de 0,9100 para 0,9599) e o **MSE cai 55,4%**; o RMSE cai 394,35 dólares.
- Os dois algoritmos coincidem em todas as quatro linhas: a maior diferença de $R^2$ entre eles é de $1{,}7\times10^{-10}$, na nona casa decimal.

Figura: nenhuma — a tabela é o slide.

*O que falar:* separar os dois efeitos: o log vale mais que o one-hot (+0,0424 de $R^2$ contra +0,0118 na escala em dólares), porque ele corrige a assimetria de 1,618 vista no slide 3 — o ganho vem do pré-processamento, não de trocar o otimizador.

### Slide 11 — O custo de convergir por gradiente

- Equações normais: entre **0,0036 e 0,0122 segundos** por ajuste.
- Gradient Descent: entre **22,04 e 56,71 segundos**, com 31.191 a 50.000 épocas.
- No caso ordinal em dólares: 0,0043 s contra 36,56 s — cerca de **8,5 mil vezes** mais lento, para o mesmo resultado.
- As duas configurações em dólares bateram no teto de 50.000 épocas sem atingir a tolerância; as duas em log pararam antes (31.191 e 38.845 épocas).

Figura: nenhuma — gráfico de barras dos tempos ou a coluna `segundos` da tabela.

*O que falar:* a conclusão prática é direta: com 9 ou 23 atributos e custo convexo, a forma fechada é a escolha certa, e o GD está aqui por dois motivos — o enunciado pede, e ele é o único dos dois que escala para milhões de atributos, onde montar $A^{\top}A$ deixa de ser viável.

---

## Parte 4 — Análise: pesos, colinearidade e convergência — 2,5 min

### Slide 12 — Análise dos pesos

- Coeficientes padronizados (ordinal, dólares), em ordem de magnitude: `carat` **5.150,26**, `y` 2.922,12, `x` **-2.007,21**, `z` **-1.934,83**, `clarity` 816,24, `color` 548,88, `cut` 145,34, `depth` 125,62, `table` -39,93.
- Fazem sentido: `carat` domina, e `clarity` (816,24) pesa mais que `color` (548,88), que pesa mais que `cut` (145,34) — a ordem que o mercado de joias usa.
- Não fazem sentido: `x` e `z` têm coeficiente **negativo** apesar de correlação de +0,89 e +0,88 com o preço. Lido isoladamente, o modelo diria que diamante mais comprido é mais barato.
- Como estão padronizados, os pesos são comparáveis entre si: cada um é o efeito de um desvio padrão do atributo, em dólares.

Figura: `06_pesos_e_vif.png` (painel esquerdo).

*O que falar:* o sinal negativo de `x` e `z` não é bug, é a assinatura da colinearidade: com `carat` já no modelo, o que sobra para `x` explicar é o resíduo, e o coeficiente passa a compensar o excesso dos outros em vez de medir efeito próprio. Coeficiente de regressão é efeito *parcial*, não correlação.

### Slide 13 — Colinearidade: por que aqueles pesos não são interpretáveis

- VIF: `y` **539,5**, `x` **537,2**, `z` **489,7**, `carat` 24,9, `depth` 8,9 — o limiar usual de alerta é 10. `cut`, `color`, `clarity` e `table` ficam entre 1,1 e 1,6.
- Reajustando o modelo em 200 reamostras bootstrap, o desvio dos coeficientes é de **761,97** em `z`, 427,87 em `y` e 396,70 em `x`, contra 8,82 em `clarity` e 7,44 em `color`.
- Razão |coeficiente| / desvio: `clarity` 92,5, `color` 73,8, `carat` 46,0, `cut` 19,5 — sólidos. Já `y` 6,8, `x` -5,1, `table` -4,9, `z` -2,5 e `depth` **1,35** — o peso de `depth` não se distingue de zero.
- Isto **não** degrada a predição: o $R^2$ de 0,96 continua valendo. Degrada a *interpretação* dos pesos das dimensões.

Figura: `07_estabilidade.png` (boxplot das 200 reamostras) com o painel de VIF de `06_pesos_e_vif.png`.

*O que falar:* enunciar a distinção que vale a nota: predição e interpretação são objetivos diferentes e falham de formas diferentes. Se o objetivo fosse explicar preço, o caminho seria remover ou combinar `x`, `y` e `z` — por exemplo trocar os três por volume; como o objetivo é prever, mantê-los é inofensivo.

### Slide 14 — Convergência do GD: onde os parâmetros param de mudar — 1,0 min

- Taxa 0,2, 50.000 épocas: o custo cai de $1{,}4726\times10^{7}$ para $1{,}4753\times10^{6}$.
- Três marcos distintos: custo a **1% do final na época 331**; a **0,01% na época 5.034**; e os pesos só param de mudar de verdade, no critério $\lVert w-w^{*}\rVert<10^{-6}$, na **época 45.776**. Distância final: $1{,}02\times10^{-7}$ — o GD chega no mesmo ponto das equações normais.
- Resposta ao enunciado: sim, existe esse ponto, mas ele depende do critério. Pelo custo, o modelo está pronto em ~331 épocas; pelos parâmetros, em ~45.776. A diferença de duas ordens de grandeza é consequência do número de condição da hessiana, $\kappa=3.579{,}9$: o vale do custo é alongado, e as últimas direções são percorridas devagar.
- Limite teórico da taxa: $\eta_{\max}=2/\lambda_{\max}=2/8{,}607=0{,}2324$. Empiricamente, **0,23 converge e 0,24 diverge** — teoria e experimento batem.

Figuras: `08_convergencia.png` (custo por época e $\lVert w-w^{*}\rVert$ por época) e `09_taxa_aprendizado.png`.

*O que falar:* fechar o arco da apresentação: $\kappa=3.580$ é a mesma colinearidade do slide 13 aparecendo em outro lugar — os atributos redundantes achatam uma direção da hessiana, $\lambda_{\min}=0{,}0024$, e é isso que faz o GD precisar de 45 mil épocas onde a forma fechada leva 4 milissegundos. Um único diagnóstico explica pesos instáveis **e** convergência lenta.

---

## Fechamento — 0,5 min

### Slide 15 — Três conclusões

1. **As duas implementações encontram o mesmo ótimo; o que difere é o preço pago.** $R^2$ igual em todas as quatro configurações, com diferença máxima de $1{,}7\times10^{-10}$ entre os dois métodos, e $\lVert w-w^{*}\rVert=1{,}02\times10^{-7}$ ao final — mas 0,0043 s contra 36,56 s. Custo convexo com 9 atributos: forma fechada.
2. **O ganho de desempenho veio do pré-processamento, não do otimizador.** Corrigir a assimetria do alvo (1,618 → 0,115 com log) e trocar ordinal por one-hot levou o $R^2$ de teste de 0,9100 para 0,9599 e cortou o MSE em 55,4% — RMSE de 792,50 dólares.
3. **Predizer bem não é explicar bem.** Com VIF de 537, 540 e 490 em `x`, `y` e `z`, dois desses pesos saem negativos e o desvio bootstrap de `z` chega a 762. Só `clarity`, `color`, `carat` e `cut` têm coeficiente de 19 a 93 vezes o próprio desvio; e a mesma colinearidade que quebra a interpretação é a que faz o Gradient Descent precisar de 45.776 épocas para estabilizar os pesos, contra 331 para estabilizar o custo.

Figura: nenhuma — três linhas, uma por conclusão.

*O que falar:* encerrar com o próximo passo honesto: o teto de 0,96 com modelo linear provavelmente não se rompe ajustando o otimizador, e sim com atributos não lineares (interação `carat` × `clarity`, volume $x\,y\,z$) ou com um modelo não linear — e o t-SNE do slide 5, que mostrou gradiente contínuo e não grupos, é a evidência de que não há regimes separados a explorar antes disso.
