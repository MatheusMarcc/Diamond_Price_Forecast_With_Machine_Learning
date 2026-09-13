# Correções aplicadas na revisão de fatos

Cada seção do relatório passou por um segundo agente que conferiu todo número
citado contra os arquivos em `resultados/`. Registro do que foi corrigido.


## introducao — 3 correção(ões)

**Trecho:** deixar visível, em cada $w_j$, quanto o mercado paga por uma unidade de cada característica

- *Problema:* Afirmacao sobre o codigo incorreta. Em src/dados.py a classe Padronizador aplica z = (x - media)/desvio a TODAS as preditoras antes do ajuste (preparar() sempre padroniza, com media e desvio do treino), e a propria figura 06 rotula o eixo como 'Coeficientes padronizados'. Portanto cada w_j mede o efeito de UM DESVIO PADRAO do atributo, nao de uma unidade fisica (quilate, milimetro, ponto percentual).
- *Correção:* Trocado para 'quanto o mercado paga por um desvio padrao de cada caracteristica (os atributos entram padronizados, com media e desvio estimados apenas no treino)'.

**Trecho:** correlação de 1,00 entre `x` e `y`

- *Problema:* Valor superestimado/impreciso. resultados/correlacao.csv da corr(x, y) = 0,9986573; resumo.json so mostra 1.0 porque arredonda para uma casa. Escrever '1,00' afirma colinearidade exata, que os dados nao sustentam (se fosse 1,00 exato o VIF seria infinito, e ele e 537,2 / 539,5).
- *Correção:* Trocado para 'correlação de 0,999 entre `x` e `y`' (valor de correlacao.csv arredondado a tres casas).

**Trecho:** e a descida de gradiente reproduz a solução exata com distância final de $1{,}02 \times 10^{-7}$ entre os vetores de peso

- *Problema:* Numero atribuido a configuracao errada pelo contexto da frase. O 1,0181e-07 de resumo.json (bloco 'convergencia') vem do experimento de 03_analises.py, que roda com codificacao='ordinal' e alvo_em_log=False, taxa 0,2 e 50.000 epocas. A frase o coloca logo apos 'a melhor configuracao, one-hot com alvo em log', sugerindo que a distancia se refere a esse ajuste — para o qual nao ha distancia final registrada em nenhum arquivo (o GD one-hot/log parou por tolerancia em 38.845 epocas).
- *Correção:* Explicitada a configuracao de origem: 'na configuracao ordinal em dolares, com taxa 0,2 e 50.000 epocas, a distancia final entre os dois vetores de peso e de 1,02 x 10^-7'.


## descricao — 7 correção(ões)

**Trecho:** Em `clarity`, a ordenação alfabética (IF, I1, SI1, SI2, VS1, VS2, VVS1, VVS2) tambem nao corresponde a escala de pureza

- *Problema:* Ordem alfabetica de clarity trocada: I1 vem antes de IF. dados/diamonds_metadados.json registra os niveis do ARFF exatamente como [I1, IF, SI1, SI2, VS1, VS2, VVS1, VVS2]. O rascunho inverteu os dois primeiros, o que contradiz a propria fonte citada no paragrafo.
- *Correção:* Trocado para (I1, IF, SI1, SI2, VS1, VS2, VVS1, VVS2).

**Trecho:** Configuracao padrao: $\eta = 0{,}2$, limite de 50.000 epocas, tolerancia $10^{-10}$

- *Problema:* Chamar isso de "configuracao padrao" e incorreto para a classe. Em src/modelos.py a assinatura e `RegressaoLinearGD(taxa=0.1, epocas=2000, tolerancia=1e-10)`. Os valores 0,2 e 50.000 sao os defaults de linha de comando de 02_treino.py e 03_analises.py (`--taxa 0.2`, `--epocas 50_000`), nao da classe.
- *Correção:* Reescrito como "Configuracao usada nos experimentos, que e o padrao de linha de comando de 02_treino.py e 03_analises.py" e acrescentada a ressalva de que os padroes da classe sao 0,1 e 2.000 epocas.

**Trecho:** Tres funcoes de diagnostico completam o modulo

- *Problema:* Sao quatro funcoes de diagnostico em src/modelos.py: espectro(), vif(), pesos_por_reamostragem() e distancia_ate_a_solucao_exata(). O proprio rascunho descreve a quarta na frase seguinte, contradizendo a contagem.
- *Correção:* Trocado para "Quatro funcoes de diagnostico" e a ultima frase reescrita como "A quarta, `distancia_ate_a_solucao_exata()`, compara...".

**Trecho:** com `carat` na ordem de $10^{-1}$ e `depth` na ordem de $10^{1}$

- *Problema:* Afirmacao quantitativa sem respaldo: resultados/estatisticas.csv da media de carat 0,7977 e mediana 0,70, ou seja, ordem de $10^0$, nao $10^{-1}$. O minimo (0,2) e a unica parte da coluna na ordem de $10^{-1}$.
- *Correção:* Substituido pelos valores exatos de estatisticas.csv: "com `carat` em torno de 0,80 e `depth` em torno de 61,75", o que preserva o argumento de escala sem inventar ordem de grandeza.

**Trecho:** uma coluna binaria por nivel, com `drop_first=True` em `codificar_onehot()`

- *Problema:* Nome de parametro errado. Em src/dados.py a assinatura e `codificar_onehot(df, descartar_primeira: bool = True)`; `drop_first` e o argumento de `pd.get_dummies` para o qual esse parametro e repassado.
- *Correção:* Reescrito como "com `descartar_primeira=True` em `codificar_onehot()`, repassado ao `drop_first` de `pd.get_dummies`".

**Trecho:** um valor fora de escala em uma coluna cujo maximo legitimo e 10,54 mm

- *Problema:* 10,54 mm e o maximo de `y` apos a limpeza; o texto sugere um unico maximo para as tres colunas filtradas. Por resultados/estatisticas.csv os maximos sao 10,74 (`x`), 10,54 (`y`) e 6,98 (`z`) - o numero citado nao vale para x nem para z.
- *Correção:* Reescrito como "em colunas cujos maximos legitimos sao 10,74 mm (`x`), 10,54 mm (`y`) e 6,98 mm (`z`)".

**Trecho:** indica que $\eta$ passou do limite estavel $2/\lambda_{\max}$

- *Problema:* Ambiguidade de notacao, nao erro de fato: o simbolo $\lambda$ aparece sem definicao nesse ponto e pode ser lido como parametro de penalizacao, tema que o enunciado retirou. Em resumo.json e em src/modelos.py o valor e estritamente o maior autovalor da hessiana (lambda_max = 8,607072, taxa_maxima = 2/8,607072 = 0,232367).
- *Correção:* Acrescentado "sendo $\lambda_{\max}$ o maior autovalor da hessiana" na propria frase. Nenhuma mencao a Ridge, regularizacao ou penalizacao existia no rascunho.


## estudo — 5 correção(ões)

**Trecho:** `depth` tem o segundo maior VIF entre os atributos não-dimensionais (8,9, contra 1,1 a 1,6 dos demais atributos não dimensionais)

- *Problema:* Ranking errado e frase internamente contraditória. Em resultados/resumo.json os VIF são: y 539,5, x 537,2, z 489,7, carat 24,9, depth 8,9, table 1,6, cut 1,5, clarity 1,2, color 1,1. Excluindo os quatro atributos de tamanho (carat, x, y, z), depth (8,9) é o MAIOR VIF, não o segundo — os demais ficam entre 1,1 e 1,6, como o próprio parêntese admite. Havia ainda repetição de 'nao dimensionais' na mesma frase.
- *Correção:* `depth` tem o maior VIF entre os atributos que não medem tamanho (8,9, contra 1,1 a 1,6 de `cut`, `color`, `clarity` e `table`)

**Trecho:** na codificação one-hot, a coluna binária de Fair é estimada com 1.609 exemplos e a de I1 com 738, contra 21.547 de Ideal. O coeficiente estimado para esses níveis tem variância maior

- *Problema:* Contradiz o código. Em src/dados.py a função codificar_onehot chama pd.get_dummies(..., drop_first=True) sobre um Categorical ordenado por ORDEM (pior para melhor), de modo que o PIOR nível de cada atributo é descartado e vira a referência. Não existe coluna binária para Fair, J ou I1, logo não há coeficiente estimado com 1.609 nem com 738 exemplos.
- *Correção:* Na codificação one-hot deste projeto o pior nível de cada atributo é descartado e passa a ser a referência (`Fair` em `cut`, `J` em `color`, `I1` em `clarity`): todos os coeficientes de `cut` são lidos contra 1.609 pedras Fair e os de `clarity` contra 738 pedras I1, enquanto o nível Ideal entra com 21.547 exemplos na sua própria coluna binária. Referência escassa e coluna binária pouco populada levam à mesma consequência — variância maior no coeficiente estimado

**Trecho:** em um diamante de lapidação redonda — a maioria da base — as duas medidas são o mesmo diâmetro medido em dois eixos

- *Problema:* Afirmação quantitativa sem respaldo: a base não tem coluna de forma ou tipo de lapidação (o ARFF traz apenas carat, cut, color, clarity, depth, table, price, x, y, z), então não há como sustentar que a lapidação redonda seja 'a maioria da base'.
- *Correção:* em uma pedra de lapidação redonda as duas medidas são o mesmo diâmetro medido em dois eixos. A base não registra a forma da lapidação, mas $r = 0{,}999$ entre as duas colunas é praticamente uma identidade

**Trecho:** A quarta configuração testada neste projeto existe exatamente para medir esse efeito.

- *Problema:* Não confere com resultados/metricas.csv: o alvo em log aparece em DUAS das quatro configurações (ordinal/log e onehot/log), não apenas em uma. Na ordem em que 02_treino.py roda, a 'quarta configuração' seria onehot/log, o que deixaria de fora justamente ordinal/log — a comparação usada no item 3 da seção 3.4.
- *Correção:* Duas das quatro configurações testadas neste projeto trocam o alvo por seu logaritmo exatamente para medir esse efeito.

**Trecho:** a correlação entre os dois é 0,952 e o erro absoluto mediano é de 0,027 ponto percentual, com 93,0% das linhas dentro de 0,05 ponto

- *Problema:* Esses três números não constam de nenhum arquivo em resultados/ (nem resumo.json nem os CSVs) e nenhum dos scripts 01/02/03 os calcula — não são reprodutíveis pela pipeline atual.
- *Correção:* MANTIDOS. Recalculei a reconstrução 2z/(x+y)*100 sobre as 53.917 linhas limpas a partir de dados/diamonds.csv: correlação 0,9518, erro absoluto mediano 0,0266 pp e 93,02% das linhas dentro de 0,05 pp. Os valores do rascunho estão corretos, mas convém acrescentar esse cálculo a 01_estudo_base.py para que passem a sair em resultados/.


## resultados — 8 correção(ões)

**Trecho:** | Configuração | Épocas do GD | $\max_j |w_j^{GD} - w_j^{*}|$ | Erro relativo máximo | $\|w^{GD} - w^{*}\|_2$ | ... quatro linhas com 7,24e-8 / 3,57e-11 / 1,02e-7, 3,16e-8 / 4,78e-8 / 4,11e-8, 4,13e-6 / 2,18e-9 / 5,58e-6, 3,69e-8 / 9,00e-8 / 4,99e-8

- *Problema:* Onze dos doze valores da tabela nao existem em nenhum arquivo de resultado. resumo.json so registra convergencia.distancia_final = 1,0181e-07, que corresponde a ||w_GD - w*||_2 da configuracao ordinal/USD. 02_treino.py calcula max|w_GD - w*| apenas para ordinal/USD (bloco CONFERENCIA) e apenas imprime no console, sem gravar; 03_analises.py roda convergencia so em ordinal/USD. Nao ha nenhuma comparacao de pesos gravada para ordinal/log, one-hot/USD ou one-hot/log. Alem disso, os pipes nao escapados dentro de $\max_j |...|$ quebram a renderizacao da tabela markdown.
- *Correção:* Tabela substituida por uma comparacao GD x equacoes normais construida com valores derivados de metricas.csv (diferenca absoluta de R2 de teste, diferenca absoluta e relativa de MSE de teste: 2,2e-14 / 3,5e-7 / 2,5e-13; 1,3e-10 / 2,1e-3 / 2,8e-9; 7,4e-13 / 1,2e-5 / 9,4e-12; 1,7e-10 / 2,6e-3 / 4,1e-9). A comparacao de pesos ficou restrita a ordinal/USD, com o valor real ||w_GD - w*||_2 = 1,02e-7, e as demais configuracoes marcadas como [VERIFICAR].

**Trecho:** A maior discrepância em toda a bateria é de $4{,}13 \times 10^{-6}$ em um coeficiente da configuração one-hot/USD, o que corresponde a $2{,}18 \times 10^{-9}$ do valor do próprio coeficiente

- *Problema:* Numeros derivados da tabela inventada. Nao ha comparacao de pesos gravada para one-hot/USD em nenhum arquivo.
- *Correção:* Substituido pela maior discrepancia efetivamente verificavel: 4,1e-9 relativo no MSE de one-hot/log (0,0026 USD sobre 628.051,98 USD2), e pela distancia de pesos documentada para ordinal/USD (1,02e-7).

**Trecho:** Em termos práticos, a diferença entre os dois algoritmos é de ordem $10^{-6}$ USD sobre coeficientes de ordem $10^{3}$ USD.

- *Problema:* A ordem 1e-6 vem da tabela inventada. O unico valor gravado (resumo.json) e 1,02e-7, ou seja, ordem 1e-7.
- *Correção:* Reescrito citando 1,02e-7 contra coeficientes de ordem 1e3 USD, com o maior valor real de pesos.csv (carat = 5.150,26 USD).

**Trecho:** A verificação foi feita comparando diretamente os vetores de pesos nas quatro configurações

- *Problema:* Afirmacao sobre o codigo incorreta. 02_treino.py compara os vetores de pesos apenas para codificacao='ordinal', alvo_em_log=False; 03_analises.py tambem usa somente ordinal/USD.
- *Correção:* Trocado por: a verificacao registrada nos arquivos e a comparacao das metricas nas quatro configuracoes, e a comparacao direta de pesos existe apenas para ordinal/USD.

**Trecho:** A razão vai de 5.120 vezes (one-hot/log) a 14.951 vezes (ordinal/USD).

- *Problema:* Contas erradas a partir dos tempos de resumo.json. 32,0383 / 0,0063 = 5.085,4 e 29,4478 / 0,0020 = 14.723,9. As configuracoes extremas citadas estao certas, so os valores estao errados.
- *Correção:* A razão vai de 5.085 vezes (one-hot/log) a 14.724 vezes (ordinal/USD).

**Trecho:** O custo computacional, porém, difere por quatro ordens de grandeza.

- *Problema:* As razoes reais vao de 5.085 (10^3,7) a 14.724 (10^4,2); a menor delas nao chega a quatro ordens de grandeza.
- *Correção:* difere por três a quatro ordens de grandeza.

**Trecho:** É por isso que as oito linhas da Tabela da Seção 4.1 aparecem em pares idênticos até a quarta casa

- *Problema:* Falso para o MSE dentro da propria tabela da Secao 4.1: one-hot/log aparece como 628.051,98 (equacoes normais) e 628.051,97 (gradient descent), que diferem ja na segunda casa decimal exibida. A coincidencia ate a quarta casa vale para o R2.
- *Correção:* É por isso que as oito linhas da tabela da Seção 4.1 aparecem aos pares, com $R^2$ idêntico até a quarta casa decimal.

**Trecho:** caso o número de atributos crescesse a ponto de inverter $A^\top A$ ficar caro

- *Problema:* Afirmacao sobre o codigo imprecisa: modelos.py usa np.linalg.solve(A.T @ A, A.T @ y), com comentario explicito dizendo que resolver o sistema e preferivel a inverter A^T A com np.linalg.inv. O relatorio nao deve atribuir inversao explicita a implementacao.
- *Correção:* caso o número de atributos crescesse a ponto de resolver $A^\top A\,w = A^\top y$ ficar caro


## analises — 7 correção(ões)

**Trecho:** e cada uma delas contra `carat` fica em 0,977–0,978

- *Problema:* O intervalo esta errado na ponta inferior. Em resultados/correlacao.csv: carat-x = 0,9778, carat-y = 0,9769 e carat-z = 0,9765. Arredondando a tres casas, o menor valor e 0,976, nao 0,977.
- *Correção:* e cada uma delas contra `carat` fica em 0,976–0,978

**Trecho:** um erro nos pesos e atenuado por um fator quase tres mil vezes menor do que na direcao de $\lambda_{\max}$

- *Problema:* O fator e a propria razao kappa = lambda_max/lambda_min = 3.579,9 (resumo.json, campo espectro.kappa). 'Quase tres mil' subestima o valor e contradiz o proprio paragrafo anterior, que ja cita kappa ~ 3.580.
- *Correção:* um erro nos pesos e atenuado por um fator cerca de 3.580 vezes menor do que na direcao de $\lambda_{\max}$

**Trecho:** Todas as oito execucoes, de todo modo, chegaram ao mesmo $R^2$ e ao mesmo MSE das equacoes normais ate a quarta casa decimal

- *Problema:* Dois erros. (1) As oito linhas de metricas.csv sao 4 equacoes normais + 4 gradient descent; so existem quatro execucoes de GD para comparar contra a solucao exata. (2) O MSE NAO coincide ate a quarta casa decimal: ordinal/log da 745.543,7797 (equacoes normais) contra 745.543,7776 (GD), e one-hot/log da 628.051,9753 contra 628.051,9727 — a divergencia aparece ja na segunda/terceira casa decimal. O que se sustenta nos dados e que as diferencas so aparecem a partir do nono algarismo significativo (pior caso: diferenca relativa de 4,1 x 10^-9 em one-hot/log).
- *Correção:* As quatro execucoes do gradient descent, de todo modo, reproduziram o $R^2$ e o MSE das equacoes normais com diferencas que so aparecem a partir do nono algarismo significativo

**Trecho:** O unico hiperparametro do algoritmo implementado e a taxa de aprendizado (junto com o orcamento de epocas).

- *Problema:* Nao bate com o codigo: `RegressaoLinearGD.__init__` (src/modelos.py) recebe tres hiperparametros — `taxa`, `epocas` e `tolerancia`. A frase tambem se contradiz internamente ('unico' seguido de 'junto com') e com o ultimo paragrafo da propria secao, que fala em 'os tres parametros — taxa, epocas e tolerancia'.
- *Correção:* O algoritmo implementado tem tres hiperparametros — taxa de aprendizado, orcamento de epocas e tolerancia de parada (`src/modelos.py`, `RegressaoLinearGD`) —, e o que decide se o metodo funciona ou nao e a taxa.

**Trecho:** as curvas de 0,001 e 0,01 ainda estao no inicio da descida quando o grafico termina, enquanto 0,1 e 0,2 ja atingiram o plato do custo

- *Problema:* Nao e o que figuras/09_taxa_aprendizado.png mostra. A curva de 0,01 ja esta achatada por volta da epoca 250, num patamar acima do de 0,1 e 0,2; apenas a de 0,001 continua em descida franca na epoca 400. As curvas de 0,1 e 0,2 chegam ao plato ainda nas primeiras dezenas de epocas.
- *Correção:* a curva de 0,001 ainda esta em plena descida quando o grafico termina, a de 0,01 so achata por volta da epoca 250 e num patamar visivelmente acima, enquanto 0,1 e 0,2 ja atingiram o plato do custo nas primeiras dezenas de epocas

**Trecho:** VIF baixo, desvio bootstrap de 1 a 5% do coeficiente, sinal estavel em todas as 200 reamostras.

- *Problema:* Afirmacao quantitativa sem respaldo nos arquivos. resultados/pesos.csv guarda apenas o desvio padrao entre reamostras (coluna `desvio_bootstrap`), nao o comportamento individual das 200 reajustagens; e a Figura 7 e gerada com `showfliers=False` (03_analises.py), portanto nem os extremos aparecem. Nao ha como verificar 'todas as 200'.
- *Correção:* VIF baixo, desvio bootstrap de 1 a 5% do coeficiente e coeficiente a no minimo 19,5 desvios bootstrap de zero.

**Trecho:** Com $\eta = 0{,}2$, a configuracao ordinal/USD consumiu todas as 50.000 epocas sem acionar a tolerancia de gradiente

- *Problema:* Incompleto de forma enganosa: em resultados/metricas.csv as DUAS configuracoes com alvo em dolares (ordinal/USD e onehot/USD) executaram 50.000 epocas. Citar so ordinal/USD sugere que a one-hot/USD teria parado antes, o que e falso.
- *Correção:* Com $\eta = 0{,}2$, as duas configuracoes com alvo em dolares consumiram todas as 50.000 epocas sem acionar a tolerancia de gradiente


## apresentacao — 3 correção(ões)

**Trecho:** um erro de 10% numa pedra de 18 mil dólares pesa no custo o equivalente a 300 erros de 10% numa pedra de 350 dólares

- *Problema:* Conta errada. O custo é quadrático: (0,10 x 18.000)^2 / (0,10 x 350)^2 = 1.800^2 / 35^2 = 2.645, nao 300. O fator 300 nao sai de nenhum arquivo e subestima o argumento em quase uma ordem de grandeza.
- *Correção:* Trocado por 'o equivalente a cerca de 2.600 erros de 10% numa pedra de 350 dolares, porque o custo e quadratico: $(1.800/35)^2 \approx 2.645$'.

**Trecho:** as diferenças de $R^2$ aparecem só a partir da nona casa decimal / $R^2$ idêntico até a nona casa decimal em todas as quatro configurações

- *Problema:* As duas frases se contradizem (uma diz que a nona casa ja difere, a outra diz que ate a nona casa e identica) e nenhuma das duas e verificavel como esta. Pelo metricas.csv, a maior diferenca entre equacoes normais e GD e 1,66e-10 (onehot+log: 0,9598856139141716 contra 0,9598856140800156); as outras sao 1,3e-10 (ordinal+log), 7,4e-13 (onehot+USD) e 2,2e-14 (ordinal+USD).
- *Correção:* Padronizado nos dois slides para o valor medido: 'a maior diferenca de $R^2$ entre eles e de $1,7\times10^{-10}$, na nona casa decimal' (slide 10) e '$R^2$ igual em todas as quatro configuracoes, com diferenca maxima de $1,7\times10^{-10}$ entre os dois metodos' (slide 15).

**Trecho:** a mesma colinearidade que quebra a interpretação é a que multiplica por 100 o número de épocas do Gradient Descent

- *Problema:* O fator 100 nao existe em nenhum arquivo e nao tem baseline definido (multiplica por 100 em relacao a que?). O unico par de numeros comparaveis no resumo.json e epoca_custo_dentro_de_1pct = 331 contra epoca_pesos_estabilizam_1e-6 = 45.776, cuja razao e 138, e esse par compara criterios de parada, nao um cenario com e sem colinearidade.
- *Correção:* Trocado por 'e a que faz o Gradient Descent precisar de 45.776 epocas para estabilizar os pesos, contra 331 para estabilizar o custo', que cita so valores presentes no resumo.json.
