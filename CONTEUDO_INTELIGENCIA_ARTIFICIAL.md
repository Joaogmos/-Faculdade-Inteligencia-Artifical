# Inteligência Artificial — Mapa completo do conteúdo da disciplina

**Universidade Presbiteriana Mackenzie — Faculdade de Computação e Informática — 7º semestre CC (2026/2)**
**Professor:** Prof. Dr. Ivan Carlos Alcântara de Oliveira

Este documento resume, **em ordem de aula**, os conteúdos presentes nos PDFs/slides da pasta `conteudo/` e nas atividades já feitas (`Atividade 1/`, `atividade2/`, `atividade4/` a `atividade7/`). Não é um resumo detalhado: serve de índice para estudo (ex.: prova P1/P2) e indica onde cada tema aparece.

---

## 0. Plano de ensino (`00_IA_PlanoDeEnsino_7N.pdf`)

- **Ementa:** IA clássica (algoritmos genéticos e de busca, sistemas especialistas e baseados em regras, agentes inteligentes); IA moderna (Redes Neurais e Deep Learning e suas arquiteturas); Processamento de Linguagem Natural e Visão Computacional; frameworks Scikit-learn, TensorFlow e PyTorch.
- **ODS relacionados:** ODS 2 (agricultura), ODS 3 (saúde), ODS 9 (indústria, inovação e infraestrutura).
- **Conteúdo programático completo:**
  1. Introdução à IA: história, aplicações, ética e IA responsável.
  2. Métodos de busca e algoritmos evolucionários (busca; algoritmos genéticos).
  3. Agentes e representação de conhecimento (agentes baseados em conhecimento, jogos, sistemas especialistas, agentes inteligentes, ontologia).
  4. Aprendizado de Máquina clássico (supervisionado; estimadores do Scikit-learn; exercício de classificação).
  5. Redes Neurais: conceitos, arquiteturas, aplicações; Perceptron multicamadas (Scikit-learn); PyTorch/TensorFlow e overfitting.
  6. Visão Computacional: CNNs e arquiteturas; redes pré-treinadas e transfer learning.
  7. PLN: representação de texto (BoW/TF-IDF, NLTK/SpaCy), embeddings (Word2Vec), redes recorrentes.
  8. Grandes modelos de linguagem: LLMs, chatbots e IA generativa.
- **Avaliação:** N1 = 0,7·P1 + 0,3·ATIV1; N2 = 0,7·P2 + 0,3·ATIV2; ATIV = 0,5·Exercícios + 0,5·Projeto; MF = (N1 + N2)/2. Projeto em três partes (Parte 1 proposta; Parte 2 no N1; Parte 3 no N2).
- **Bibliografia:** Aggarwal (*AI: A Textbook*), Chollet (*Deep Learning with Python*), Géron (*Hands-On ML*), Microsoft *AI for Beginners*; complementar: Goodfellow/Bengio/Courville, Russell & Norvig (AIMA); adicional: Faceli et al.

> Os tópicos 5 a 8 (redes neurais, CNN, PLN, LLMs) constam do plano, mas ainda **não têm material nem atividade** na pasta.

---

## 1. Introdução à IA (`01_IA_AulaI_Introducao_vFinal*.pdf`)

- Etimologia: inteligência (*inter + legere*, escolher) e artificial (feito pelo homem).
- Definições de IA em ordem cronológica (Bellman etc.) e as **quatro categorias** de Russell & Norvig: pensar como humano, agir como humano, pensar racionalmente, agir racionalmente.
- Inteligência humana × inteligência artificial; o jogo Go como marco.
- Relação **IA ⊃ Machine Learning ⊃ Deep Learning ⊃ IA Generativa**; definição e tipos de IA; motivos do avanço recente (dados, poder computacional, algoritmos).
- Conceitos introdutórios de Machine Learning e de Deep Learning.

## 2. Ética e Responsabilidade (`01_IA_EticaEResponsabilidade_vFinal.pdf`)

- O que é ética (Aristóteles, eudaimonia) e casos motivadores (redes de agentes de IA, terapia por IA, chatbots).
- Sete eixos da IA responsável:
  1. **Transparência e explicabilidade** — caixas-pretas, XAI, direito à explicação (GDPR).
  2. **Justiça, imparcialidade e mitigação de viés** — discriminação algorítmica, fairness, auditorias.
  3. **Privacidade e proteção de dados** — LGPD, GDPR, anonimização e consentimento.
  4. **Segurança e robustez** — ataques adversariais, adversarial training, detecção de anomalias, dropout/batch norm.
  5. **Impacto social e responsabilidade** — desemprego tecnológico, humano no circuito em decisões críticas.
  6. **Governança e regulamentação.**
  7. **Sustentabilidade.**
- Metodologia de verificação de DeepFake e Fake News; atividade em grupo sobre o tema.
- Leituras extras: `ai-2027.pdf` (cenário de futuro de superinteligência do AI Futures Project: agentes, corrida pela AGI, alinhamento) como provocação sobre riscos e governança.

## 3. Ferramentas: Python, Anaconda e Colab (`03_IA_pythonAnaconda_IntroducaoInstalacao_v2.pdf`)

- Instalação do Anaconda, Anaconda Navigator, Spyder, Jupyter Notebook (`localhost:8888`), JupyterLab, e alternativa **Google Colab** (planos pagos, uso na nuvem).

## 4. Agentes Inteligentes (`03_IA_AgentesInteligentes_aula_*.pdf`, `03_IA_EAD_orientacoes_...`)

- Definição de agente (sensores → percepção → atuadores → ação), diferença entre "não agente" e agente; exemplos: aspirador de pó (mundo de duas salas), Amazon Echo, motorista de táxi.
- **PEAS** (Performance, Environment, Actuators, Sensors).
- **Propriedades do ambiente:** observabilidade (total/parcial); agente único/multiagente; determinístico/estocástico/não determinístico; episódico/sequencial; estático/dinâmico/semidinâmico; discreto/contínuo; conhecido/desconhecido.
- Exemplo aplicado: mineração de argumentos em redes sociais.
- Metodologia de desenvolvimento; PEAS formal × IA agêntica moderna.
- **Tipos de programas de agente:** reativo simples e reativo baseado em modelo (o restante vem na aula 4).
- **Atividade 1** (`Atividade 1/`, individual): Robô Gato no Parque do Carmo — agente reativo simples com `agentes.py` (AIMA), relatório com PEAS, classificação do ambiente e print de execução.

## 5. Agentes baseados em objetivos e em utilidade + Métodos de busca (`04_IA_AgentesInteligentes_ARPeMetodosDeBusca_*.pdf` e `.pptx`)

- Revisão dos tipos de agente; **agentes de resolução de problemas** (baseados em objetivos): exemplo da viagem à Romênia; componentes e formulação de um problema de busca (estado inicial, ações, modelo de transição, teste de objetivo, custo de caminho).
- **Busca com informação (heurística):**
  - **Busca Gulosa pela Melhor Escolha** — usa apenas h(n); rastreio passo a passo com heap (Arad → Bucareste).
  - **A\*** — f(n) = g(n) + h(n); rastreio completo na Romênia; **heurística admissível** (nunca superestima; ex.: distância em linha reta) e **consistente**; A\* é ótimo com heurística admissível.
  - Funções heurísticas (exemplo do ENADE 2011).
- **Agentes baseados em utilidade:** função de utilidade, diferença para agente baseado em objetivos, exemplo de mineração de argumentos.
- **Atividade 4** (`atividade4/`, grupo): A\* no mapa do Brasil com `search.py`/`utils.py` (AIMA), `UndirectedGraph` + `GraphProblem`, 20 cidades do enunciado + 3 acrescentadas (Fortaleza, Vitória, Goiânia).

## 6. Agentes lógicos / baseados em conhecimento (`05_IA_AgentesLogicos_*.pdf`, `05_IA_EAD_orientacoes_agentesLogicos.pdf`)

- Estudo do conhecimento: características do conhecimento e da sua representação.
- **Categorias de raciocínio:** dedução, indução, abdução, analogia.
- Componentes do agente KB: base de conhecimento (KB), **TELL/ASK**, algoritmo genérico `AGENTE-KB`; ciclo de vida (aquisição, formalização, implementação, refinamento).
- **Lógica proposicional:** sintaxe (gramática e precedência), semântica (tabelas-verdade), KB e consequência lógica (KB ⊨ α), **regras de equivalência** (comutatividade, De Morgan, contraposição, condicional etc.) e **regras de inferência** (Modus Ponens, Modus Tollens, silogismo hipotético/disjuntivo, conjunção, simplificação, adição...), exemplos de prova de argumentos; problemas da lógica proposicional (Mundo de Wumpus).
- **Lógica de primeira ordem** (quantificadores, predicados) e **PROLOG** (`..._PROLOG.pdf`).
- **Ontologia:** conceito, componentes, RDF, OWL (`..._Ontologia.pdf`).
- Tecnologias para desenvolvimento de cada tipo de agente (reativo simples, baseado em modelo, objetivos, utilidade, com aprendizagem) e plataformas/infraestrutura (`..._tecnologiasParaDesenvolvimento.pdf`).
- **Atividade 5** (`atividade5/`, grupo): ontologia do Smart Campus Mackenzie — classes/subclasses, propriedades de objeto e de dados, axiomas e instâncias (documento `.docx`, sem código).

## 7. Paradigmas do raciocínio, Sistemas Especialistas e Computação Evolutiva (`06_IA_EAD_SistemasEspecialistas...vFinal.pdf`, `06_..._orientacoes_...`)

- **Paradigmas:** conexionista (redes neurais, aprender a função entrada-saída) × simbólico (hipótese dos sistemas de símbolos físicos).
- **Sistemas Especialistas:** origem nos anos 70; três componentes — **base de conhecimento** (regras se-então), **motor de inferência** (*forward chaining*, guiado por dados; *backward chaining*, guiado por objetivos) e **interface**; memória de trabalho e módulo de explicação; exemplo de diagnóstico de carro que não liga; exemplo do "Coronel West" em lógica de primeira ordem; sistemas especialistas hoje (DXplain, SHINE, HYPO) e aplicações (medicina, engenharia, finanças, educação, agricultura).
- **Computação Evolutiva e Algoritmos Genéticos:** história (Turing, John Holland), motivação (espaços grandes, multimodais); conceitos de **população, cromossomo, gene, fitness, seleção, crossover, mutação**; taxas de crossover/mutação; pseudocódigo e fluxograma do AG; AGs na atualidade.
- **Exemplos práticos** (`06_IA_EAD_AlgoritmosGeneticos_Exemplos.ipynb`): (1) evolução de sequência de ações de um robô; (2) otimização de hiperparâmetros de rede neural; (3) problema da mochila; (4) caixeiro viajante (TSP); (5) maximização de função quadrática.

## 8. Projeto da disciplina (`atividade2/`, `atividade7/`, PDF `01_IA_EAD_projeto_20262_7o_vFinal.pdf`)

- Projeto em grupo aplicando IA a um problema real e alinhado aos ODS, com dados reais (Kaggle, empresa ou coletados).
- **Parte 1 — Proposta** (`atividade2/`): documento de informações da proposta + rubrica.
- **Parte 2 — N1** (`atividade7/`): artigo parcial no **template SBC** (título, resumo, introdução, fundamentação teórica/trabalhos relacionados, metodologia e dataset, resultados parciais, ética, GitHub, referências), análise exploratória e preparação de dados em notebook. Tema do grupo: **diagnóstico de ECG com Transformers** sobre o MIT-BIH Arrhythmia Database (20 registros na entrega; classes desbalanceadas; segmentação de batimentos).
- **Parte 3 — N2:** desenvolvimento completo, artigo final, GitHub público e vídeo de até 5 minutos.

## 9. Aprendizado de Máquina — Parte I: conceitos e regressão (`IA_EAD_MachineLearning_parteI_...Completo_vFinal__2.pdf`, `IA_Presencial_MachineLearning_parteI_..._sintese...pdf`, `07_IA_EAD_orientacoes_...`)

- **Conceitos:** indução de hipóteses, atributos de entrada e de saída/alvo, **overfitting × underfitting**.
- **Tipos de aprendizado:** supervisionado (classificação e regressão), **não supervisionado** (agrupamento, associação, redução de dimensionalidade/PCA), semisupervisionado.
- **Ciclo de vida de Data Science — CRISP-DM.**
- Modelagem supervisionada: modelo preditivo, processo de aprendizagem, treino e teste, elementos do processo.
- **Regressão linear simples e múltipla:** fórmula, interpretação dos coeficientes, exemplos (vendas de carros, grilos e temperatura, preço de veículos, imóveis de São Paulo), armadilhas, regras gerais, **multicolinearidade** e critério para remover variáveis, 5 erros comuns.
- **Métricas de erro:** MAE, MSE, RMSE, **SST, SSR, SSE e R²**, com exemplos numéricos.
- **Python:** `statsmodels` (`OLS`, `summary`) e **scikit-learn** em 8 passos (imports, carga, preparação, treino/teste ~65/35, escolha do modelo, treino, avaliação, previsão).
- **Atividade 6** (`atividade6/`, grupo): regressão linear múltipla com statsmodels/scikit-learn sobre `imoveis_SCS_depoislimpeza.csv` (imóveis de Santo André/São Caetano — base do professor), notebook `IA_EAD_Atividade_modeling_statsModel_vFinal.ipynb` + `.docx` de resultados.

## 10. Aprendizado de Máquina — Parte II: classificação (`IA_EAD_ML_supervisionadoClassificacao_RL_KNN_20262_vFinal.pdf`, sínteses, `08_IA_EAD_Orientacoes_...`)

- Revisão de ciclo de vida, indução de hipóteses e **dilema viés × variância** (underfitting/overfitting) e como reduzir overfitting com conjuntos de treino/teste.
- **Regressão Logística:** função sigmoide, estimação de parâmetros, versões multinomial e multiclasse/multirrótulo (exemplo das flores Iris: setosa, versicolor, virginica).
- **k-NN:** funcionamento, métricas de distância (euclidiana, Manhattan...), escolha do k, **normalização** dos atributos; exemplo "jogo da velha/tempo".
- Passos para gerar um modelo preditivo em Python (8 passos, treino/teste 20–35%, acurácia, prever novos casos).
- Complementos de Python para análise de dados: PyPI, PyData stack, **NumPy, Pandas, Matplotlib, Seaborn** (Partes I e II) e **Scikit-learn** (Parte III).

## 11. Avaliação de modelos preditivos (`IA_EAD_avaliacaoDosModelosPreditivos_20262.pdf`)

- Divisão treino (70–80%) / teste; taxa de erro e **acurácia**.
- **Matriz de confusão** (VP, VN, FP, FN) e medidas para duas classes: taxa de acerto, **precisão** positiva/negativa, **sensibilidade (recall)**, especificidade, **medida-F / F1**; cálculo em Python; matriz de confusão para *n* categorias.
- Exemplo aplicado: diagnóstico de nódulo (câncer de mama, base Mammographic Mass da UCI).

## 12. Aula VIII (orientações apenas — sem pasta de atividade ainda)

- Atividade em grupo: **classificação de óbito por COVID-19** com **k-NN e Regressão Logística** (notebook `IA_EAD_COVID19_...ipynb`, `.docx` de resultados, artigo de referência sobre classificação de COVID-19 com métodos de ML), usando as métricas da seção 11.

---

## Quadro-resumo das atividades

| Nº | Pasta | Tipo | Tema | Conceitos cobertos |
|----|-------|------|------|--------------------|
| 1 | `Atividade 1/` | Individual | Robô Gato (Parque do Carmo) | Agente reativo simples, PEAS, `agentes.py` |
| 2 | `atividade2/` | Grupo | Projeto — Parte 1 (proposta) | Definição do problema, ODS |
| 4 | `atividade4/` | Grupo | A\* no mapa do Brasil | Busca heurística, grafos, `search.py` |
| 5 | `atividade5/` | Grupo | Ontologia Smart Campus | Representação de conhecimento, ontologia |
| 6 | `atividade6/` | Grupo | Regressão linear múltipla (imóveis) | ML supervisionado, statsmodels, R², multicolinearidade |
| 7 | `atividade7/` | Grupo | Projeto — Parte 2 (ECG com Transformers) | Análise exploratória, artigo SBC, ética |
| 8 | *(a criar)* | Grupo | Classificação de óbito por COVID-19 | k-NN, regressão logística, métricas |

*(Não há Atividade 3 nas pastas; a aula 3 corresponde à Atividade 1 de Agentes Inteligentes.)*

## Roteiro de estudo sugerido

1. Introdução, IA × ML × DL, ética (seções 1–2).
2. Agentes, PEAS, tipos de ambiente e de agente (4–5).
3. Busca heurística: gulosa e A\* (5).
4. Lógica proposicional e de primeira ordem, ontologia, sistemas especialistas (6–7).
5. Algoritmos genéticos (7).
6. ML: overfitting, regressão linear e R² (9); regressão logística, k-NN e matriz de confusão (10–11).
7. Pendentes no plano: redes neurais, CNN, PLN, LLMs.
