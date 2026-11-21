# Diagnóstico Automático de Eletrocardiogramas com Transformers

Projeto da disciplina de Inteligência Artificial (7º semestre de Ciência da Computação, turma 7N,
2026/2), Universidade Presbiteriana Mackenzie, Faculdade de Computação e Informática.
Professor: Prof. Dr. Ivan Carlos Alcântara de Oliveira.

Esta pasta é a **Parte 3 (N2)**, a entrega final. A Parte 2 (N1), com a análise exploratória
inicial e o artigo parcial, está em [`../atividade7`](../atividade7).

## Integrantes

| Nome | RA | E-mail |
|---|---|---|
| Enzo Ponte Gamberi | 10389931 | 10389931@mackenzista.com.br |
| João Guilherme Messias de Oliveira Santos | 10426110 | 10426110@mackenzista.com.br |
| Luís Henrique Ribeiro Fernandes | 10420079 | 10420079@mackenzista.com.br |

## O projeto em um parágrafo

Um classificador de batimentos cardíacos que recebe o ECG (derivação MLII) e marca cada batimento
como normal (N), supraventricular (S), ventricular (V) ou fusão (F), seguindo a norma AAMI EC57. O
modelo é um Transformer pequeno (112 mil parâmetros) que recebe a forma do batimento dividida em
trechos, a diferença entre o batimento e o batimento típico do mesmo paciente, e cinco atributos de
ritmo. O treino usa 22 pacientes do MIT-BIH Arrhythmia Database e o teste usa outros 22 que o modelo
nunca viu (divisão DS1/DS2 de de Chazal et al., 2004). Os resultados estão em
[`resultados/`](resultados) e no artigo.

Relação com os ODS: objetivo 3 (Saúde e Bem-Estar), meta de reduzir a mortalidade por doenças não
transmissíveis, entre elas as cardiovasculares.

## Links

- Artigo final: [`artigo/artigo_final_projeto_IA.pdf`](artigo/artigo_final_projeto_IA.pdf)
- Vídeo de apresentação (YouTube): _link a inserir após a gravação_
- Rubrica: [`projetoDeIA_parte3_rubrica.docx`](projetoDeIA_parte3_rubrica.docx)

## Estrutura

```
projeto/
├── README.md                    este arquivo
├── requirements.txt             dependências com as versões usadas
├── artigo/                      artigo final no template SBC (.docx e .pdf)
├── dataset/
│   ├── README.md                origem, licença, classes, divisão DS1/DS2, pré-processamento
│   ├── raw/mitdb/               registros brutos (baixados pelo notebook 01, fora do git)
│   └── processed/               tabelas e figuras da análise exploratória (.npz fora do git)
├── notebooks/
│   ├── 01_analise_exploratoria_preparacao_dados.ipynb
│   └── 02_modelagem_treinamento_avaliacao.ipynb
├── src/
│   ├── ecg_utils.py             leitura, filtro, segmentação, atributos RR, classes AAMI, DS1/DS2
│   ├── modelo.py                arquitetura Transformer, treino, predição, mapas de atenção
│   ├── demo.py                  classifica um registro inteiro e plota o resultado
│   └── figura_arquitetura.py    gera o diagrama da arquitetura usado no artigo
├── modelos/ecg_transformer.pt   pesos do modelo treinado (semente 0, 450 KB)
├── resultados/                  métricas (.csv) e figuras dos experimentos
├── video/roteiro_video.md       roteiro do vídeo de apresentação
└── projetoDeIA_parte3_rubrica.docx
```

Todo arquivo de código começa com um cabeçalho com os integrantes, a síntese do conteúdo e o
histórico de alterações (data, autor, descrição).

## Como reproduzir

Testado com Python 3.13 no Windows 11, com GPU NVIDIA RTX 4060. Sem GPU tudo roda na CPU, só que o
notebook 02 fica bem mais lento.

```bash
cd projeto
pip install torch --index-url https://download.pytorch.org/whl/cu124   # ou: pip install torch (CPU)
pip install -r requirements.txt

# 1) baixa a base (~100 MB), faz a análise exploratória e gera dataset/processed/mitdb_ds1_ds2.npz
jupyter nbconvert --to notebook --execute --inplace notebooks/01_analise_exploratoria_preparacao_dados.ipynb

# 2) treina e avalia todos os modelos (~15 min na RTX 4060), salva modelos/ e resultados/
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/02_modelagem_treinamento_avaliacao.ipynb
```

Os notebooks já estão executados no repositório, com todas as saídas visíveis. Rodar de novo na
GPU pode dar diferenças pequenas na terceira casa decimal, porque algumas operações do cuDNN não
são determinísticas.

## Demonstração

O script `demo.py` usa o modelo salvo para classificar todos os batimentos de um registro e compara
com a anotação do cardiologista:

```bash
python src/demo.py 213                         # paciente do conjunto de teste
python src/demo.py 200 --inicio 120 --segundos 12
```

Ele imprime a tabela cardiologista × modelo e salva em `resultados/demo_<registro>.png` um trecho do
ECG com os dois rótulos sobre cada batimento.

## Resultados principais

Teste no DS2 (22 pacientes que o modelo não viu), média ± desvio de 5 execuções:

| Métrica | Transformer | Random Forest (baseline) |
|---|---|---|
| F1-macro | 0,533 ± 0,017 | 0,522 |
| Acurácia | 92,9 ± 1,1% | 94,6% |
| Sensibilidade / VPP de V | 94,8% / 86,6% | 92,2% / 95,7% |
| Sensibilidade / VPP de S | 20,4% / 31,8% | 11,5% / 39,4% |
| F1 de F | 0,015 | 0,000 |

- Na classe V o modelo supera a referência do protocolo (de Chazal et al., 2004: Se 77,7%, VPP 81,9%).
  Em S fica abaixo (Se 75,9% na referência), e F praticamente não é reconhecida.
- O F1-macro do Transformer fica dentro de um desvio padrão do Random Forest. O ganho maior veio da
  entrada normalizada por paciente: sem o canal relativo ao molde do paciente, o F1-macro cai para
  0,452; sem os atributos RR, a sensibilidade de S cai à metade.
- O registro 232 tem 75% dos batimentos S do teste, e o modelo erra quase todos. Sem ele, a
  sensibilidade de S sobe de 16% para 63%.
- Com divisão aleatória de batimentos (intra-paciente), o mesmo modelo chega a 99,1% de acurácia e
  F1-macro de 0,931. É por isso que esse protocolo não é usado como resultado principal.

Detalhes: `notebooks/02_modelagem_treinamento_avaliacao.ipynb` (seção 9) e seção 4 do artigo.
