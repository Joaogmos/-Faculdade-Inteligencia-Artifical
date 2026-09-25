# Atividade 7 - Projeto da Disciplina, Parte 2 (Primeiro Bimestre / N1)

Diagnóstico automático de eletrocardiogramas com Transformers, a partir do MIT-BIH Arrhythmia
Database. Este é o projeto da disciplina em grupo, aproveitado do TCC de um dos integrantes,
conforme previsto no enunciado (`01_IA_EAD_projeto_20262_7o_vFinal.pdf`).

## Integrantes

- Enzo Ponte Gamberi - RA 10389931
- João Guilherme Messias de Oliveira Santos - RA 10426110
- Luís Henrique Ribeiro Fernandes - RA 10420079

## Conteúdo desta entrega

- `artigo/artigo_parcial_projeto_IA.docx` (e respectivo PDF): artigo parcial no template SBC,
  com todas as seções pedidas (título, resumo, introdução, fundamentação teórica, metodologia,
  resultados parciais, referências).
- `notebooks/01_analise_exploratoria_preparacao_dados.ipynb`: código Python (análise exploratória
  do MIT-BIH Arrhythmia Database e preparação/segmentação dos dados), já executado.
- `dataset/`: descrição do dataset utilizado (`README.md`), metadados dos registros e os
  resultados da análise exploratória (gráficos e tabelas de distribuição de classes).
- `projetoDeIA_parte2_rubrica.docx`: Rubrica Analítica preenchida com título do projeto e
  integrantes (avaliação a preencher pelo professor).

## Como reproduzir

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/01_analise_exploratoria_preparacao_dados.ipynb
```

O notebook baixa automaticamente a amostra de 20 registros do MIT-BIH Arrhythmia Database
usados nesta entrega (ver `dataset/README.md`).

## Próximas etapas (N2)

Ampliar a coleta para os 48 registros completos da base, implementar e treinar a arquitetura
Transformer sobre os batimentos processados, e avaliar o modelo com métricas adequadas ao
desbalanceamento de classes (F1 por classe, matriz de confusão, recall por classe).
