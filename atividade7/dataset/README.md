# Dataset: MIT-BIH Arrhythmia Database

Fonte: [PhysioNet - MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
(Moody & Mark, 2001; Goldberger et al., 2000), licença aberta (Open Data Commons Attribution
License v1.0), sem dados sensíveis ou identificáveis de pacientes.

## Descrição

Base de referência para classificação de arritmias cardíacas, com 48 registros de ECG de
meia hora (dois canais cada, um deles quase sempre a derivação MLII), amostrados a 360 Hz e
anotados batimento a batimento por cardiologistas do BIH Arrhythmia Laboratory.

Nesta entrega parcial (Atividade 7 / N1) foram usados **20 dos 48 registros**: `100, 101, 103,
105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124`. A ampliação
para os 48 registros completos está prevista para a entrega final (N2).

## Como obter os dados

Os arquivos brutos (`.dat`, `.hea`, `.atr`) **não são versionados neste repositório** (são cerca
de 38 MB e totalmente reproduzíveis a partir do PhysioNet). Para baixá-los, execute a primeira
célula de código do notebook `notebooks/01_analise_exploratoria_preparacao_dados.ipynb`, ou
diretamente em Python:

```python
import wfdb
records = ['100','101','103','105','106','108','109','111','112','113',
           '114','115','116','117','118','119','121','122','123','124']
wfdb.dl_database('mitdb', dl_dir='dataset/raw/mitdb', records=records, annotators=['atr'])
```

## Dataset processado

Depois do pré-processamento (filtragem, segmentação por batimento centrado no pico R,
normalização e mapeamento para as classes AAMI), o notebook salva em `dataset/processed/`:

- `X_train.npy`, `X_test.npy`: batimentos segmentados (360 amostras cada, 1s de sinal), separados
  em treino e teste de forma estratificada por classe. Não versionados (arquivos grandes,
  regeneráveis pelo notebook).
- `y_train.npy`, `y_test.npy`: rótulos AAMI (N, S, V, F, Q) correspondentes.
- `registros_utilizados.csv`: metadados de cada registro (frequência de amostragem, duração,
  número de batimentos anotados).
- `distribuicao_classes.csv` e `distribuicao_classes.png`: contagem de batimentos por classe AAMI.
- `exemplo_sinal_bruto.png`, `exemplo_batimentos_por_classe.png`: gráficos da análise exploratória.

## Classes (norma AAMI EC57)

| Classe | Significado | Códigos originais do MIT-BIH agrupados |
|---|---|---|
| N | Normal | N, L, R, e, j |
| S | Supraventricular | A, a, J, S |
| V | Ventricular | V, E |
| F | Fusão | F |
| Q | Não classificável / paced | /, f, Q |

## Distribuição de classes na amostra usada (20 registros)

| Classe | Nº de batimentos | % |
|---|---|---|
| N | 39.579 | 96,4% |
| V | 1.281 | 3,1% |
| S | 192 | 0,5% |
| F | 13 | <0,1% |
| Q | 7 | <0,1% |

O forte desbalanceamento (classe N dominante) é esperado nesse tipo de base e será tratado na
etapa de treinamento do modelo (pesos de classe, reamostragem ou métricas robustas a
desbalanceamento, como F1 por classe).
