# Dataset: MIT-BIH Arrhythmia Database

Fonte: [PhysioNet, MIT-BIH Arrhythmia Database v1.0.0](https://physionet.org/content/mitdb/1.0.0/)
(Moody e Mark, 2001; Goldberger et al., 2000). Licença Open Data Commons Attribution v1.0. Os
registros já são distribuídos anonimizados: o cabeçalho traz só idade e sexo, sem nome, data ou
outro dado que identifique o paciente.

## Conteúdo

São 48 registros de ECG ambulatorial (Holter) de 47 pacientes, gravados entre 1975 e 1979 no
laboratório de arritmias do Beth Israel Hospital, em Boston. Cada registro tem cerca de 30 minutos
em dois canais, amostrados a 360 Hz com resolução de 11 bits, e cada batimento foi anotado por pelo
menos dois cardiologistas. São cerca de 110 mil anotações no total.

## Como o projeto usa a base

| Decisão | Motivo |
|---|---|
| Exclui os registros 102, 104, 107 e 217 | Pacientes com marcapasso; recomendação da norma AAMI EC57 |
| Usa só a derivação MLII | Presente nos 44 registros restantes (no 114 está no 2º canal) |
| Agrupa os símbolos nas classes AAMI N, S, V, F | Padrão da literatura; Q fica de fora (só 15 batimentos) |
| Divide por paciente: DS1 (treino) e DS2 (teste) | Protocolo de de Chazal et al. (2004); nenhum paciente aparece nos dois |

Mapeamento dos símbolos do MIT-BIH para as classes AAMI:

| Classe | Significado | Símbolos |
|---|---|---|
| N | Normal, bloqueios de ramo e escapes | N, L, R, e, j |
| S | Ectópico supraventricular | A, a, J, S |
| V | Ectópico ventricular | V, E |
| F | Fusão de ventricular com normal | F |
| Q | Desconhecido / estimulado (descartado) | /, f, Q |

Registros de cada conjunto:

- **DS1 (treino):** 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207,
  208, 209, 215, 220, 223, 230
- **DS2 (teste):** 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221,
  222, 228, 231, 232, 233, 234

Batimentos usados depois da segmentação (alguns poucos nas bordas dos registros são descartados):

| Classe | DS1 (treino) | DS2 (teste) |
|---|---|---|
| N | 45.848 | 44.240 |
| S | 944 | 1.837 |
| V | 3.788 | 3.220 |
| F | 414 | 388 |
| **Total** | **50.994** | **49.685** |

As classes raras dependem de poucos pacientes: o registro 232 tem 75% dos batimentos S do DS2, e o
208 tem 90% dos batimentos F do DS1. Isso pesa nos resultados e é discutido no artigo e no notebook 02.

## Pré-processamento (`src/ecg_utils.py`)

1. Filtro passa-faixa Butterworth de 3ª ordem, 0,5-45 Hz, fase zero (`filtfilt`).
2. Janela de 256 amostras em torno do pico R anotado (90 antes, 166 depois), normalizada por z-score.
3. Cinco atributos de ritmo por batimento: RR anterior, RR posterior e RR médio dos últimos 10
   batimentos (os três divididos pela mediana do RR do registro), RR anterior / RR local e
   RR posterior / RR anterior.
4. Segundo canal: diferença entre o batimento e o batimento mediano do mesmo registro (calculado sem
   rótulos).

## Como obter os dados

Os arquivos brutos (`raw/mitdb/`, cerca de 100 MB) e os arrays processados (`processed/*.npz`)
**não ficam no GitHub**. São baixados e gerados pelo notebook
`notebooks/01_analise_exploratoria_preparacao_dados.ipynb`. Para baixar só os brutos:

```python
import wfdb
wfdb.dl_database('mitdb', dl_dir='dataset/raw/mitdb', annotators=['atr'])
```

## Arquivos versionados em `processed/`

- `registros_utilizados.csv`: os 48 registros com conjunto, idade, sexo, canais, duração e nº de anotações.
- `distribuicao_classes.csv`: batimentos por classe AAMI em DS1 e DS2.
- Figuras da análise exploratória: `perfil_pacientes.png`, `exemplo_sinal_bruto.png`,
  `distribuicao_classes.png`, `classes_por_registro.png`, `exemplo_batimentos_por_classe.png`,
  `atributos_rr.png`, `canal_relativo.png`.
