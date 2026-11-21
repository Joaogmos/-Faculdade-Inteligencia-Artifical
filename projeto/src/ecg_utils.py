"""
ecg_utils.py - Funções compartilhadas do pipeline de dados de ECG (MIT-BIH Arrhythmia Database)

Projeto: Diagnóstico Automático de Eletrocardiogramas com Transformers
Disciplina: Inteligência Artificial - 7º semestre CC, Turma 7N - Universidade Presbiteriana Mackenzie
Professor: Prof. Dr. Ivan Carlos Alcântara de Oliveira

Integrantes:
    Enzo Ponte Gamberi                          - RA 10389931 - 10389931@mackenzista.com.br
    João Guilherme Messias de Oliveira Santos   - RA 10426110 - 10426110@mackenzista.com.br
    Luís Henrique Ribeiro Fernandes             - RA 10420079 - 10420079@mackenzista.com.br

Síntese do conteúdo:
    Download dos registros do PhysioNet, leitura da derivação MLII, filtragem passa-faixa,
    segmentação de batimentos em torno do pico R, cálculo de atributos de intervalo RR
    normalizados por paciente, mapeamento dos códigos MIT-BIH para as classes AAMI EC57 e
    definição da divisão inter-paciente DS1/DS2 (de Chazal et al., 2004).
    Usado pelos notebooks 01 e 02 e pelo script demo.py.

Histórico de alterações:
    2026-09-25 - Grupo (Enzo, João, Luís) - pré-processamento original, escrito dentro do notebook
                 de análise exploratória da Parte 2 (N1): filtragem, segmentação, mapeamento AAMI,
                 20 registros e split aleatório estratificado.
    2026-11-19 - João Guilherme - extração para módulo próprio; 48 registros; exclusão dos 4
                 registros com marcapasso; atributos RR; janela assimétrica de 256 amostras;
                 divisão inter-paciente DS1/DS2; canal de morfologia relativa ao molde mediano
                 do registro.
"""

from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt

FS = 360                      # frequência de amostragem do MIT-BIH (Hz)
JANELA_ANTES = 90             # amostras antes do pico R (0,25 s)
JANELA_DEPOIS = 166           # amostras depois do pico R (~0,46 s) -> total 256 amostras
TAM_JANELA = JANELA_ANTES + JANELA_DEPOIS

TODOS_REGISTROS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '111', '112', '113',
    '114', '115', '116', '117', '118', '119', '121', '122', '123', '124', '200', '201', '202',
    '203', '205', '207', '208', '209', '210', '212', '213', '214', '215', '217', '219', '220',
    '221', '222', '223', '228', '230', '231', '232', '233', '234',
]

# Registros com marcapasso: a recomendação AAMI EC57 é excluí-los da avaliação
REGISTROS_MARCAPASSO = ['102', '104', '107', '217']

# Divisão inter-paciente proposta por de Chazal, O'Dwyer e Reilly (2004)
DS1 = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122',
       '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230']
DS2 = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210',
       '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']

# Mapeamento dos símbolos de anotação do MIT-BIH para as classes AAMI EC57
MAPA_AAMI = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q',
}

# A classe Q praticamente some quando os registros com marcapasso são excluídos
# (são ~15 batimentos em 44 registros), então o modelo trabalha com 4 classes.
CLASSES = ['N', 'S', 'V', 'F']
CLASSE_IDX = {c: i for i, c in enumerate(CLASSES)}

NOMES_RR = ['rr_pre', 'rr_pos', 'rr_local', 'rr_pre/rr_local', 'rr_pos/rr_pre']


def baixar_registros(destino, registros=TODOS_REGISTROS):
    """Baixa do PhysioNet os arquivos .dat/.hea/.atr que ainda não existem em `destino`."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    faltando = [r for r in registros if not (destino / f'{r}.dat').exists()]
    if faltando:
        wfdb.dl_database('mitdb', dl_dir=str(destino), records=faltando, annotators=['atr'])
    return destino


def filtrar(sinal, fs=FS, baixa=0.5, alta=45.0, ordem=3):
    """Passa-faixa Butterworth de fase zero: remove deriva de linha de base e ruído de rede/EMG."""
    b, a = butter(ordem, [baixa / (fs / 2), alta / (fs / 2)], btype='band')
    return filtfilt(b, a, sinal)


def ler_registro(pasta, registro):
    """Lê a derivação MLII de um registro e suas anotações de batimento."""
    rec = wfdb.rdrecord(str(Path(pasta) / registro))
    ann = wfdb.rdann(str(Path(pasta) / registro), 'atr')
    # o registro 114 tem os canais invertidos, por isso a busca pelo nome do canal
    canal = rec.sig_name.index('MLII') if 'MLII' in rec.sig_name else 0
    return rec, rec.p_signal[:, canal], ann


def info_paciente(rec):
    """Extrai idade e sexo do comentário do cabeçalho (.hea), quando disponível."""
    if not rec.comments:
        return None, None
    partes = rec.comments[0].split()
    idade = int(partes[0]) if partes and partes[0].isdigit() else None
    sexo = partes[1] if len(partes) > 1 and partes[1] in ('M', 'F') else None
    return idade, sexo


def atributos_rr(picos, fs=FS, n_local=10):
    """
    Atributos de ritmo por batimento, normalizados pela mediana do RR do próprio registro.
    A normalização por paciente é o que permite comparar um batimento "precoce" entre
    pessoas com frequências cardíacas basais diferentes.
    """
    rr = np.diff(picos) / fs
    rr_pre = np.concatenate([[rr[0]], rr])
    rr_pos = np.concatenate([rr, [rr[-1]]])
    rr_local = np.array([rr_pre[max(0, i - n_local):i + 1].mean() for i in range(len(rr_pre))])
    ref = np.median(rr)
    feats = np.stack([
        rr_pre / ref,
        rr_pos / ref,
        rr_local / ref,
        rr_pre / rr_local,
        rr_pos / rr_pre,
    ], axis=1)
    return feats.astype(np.float32)


def segmentar_registro(pasta, registro, incluir_q=False):
    """
    Devolve (X, rr, y, amostras) de um registro:
        X        -> batimentos filtrados e normalizados (z-score), shape (n, 256)
        rr       -> atributos de ritmo, shape (n, 5)
        y        -> classe AAMI (string)
        amostras -> posição do pico R no sinal original
    Os atributos RR são calculados sobre todas as anotações de batimento, antes de descartar
    as classes ignoradas, para não distorcer os intervalos.
    """
    _, sinal, ann = ler_registro(pasta, registro)
    sinal = filtrar(sinal)
    simbolos = np.array(ann.symbol)
    amostras = np.array(ann.sample)
    eh_batimento = np.isin(simbolos, list(MAPA_AAMI.keys()))
    simbolos, amostras = simbolos[eh_batimento], amostras[eh_batimento]
    rr = atributos_rr(amostras)

    X, R, y, pos = [], [], [], []
    for i, (s, p) in enumerate(zip(simbolos, amostras)):
        classe = MAPA_AAMI[s]
        if classe == 'Q' and not incluir_q:
            continue
        ini, fim = p - JANELA_ANTES, p + JANELA_DEPOIS
        if ini < 0 or fim > len(sinal):
            continue
        seg = sinal[ini:fim]
        seg = (seg - seg.mean()) / (seg.std() + 1e-8)
        X.append(seg.astype(np.float32))
        R.append(rr[i])
        y.append(classe)
        pos.append(p)
    return np.array(X), np.array(R), np.array(y), np.array(pos)


def adicionar_canal_relativo(X, origem):
    """
    Monta a entrada de 2 canais do modelo: (0) o batimento normalizado e (1) a diferença entre
    ele e o batimento mediano do próprio registro. O molde mediano é calculado sem rótulos,
    então pode ser obtido também para um paciente novo. É o equivalente, para a forma de onda,
    da normalização do RR pelo ritmo basal do paciente.
    """
    saida = np.empty((len(X), 2, X.shape[1]), dtype=np.float32)
    saida[:, 0] = X
    for r in np.unique(origem):
        k = origem == r
        saida[k, 1] = X[k] - np.median(X[k], axis=0)
    return saida


def montar_conjunto(pasta, registros):
    """Concatena os batimentos de vários registros, guardando o registro de origem de cada um."""
    Xs, Rs, ys, origem = [], [], [], []
    for r in registros:
        X, R, y, _ = segmentar_registro(pasta, r)
        Xs.append(X)
        Rs.append(R)
        ys.append(y)
        origem.extend([r] * len(y))
    y = np.concatenate(ys)
    y_idx = np.array([CLASSE_IDX[c] for c in y])
    return np.concatenate(Xs), np.concatenate(Rs), y_idx, np.array(origem)
