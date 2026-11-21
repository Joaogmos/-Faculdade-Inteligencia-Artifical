"""
demo.py - Demonstração do classificador: analisa um registro do MIT-BIH com o modelo treinado

Projeto: Diagnóstico Automático de Eletrocardiogramas com Transformers
Disciplina: Inteligência Artificial - 7º semestre CC, Turma 7N - Universidade Presbiteriana Mackenzie
Professor: Prof. Dr. Ivan Carlos Alcântara de Oliveira

Integrantes:
    Enzo Ponte Gamberi                          - RA 10389931 - 10389931@mackenzista.com.br
    João Guilherme Messias de Oliveira Santos   - RA 10426110 - 10426110@mackenzista.com.br
    Luís Henrique Ribeiro Fernandes             - RA 10420079 - 10420079@mackenzista.com.br

Síntese do conteúdo:
    Carrega o modelo salvo em modelos/ecg_transformer.pt, processa um registro inteiro (filtro,
    segmentação, atributos RR, molde mediano do paciente), classifica todos os batimentos e
    mostra: (1) resumo por classe, real x previsto; (2) um trecho do ECG com o rótulo do
    cardiologista e o do modelo sobre cada batimento, salvo em resultados/demo_<registro>.png.
    Pensado para os registros do DS2, que o modelo não viu no treino.

    Uso (a partir da pasta do projeto):
        python src/demo.py 208            # registro do DS1 (visto no treino)
        python src/demo.py 213 --inicio 300 --segundos 12

Histórico de alterações:
    2026-11-19 - João Guilherme - criação do script (N2).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import ecg_utils as eu
from modelo import ECGTransformer, prever

RAIZ = Path(__file__).resolve().parent.parent
CORES = {'N': '#4a6fa5', 'S': '#e39b3b', 'V': '#c0504d', 'F': '#7a5195'}


def main():
    ap = argparse.ArgumentParser(description='Classifica os batimentos de um registro do MIT-BIH.')
    ap.add_argument('registro', help='número do registro, ex.: 213')
    ap.add_argument('--inicio', type=float, default=None, help='início do trecho plotado (s)')
    ap.add_argument('--segundos', type=float, default=10, help='duração do trecho plotado (s)')
    args = ap.parse_args()
    r = args.registro

    pasta = eu.baixar_registros(RAIZ / 'dataset' / 'raw' / 'mitdb', [r])
    X, R, y, pos = eu.segmentar_registro(pasta, r)
    Xc = eu.adicionar_canal_relativo(X, np.array([r] * len(X)))

    disp = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(RAIZ / 'modelos' / 'ecg_transformer.pt', map_location=disp)
    modelo = ECGTransformer(**ckpt['config']).to(disp)
    modelo.load_state_dict(ckpt['estado'])
    prob = prever(modelo, Xc, R, disp)
    pred = np.array(eu.CLASSES)[prob.argmax(1)]

    conjunto = 'DS2 - paciente não visto no treino' if r in eu.DS2 else 'DS1 - paciente usado no treino'
    print(f'\nRegistro {r} ({conjunto}): {len(y)} batimentos analisados')
    print(f'concordância com o cardiologista: {100 * (pred == y).mean():.1f}%\n')
    tab = pd.crosstab(pd.Series(y, name='cardiologista'), pd.Series(pred, name='modelo'))
    print(tab.reindex(index=[c for c in eu.CLASSES if c in tab.index],
                      columns=[c for c in eu.CLASSES if c in tab.columns], fill_value=0))

    # Trecho a plotar: se não for informado, começa um pouco antes do primeiro batimento não-N
    if args.inicio is None:
        anormais = np.where(y != 'N')[0]
        args.inicio = max(0, pos[anormais[0]] / eu.FS - 2) if len(anormais) else 0
    _, sinal, _ = eu.ler_registro(pasta, r)
    sinal = eu.filtrar(sinal)
    a, b = int(args.inicio * eu.FS), int((args.inicio + args.segundos) * eu.FS)
    t = np.arange(a, b) / eu.FS

    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.plot(t, sinal[a:b], color='black', lw=0.8)
    topo = sinal[a:b].max()
    for p, real, prev, pr in zip(pos, y, pred, prob.max(1)):
        if a <= p < b:
            cor = CORES[prev]
            ax.axvspan((p - 30) / eu.FS, (p + 30) / eu.FS, color=cor, alpha=0.18)
            marca = '' if real == prev else ' ✗'
            ax.text(p / eu.FS, topo * 1.08, f'{prev}{marca}', ha='center', color=cor, fontsize=10,
                    fontweight='bold')
            ax.text(p / eu.FS, topo * 1.22, real, ha='center', color='gray', fontsize=8)
    ax.set_ylim(top=topo * 1.35)
    ax.set_xlabel('tempo (s)'); ax.set_ylabel('mV (MLII filtrado)')
    ax.set_title(f'Registro {r} ({conjunto})\nrótulo de cima (cinza): cardiologista;  '
                 f'de baixo (cor): modelo;  ✗ = discordância', fontsize=10)
    plt.tight_layout()
    saida = RAIZ / 'resultados' / f'demo_{r}.png'
    plt.savefig(saida, dpi=130, bbox_inches='tight')
    print(f'\nfigura salva em {saida}')
    plt.show()


if __name__ == '__main__':
    main()
