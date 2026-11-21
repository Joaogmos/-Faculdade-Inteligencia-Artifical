"""
figura_arquitetura.py - Gera o diagrama do pipeline e da arquitetura usado no artigo

Projeto: Diagnóstico Automático de Eletrocardiogramas com Transformers
Disciplina: Inteligência Artificial - 7º semestre CC, Turma 7N - Universidade Presbiteriana Mackenzie

Integrantes:
    Enzo Ponte Gamberi                          - RA 10389931 - 10389931@mackenzista.com.br
    João Guilherme Messias de Oliveira Santos   - RA 10426110 - 10426110@mackenzista.com.br
    Luís Henrique Ribeiro Fernandes             - RA 10420079 - 10420079@mackenzista.com.br

Síntese do conteúdo:
    Desenha, com matplotlib, o fluxo sinal -> pré-processamento -> Transformer -> classe e salva
    em resultados/arquitetura.png. Uso: python src/figura_arquitetura.py

Histórico de alterações:
    2026-11-19 - João Guilherme - criação do script (N2).
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RAIZ = Path(__file__).resolve().parent.parent


def caixa(ax, x, y, w, h, texto, cor='#e8eef6', borda='#1f3b63', fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc=cor, ec=borda, lw=1.1))
    ax.text(x + w / 2, y + h / 2, texto, ha='center', va='center', fontsize=fs)


def seta(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.1))


fig, ax = plt.subplots(figsize=(12, 4.6))
ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis('off')

# Linha de cima: pré-processamento
caixa(ax, 0.1, 3.3, 1.7, 0.9, 'ECG MLII\n360 Hz\n(MIT-BIH)', cor='#f2f2f2')
caixa(ax, 2.2, 3.3, 1.8, 0.9, 'Filtro passa-faixa\n0,5-45 Hz')
caixa(ax, 4.4, 3.3, 2.0, 0.9, 'Janela de 256 amostras\nem torno do pico R\nz-score')
caixa(ax, 6.8, 3.3, 2.3, 0.9, 'Canal 2: batimento menos\nmolde mediano do paciente')
caixa(ax, 9.5, 3.3, 2.3, 0.9, '5 atributos RR\nnormalizados pelo\nritmo do paciente')
for x1, x2 in [(1.8, 2.2), (4.0, 4.4), (6.4, 6.8)]:
    seta(ax, x1, 3.75, x2, 3.75)
ax.annotate('', xy=(10.65, 4.2), xytext=(3.1, 4.2),
            arrowprops=dict(arrowstyle='->', color='#999999', lw=0.9, connectionstyle='arc3,rad=-0.12'))
ax.text(6.9, 4.47, 'posições dos picos R', fontsize=7.5, color='#666666', ha='center')

# Linha de baixo: modelo
caixa(ax, 0.1, 0.9, 1.9, 1.3, 'Entrada\n2 canais × 256', cor='#f2f2f2')
caixa(ax, 2.4, 0.9, 1.9, 1.3, 'Conv1D\npatch 16, passo 8\n→ 31 tokens × 64')
caixa(ax, 4.7, 0.9, 1.7, 1.3, '[CLS] +\nembedding\nposicional')
caixa(ax, 6.8, 0.9, 2.1, 1.3, 'Encoder Transformer\n3 camadas\n4 cabeças de atenção', cor='#dfe7f1')
caixa(ax, 9.3, 1.75, 1.2, 0.65, 'MLP RR\n32', fs=8)
caixa(ax, 9.3, 0.55, 1.2, 0.65, 'saída [CLS]\n64', fs=8)
caixa(ax, 10.85, 0.9, 1.05, 1.3, 'Concat\n→ MLP\n→ softmax\nN S V F', cor='#f6e3e1', borda='#8c2d2a', fs=8)
for x1, x2 in [(2.0, 2.4), (4.3, 4.7), (6.4, 6.8)]:
    seta(ax, x1, 1.55, x2, 1.55)
seta(ax, 8.9, 1.3, 9.3, 0.88)
seta(ax, 10.5, 2.05, 10.85, 1.75)
seta(ax, 10.5, 0.88, 10.85, 1.3)
seta(ax, 7.95, 3.3, 1.05, 2.2)
seta(ax, 10.65, 3.3, 9.9, 2.4)

ax.text(0.1, 2.75, 'Pré-processamento (src/ecg_utils.py)', fontsize=9, fontweight='bold', color='#1f3b63')
ax.text(0.1, 0.25, 'Modelo (src/modelo.py) - 112 mil parâmetros', fontsize=9, fontweight='bold', color='#1f3b63')

saida = RAIZ / 'resultados' / 'arquitetura.png'
saida.parent.mkdir(exist_ok=True)
plt.savefig(saida, dpi=200, bbox_inches='tight')
print('salvo em', saida)
