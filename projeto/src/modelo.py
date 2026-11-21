"""
modelo.py - Arquitetura Transformer para classificação de batimentos e rotinas de treino/avaliação

Projeto: Diagnóstico Automático de Eletrocardiogramas com Transformers
Disciplina: Inteligência Artificial - 7º semestre CC, Turma 7N - Universidade Presbiteriana Mackenzie
Professor: Prof. Dr. Ivan Carlos Alcântara de Oliveira

Integrantes:
    Enzo Ponte Gamberi                          - RA 10389931 - 10389931@mackenzista.com.br
    João Guilherme Messias de Oliveira Santos   - RA 10426110 - 10426110@mackenzista.com.br
    Luís Henrique Ribeiro Fernandes             - RA 10420079 - 10420079@mackenzista.com.br

Síntese do conteúdo:
    ECGTransformer: o batimento (256 amostras, 2 canais: sinal normalizado e diferença em relação
    ao batimento mediano do registro) é dividido em "patches" por uma convolução 1D,
    cada patch vira um token, um token [CLS] aprendível é adicionado e um encoder Transformer
    (auto-atenção multi-cabeça) produz a representação do batimento. Essa representação é
    concatenada com um embedding dos atributos de ritmo (RR) antes da camada de classificação.
    Também contém o laço de treino com perda ponderada por classe, early stopping pelo
    F1-macro de validação, aumento de dados simples e a extração dos mapas de atenção.

Histórico de alterações:
    2026-11-19 - João Guilherme - criação do arquivo (N2): arquitetura, treino, avaliação e
                 extração de atenção.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


class ECGTransformer(nn.Module):
    def __init__(self, tam_janela=256, patch=16, passo=8, d_model=64, n_cabecas=4,
                 n_camadas=3, d_ff=128, dropout=0.2, n_rr=5, n_classes=4, usar_rr=True,
                 n_canais=2):
        super().__init__()
        self.usar_rr = usar_rr
        self.n_canais = n_canais
        # Embedding dos patches: convolução com passo < tamanho -> patches sobrepostos
        self.embed = nn.Conv1d(n_canais, d_model, kernel_size=patch, stride=passo)
        n_tokens = (tam_janela - patch) // passo + 1
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.randn(1, n_tokens + 1, d_model) * 0.02)
        camada = nn.TransformerEncoderLayer(d_model, n_cabecas, d_ff, dropout,
                                            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(camada, n_camadas)
        self.norm = nn.LayerNorm(d_model)
        d_saida = d_model
        if usar_rr:
            self.rr_mlp = nn.Sequential(nn.Linear(n_rr, 32), nn.GELU(), nn.Linear(32, 32))
            d_saida += 32
        self.cabeca = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_saida, 64), nn.GELU(),
                                    nn.Linear(64, n_classes))

    def tokens(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x[:, :self.n_canais]
        z = self.embed(x).transpose(1, 2)                        # (B, n_tokens, d)
        cls = self.cls.expand(z.size(0), -1, -1)
        return torch.cat([cls, z], dim=1) + self.pos

    def forward(self, x, rr):
        h = self.norm(self.encoder(self.tokens(x)))[:, 0]       # saída do token [CLS]
        if self.usar_rr:
            h = torch.cat([h, self.rr_mlp(rr)], dim=1)
        return self.cabeca(h)

    @torch.no_grad()
    def atencao_cls(self, x):
        """
        Atenção do token [CLS] sobre os patches, média entre cabeças, na última camada.
        É uma forma simples de explicabilidade: mostra quais trechos do batimento pesaram
        mais na representação usada pela classificação.
        """
        h = self.tokens(x)
        camadas = self.encoder.layers
        for camada in camadas[:-1]:
            h = camada(h)
        ultima = camadas[-1]
        hn = ultima.norm1(h)
        _, pesos = ultima.self_attn(hn, hn, hn, need_weights=True, average_attn_weights=True)
        return pesos[:, 0, 1:]                                   # (B, n_tokens)


def aumentar(x):
    """Aumento de dados leve: escala de amplitude, ruído gaussiano e pequeno deslocamento."""
    b = x.size(0)
    escala = 1 + 0.1 * torch.randn(b, *([1] * (x.dim() - 1)), device=x.device)
    x = x * escala + 0.05 * torch.randn_like(x)
    desloc = int(torch.randint(-5, 6, (1,)).item())
    return torch.roll(x, desloc, dims=-1)


def pesos_classe(y, n_classes=4, potencia=0.5):
    """Peso inversamente proporcional à frequência, suavizado pela raiz (potencia=0.5)."""
    cont = np.bincount(y, minlength=n_classes).astype(float)
    w = (cont.sum() / np.maximum(cont, 1)) ** potencia
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def prever(modelo, X, R, disp, lote=1024):
    modelo.eval()
    saidas = []
    for i in range(0, len(X), lote):
        xb = torch.from_numpy(X[i:i + lote]).to(disp)
        rb = torch.from_numpy(R[i:i + lote]).to(disp)
        saidas.append(torch.softmax(modelo(xb, rb), dim=1).cpu().numpy())
    return np.concatenate(saidas)


def treinar(modelo, treino, valid, disp, epocas=30, lote=256, lr=1e-3, paciencia=8,
            potencia_peso=0.5, semente=42, verbose=True):
    """
    Treina com AdamW + cosine annealing e guarda os pesos da época com maior F1-macro
    na validação. Retorna o modelo restaurado e o histórico por época.
    """
    torch.manual_seed(semente)
    np.random.seed(semente)
    Xtr, Rtr, ytr = treino
    Xva, Rva, yva = valid
    modelo.to(disp)
    crit = nn.CrossEntropyLoss(weight=pesos_classe(ytr, potencia=potencia_peso).to(disp))
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epocas)
    Xt, Rt, yt = (torch.from_numpy(Xtr), torch.from_numpy(Rtr), torch.from_numpy(ytr).long())

    melhor, melhor_estado, sem_melhora, hist = -1, None, 0, []
    for ep in range(epocas):
        modelo.train()
        perm = torch.randperm(len(yt))
        perda_total = 0.0
        for i in range(0, len(perm), lote):
            idx = perm[i:i + lote]
            xb, rb, yb = Xt[idx].to(disp), Rt[idx].to(disp), yt[idx].to(disp)
            xb = aumentar(xb)
            opt.zero_grad()
            perda = crit(modelo(xb, rb), yb)
            perda.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            perda_total += perda.item() * len(idx)
        sched.step()
        pred_va = prever(modelo, Xva, Rva, disp).argmax(1)
        f1 = f1_score(yva, pred_va, average='macro')
        hist.append({'epoca': ep + 1, 'perda_treino': perda_total / len(yt), 'f1_macro_valid': f1})
        if verbose:
            print(f'época {ep + 1:2d}  perda={perda_total / len(yt):.4f}  F1-macro valid={f1:.4f}')
        if f1 > melhor:
            melhor, melhor_estado, sem_melhora = f1, copy.deepcopy(modelo.state_dict()), 0
        else:
            sem_melhora += 1
            if sem_melhora >= paciencia:
                if verbose:
                    print(f'early stopping na época {ep + 1}')
                break
    modelo.load_state_dict(melhor_estado)
    return modelo, hist
