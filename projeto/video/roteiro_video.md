# Roteiro do vídeo (máximo 5 minutos)

Sugestão de divisão entre os três integrantes. Os tempos somam 4min40s e deixam folga. Os números
vêm do artigo, seção 4.

## 1. Abertura (0:00-0:25) - Enzo

Tela: slide com título, nomes, professor, disciplina, curso e instituição.

> Olá, nós somos Enzo Ponte Gamberi, João Guilherme Messias de Oliveira Santos e Luís Henrique
> Ribeiro Fernandes, do 7º semestre de Ciência da Computação da Universidade Presbiteriana
> Mackenzie. Este é o projeto da disciplina de Inteligência Artificial, com o professor Ivan Carlos
> Alcântara de Oliveira: "Diagnóstico automático de eletrocardiogramas com Transformers".

## 2. O problema (0:25-1:05) - Enzo

Tela: figura `dataset/processed/exemplo_sinal_bruto.png`.

- O ECG é o exame mais usado para detectar arritmias. Um Holter de 24 horas tem mais de 100 mil
  batimentos, e a leitura manual leva tempo.
- A ideia é classificar cada batimento automaticamente nas quatro classes da norma AAMI: normal,
  supraventricular, ventricular e fusão, como apoio ao médico.
- Relação com o ODS 3 (saúde e bem-estar).

## 3. Dados e IA usada (1:05-2:15) - Luís

Tela: `classes_por_registro.png`, depois `atributos_rr.png`, depois um esquema da arquitetura (pode
ser a Figura de arquitetura do artigo).

- Base MIT-BIH, PhysioNet: 44 pacientes, cerca de 100 mil batimentos, 89% normais.
- Fundamentos da disciplina: aprendizado supervisionado, tratamento de desbalanceamento, avaliação
  com métricas por classe, comparação com um baseline clássico (Random Forest) e um modelo de deep
  learning com mecanismo de atenção (Transformer).
- Como o Transformer vê o batimento: cada batimento é cortado em pedaços, cada pedaço vira um
  token, e a auto-atenção decide quais pedaços importam. A gente soma informação de ritmo (intervalos
  RR) e a diferença do batimento em relação ao batimento típico do próprio paciente.
- Ponto de método: treino com 22 pacientes e teste com outros 22 que o modelo nunca viu.

## 4. Solução funcionando (2:15-3:40) - João

Tela: terminal, na pasta `projeto/`:

```bash
python src/demo.py 213
python src/demo.py 200
```

- No 213 o modelo acerta 87,8% (erra a maioria dos F); no 200 acerta 96,9%, com bigeminismo
  ventricular bem marcado na figura. Bom para mostrar acerto e limitação no mesmo vídeo.
- Mostrar a tabela cardiologista x modelo que o script imprime e a figura com o trecho do ECG,
  com o rótulo do médico em cinza e o do modelo em cor (um ✗ marca os erros).
- Resultado geral no teste: acurácia de 92,9%, F1-macro de 0,53; ventricular com sensibilidade de
  94,8% e VPP de 86,6%, acima do trabalho de referência de de Chazal (77,7% e 81,9%).
- Mostrar `resultados/mapas_atencao.png`: o modelo presta atenção no QRS e na onda P.
- Limitações, sem esconder: supraventricular e fusão continuam difíceis em paciente novo; o
  paciente 232 sozinho tem 75% dos batimentos S do teste. No protocolo intra-paciente o mesmo modelo
  chega a 99,1% de acurácia, e por isso a gente não usou esse número como resultado.

## 5. Ética e GitHub (3:40-4:40) - Luís e Enzo

Tela: navegador no repositório, pasta `projeto/`.

- Ética em uma frase cada: ferramenta de apoio, nunca de diagnóstico sozinha; a base é antiga,
  pequena e de uma só população, então não pode ser levada para o hospital sem validação local;
  um falso negativo em arritmia é mais grave que um falso positivo.
- GitHub: mostrar README, notebooks já executados, `src/`, artigo em PDF, `resultados/` e
  `requirements.txt`. Dizer que dá para reproduzir tudo rodando os dois notebooks.
- Encerramento: "Obrigado."

## Dicas de gravação

- Gravar a tela do terminal com o `demo.py` antes e cortar a espera do carregamento.
- Conferir no fim que o vídeo tem menos de 5:00; o limite é critério da rubrica.
- Subir no YouTube como "não listado" e colar o link no README e no artigo (seção 4.4).
