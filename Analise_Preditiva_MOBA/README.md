# Game Analytics — Previsão de Balanceamento do LoL

Sistema de análise preditiva que utiliza Machine Learning para identificar campeões candidatos a nerf ou buff nos próximos patches do League of Legends.

---

## Como funciona

O modelo aprende com o histórico de estatísticas de patches anteriores e os patch notes reais da Riot para estimar quais campeões têm maior probabilidade de serem alterados no próximo patch.

O pipeline é dividido em 4 etapas sequenciais:

```
1_coletor_historico.py
        ↓
2_gerar_gabarito.py
        ↓
3_IA_treinar_e_validar.py
        ↓
4_IA_prever_proximo.py
```

---

## Estrutura de pastas

```
GA-LOL/
├── dados/
│   ├── ugg_2608.html          ← HTMLs salvos do U.GG (um por patch)
│   ├── ugg_2609.html
│   ├── ...
│   ├── historico_completo.csv ← gerado pelo script 1
│   ├── dataset_treino.csv     ← gerado pelo script 2
│   ├── dataset_teste.csv      ← gerado pelo script 2
│   ├── resultado_validacao.csv
│   └── previsao_proximo_patch.csv
└── script/
    ├── 1_coletor_historico.py
    ├── 2_gerar_gabarito.py
    ├── 3_IA_treinar_e_validar.py
    ├── 4_IA_prever_proximo.py
    ├── extrator_api.py
    └── modelo_moba.pkl        ← modelo treinado (gerado pelo script 3)
```

---

## Pré-requisitos

```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn requests
```

---

## Como usar

### Execução completa

A partir da pasta `GA-LOL/`, execute os scripts na ordem:

```bash
python script/1_coletor_historico.py
python script/2_gerar_gabarito.py
python script/3_IA_treinar_e_validar.py
python script/4_IA_prever_proximo.py
```

---

## Scripts — descrição detalhada

### 1_coletor_historico.py

Lê todos os arquivos `ugg_*.html` salvos na pasta `dados/` e extrai as estatísticas de cada campeão por patch.

Os HTMLs devem ser salvos manualmente do U.GG:
1. Acesse `u.gg/lol/tier-list`
2. Pressione `Ctrl+U` para abrir o código fonte
3. Salve como `ugg_XXYY.html` (ex: `ugg_2612.html`) dentro de `dados/`

**Saída:** `dados/historico_completo.csv`

---

### 2_gerar_gabarito.py

Constrói os datasets de treino e teste com:
- Estatísticas do patch anterior como **features** (o que o modelo analisa)
- Patch notes reais da Riot como **target** (o que o modelo aprende)
- Features temporais calculadas automaticamente

**Features utilizadas:**

| Feature | Descrição |
|---|---|
| `Win %` | Taxa de vitória no patch |
| `Pick %` | Taxa de escolha |
| `Ban %` | Taxa de banimento |
| `Presenca_Global %` | Pick % + Ban % |
| `delta_wr_1p` | Variação de WinRate em relação ao patch anterior |
| `tendencia_3p` | Média das variações nos últimos 3 patches |
| `pressao_acumulada` | Patches seguidos acima do limiar dinâmico de WinRate |
| `wr_vs_media` | Desvio em relação à média histórica do próprio campeão |

**Limiar dinâmico de pressão:**
Em vez de um valor fixo (ex: 52%), o limiar é calculado por patch:
```
limiar = média global de WinRate do patch + (1.5 × desvio padrão)
```
Isso garante que a pressão acumulada reflita o contexto do meta de cada patch.

**Como atualizar a cada patch novo:**
```python
# Em 2_gerar_gabarito.py:

PATCH_NOTES = {
    '2612': {
        'nerfs': ['leesin', 'nocturne', ...],
        'buffs': ['aatrox', 'gwen', ...],
    },
    '2613': {          # ← adicionar novo patch aqui
        'nerfs': [],
        'buffs': [],
    },
}

PATCHES_TREINO = ['2609', '2610', '2611', '2612']  # mover patch anterior para cá
PATCH_TESTE    = '2613'                             # atualizar para o mais recente
```

**Saída:** `dados/dataset_treino.csv` e `dados/dataset_teste.csv`

---

### 3_IA_treinar_e_validar.py

Treina o modelo com os dados históricos e exibe os top 10 candidatos a nerf e buff para o patch de teste.

**Algoritmo:** Random Forest Classifier com 200 árvores de decisão.
- `class_weight='balanced'` — compensa o desequilíbrio entre campeões estáveis (~95%) e alterados (~5%)
- `max_depth=6` — limita a profundidade das árvores para evitar overfitting

**Saída:**
- `script/modelo_moba.pkl` — modelo treinado salvo para reutilização
- `dados/resultado_validacao.csv` — probabilidades de cada campeão
- `dados/importancia_features.png` — gráfico de importância das features

---

### 4_IA_prever_proximo.py

Carrega o modelo treinado e usa as estatísticas do patch mais recente para prever os candidatos do próximo patch.

Calcula automaticamente as features temporais com base no histórico disponível e exibe os top 10 candidatos a nerf e buff por probabilidade.

**Saída:** `dados/previsao_proximo_patch.csv`

---

### extrator_api.py

Busca dados complementares dos campeões diretamente da API oficial da Riot (Data Dragon), como classe e versão do patch. Opcional — não é necessário para o pipeline principal.

```bash
python script/extrator_api.py
```

**Saída:** `dados/campeoes_info.csv`

---

## Ritual a cada patch novo (15 minutos)

1. Salvar o novo HTML do U.GG em `dados/ugg_XXYY.html`
2. Abrir `2_gerar_gabarito.py` e:
   - Adicionar os patch notes do novo patch em `PATCH_NOTES`
   - Mover `PATCH_TESTE` atual para `PATCHES_TREINO`
   - Atualizar `PATCH_TESTE` para o patch novo
3. Abrir `3_IA_treinar_e_validar.py` e atualizar `PATCH_TESTE`
4. Executar os 4 scripts na ordem

---

## Algoritmo — Random Forest

O Random Forest cria múltiplas árvores de decisão independentes, cada uma aprendendo regras como:

```
SE Win% alto E pressao_acumulada > 2 E delta_wr positivo
   → candidato a Nerf

SE Win% abaixo da média E Presenca baixa E wr_vs_media negativo
   → candidato a Buff
```

Com 200 árvores votando, o resultado final é mais robusto do que uma árvore única. O modelo retorna uma **probabilidade** para cada classe (Estável, Nerf, Buff), e os campeões são rankeados por essa probabilidade.

---

## Validação — Leave-One-Patch-Out

Para medir a qualidade do modelo, utiliza-se validação temporal: o modelo é treinado com N-1 patches e testado no patch deixado de fora. Isso evita **data leakage** — situação onde o modelo "vê o futuro" durante o treino.

---

## Limitações atuais

- O modelo melhora proporcionalmente à quantidade de patches históricos disponíveis
- Com menos de 10 patches contínuos, os resultados são indicativos, não definitivos
- Buffs são mais difíceis de prever do que nerfs, pois dependem de contexto de meta mais amplo
- As listas exibidas são **candidatos por probabilidade**, não previsões definitivas

---

## Patches disponíveis

| Patch | Status |
|---|---|
| 26.08 | ✅ Disponível |
| 26.09 | ✅ Disponível |
| 26.10 | ✅ Disponível |
| 26.11 | ✅ Disponível |
| 26.12 | ✅ Disponível (patch de teste atual) |
| 26.13 | ⏳ Aguardando |
