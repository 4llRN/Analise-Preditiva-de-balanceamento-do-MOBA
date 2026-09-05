"""
2_gerar_gabarito.py
--------------------
Gera os datasets de treino e teste com features temporais
e gabarito baseado nos patch notes reais da Riot.

COMO ATUALIZAR A CADA PATCH NOVO:
  1. Adicione o novo patch em PATCH_NOTES
  2. Mova o patch anterior de PATCH_TESTE para PATCHES_TREINO
  3. Atualize PATCH_TESTE para o patch mais recente

Uso: python script/2_gerar_gabarito.py
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# ▼ ATUALIZAR A CADA PATCH NOVO ▼
# ─────────────────────────────────────────────
PATCH_NOTES = {
    '2608': {
        'nerfs': ['drmundo', 'karma', 'mel'],
        'buffs': ['hwei', 'lillia', 'lucian'],
    },
    '2609': {
        'nerfs': ['ambessa', 'briar'],
        'buffs': ['gragas', 'taliyah', 'warwick', 'tahmkench'],
    },
    '2610': {
        'nerfs': ['anivia', 'ashe', 'naafiri', 'shyvana', 'zed'],
        'buffs': ['ambessa', 'galio', 'quinn', 'wukong', 'zeri'],
    },
    '2611': {
        'nerfs': ['brand', 'teemo'],
        'buffs': ['diana', 'ekko', 'heimerdinger', 'kassadin', 'quinn'],
    },
    '2612': {
        'nerfs': ['leesin', 'nocturne', 'orianna', 'ryze', 'varus', 'xinzhao'],
        'buffs': ['aatrox', 'gwen', 'hwei', 'jax', 'sylas', 'syndra', 'tristana', 'yuumi'],
    },
    # '2613': {         ← PRÓXIMO PATCH: descomente e preencha
    #     'nerfs': [],
    #     'buffs': [],
    # },
}

# Patches usados para TREINAR o modelo
PATCHES_TREINO = ['2609', '2610', '2611']

# Patch usado para TESTAR (sempre o mais recente com patch notes)
PATCH_TESTE = '2612'
# ─────────────────────────────────────────────

ARQUIVO_HISTORICO = "dados/historico_completo.csv"
ARQUIVO_TREINO    = "dados/dataset_treino.csv"
ARQUIVO_TESTE     = "dados/dataset_teste.csv"


def ordenar_patches(df):
    return sorted(df['patch'].unique().tolist(), key=lambda p: int(p))


def aplicar_gabarito(nome_join, patch_atual):
    notas = PATCH_NOTES.get(str(patch_atual), {})
    if nome_join in notas.get('nerfs', []):
        return 1
    if nome_join in notas.get('buffs', []):
        return 2
    return 0


def calcular_features_temporais(df_hist, patches, patch_features, nome_join):
    idx        = patches.index(patch_features)
    historico  = (
        df_hist[
            (df_hist['nome_join'] == nome_join) &
            (df_hist['patch'].isin(patches[:idx + 1]))
        ]
        .sort_values('patch')['Win %'].tolist()
    )

    if len(historico) < 2:
        return 0.0, 0.0, 0, 0.0

    delta_wr_1p  = round(historico[-1] - historico[-2], 2)
    deltas       = [historico[i+1] - historico[i] for i in range(len(historico)-1)]
    tendencia_3p = round(np.mean(deltas[-3:]), 2)

    stats_patch     = df_hist[df_hist['patch'] == patch_features]
    limiar_dinamico = stats_patch['Win %'].mean() + (1.5 * stats_patch['Win %'].std())
    
    pressao = 0
    for wr in reversed(historico):
        if wr >= limiar_dinamico:
            pressao += 1
        else:
            break

    wr_media    = round(np.mean(historico), 2)
    wr_vs_media = round(historico[-1] - wr_media, 2)

    return delta_wr_1p, tendencia_3p, pressao, wr_vs_media


def construir_registros(df_hist, patches, patches_alvo):
    registros = []
    for i, patch_atual in enumerate(patches):
        if str(patch_atual) not in patches_alvo or i == 0:
            continue

        patch_anterior = patches[i - 1]
        df_feat        = df_hist[df_hist['patch'] == patch_anterior].copy()

        for _, row in df_feat.iterrows():
            nome_join = row['nome_join']
            delta, tendencia, pressao, wr_vs_media = calcular_features_temporais(
                df_hist, patches, patch_anterior, nome_join
            )
            registros.append({
                'patch'            : patch_atual,
                'patch_features'   : patch_anterior,
                'nome_join'        : nome_join,
                'Name'             : row['Name'],
                'Win %'            : row['Win %'],
                'Pick %'           : row['Pick %'],
                'Ban %'            : row['Ban %'],
                'Presenca_Global %': round(row['Pick %'] + row['Ban %'], 2),
                'delta_wr_1p'      : delta,
                'tendencia_3p'     : tendencia,
                'pressao_acumulada': pressao,
                'wr_vs_media'      : wr_vs_media,
                'target'           : aplicar_gabarito(nome_join, patch_atual),
            })
    return pd.DataFrame(registros)


if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"  GABARITO — PATCH NOTES REAIS DA RIOT")
    print(f"{'='*55}\n")

    df_hist = pd.read_csv(ARQUIVO_HISTORICO, sep=';')
    patches = ordenar_patches(df_hist)
    print(f"Patches disponíveis: {patches}")
    print(f"Treino : {PATCHES_TREINO}")
    print(f"Teste  : {PATCH_TESTE}")

    df_treino = construir_registros(df_hist, patches, PATCHES_TREINO)
    df_teste  = construir_registros(df_hist, patches, [PATCH_TESTE])

    df_treino.to_csv(ARQUIVO_TREINO, index=False, sep=';')
    df_teste.to_csv(ARQUIVO_TESTE,   index=False, sep=';')

    dist_t = df_treino['target'].value_counts()
    dist_v = df_teste['target'].value_counts()

    print(f"\nTreino ({len(df_treino)} linhas):")
    print(f"  🟡 Estáveis : {dist_t.get(0,0)} | 🔴 Nerfs: {dist_t.get(1,0)} | 🟢 Buffs: {dist_t.get(2,0)}")
    print(f"\nTeste ({len(df_teste)} linhas):")
    print(f"  🟡 Estáveis : {dist_v.get(0,0)} | 🔴 Nerfs: {dist_v.get(1,0)} | 🟢 Buffs: {dist_v.get(2,0)}")

    todos = set(df_hist['nome_join'].unique())
    print(f"\n[CHECAGEM] Campeões não encontrados no histórico:")
    ok = True
    for patch, notas in PATCH_NOTES.items():
        for tipo, lista in notas.items():
            for nome in lista:
                if nome not in todos:
                    print(f"  ⚠️  '{nome}' ({patch}/{tipo})")
                    ok = False
    if ok:
        print("  ✓ Todos encontrados!")

    print(f"\n{'='*55}")
    print(f"  CONCLUÍDO! → {ARQUIVO_TREINO} | {ARQUIVO_TESTE}")
    print(f"{'='*55}")
    print("\nPróximo passo: python script/3_IA_treinar_e_validar.py")
