"""
4_IA_prever_proximo.py
-----------------------
Usa o modelo treinado para prever o PRÓXIMO patch
com base nas stats do patch mais recente disponível.

Uso: python script/4_IA_prever_proximo.py
"""

import pandas as pd
import numpy as np
import joblib

ARQUIVO_HISTORICO = "dados/historico_completo.csv"
ARQUIVO_MODELO    = "script/modelo_moba.pkl"
ARQUIVO_RESULT    = "dados/previsao_proximo_patch.csv"

FEATURES = [
    'Win %', 'Pick %', 'Ban %', 'Presenca_Global %',
    'delta_wr_1p', 'tendencia_3p', 'pressao_acumulada', 'wr_vs_media'
]

def ordenar_patches(df):
    return sorted(df['patch'].unique().tolist(), key=lambda p: int(p))

def calcular_features_temporais(df_hist, patches, patch_atual, nome_join):
    idx       = patches.index(patch_atual)
    historico = (
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

    stats_patch     = df_hist[df_hist['patch'] == patch_atual]
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

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  PREVISÃO — PRÓXIMO PATCH")
    print(f"{'='*60}\n")

    df_hist       = pd.read_csv(ARQUIVO_HISTORICO, sep=';')
    patches       = ordenar_patches(df_hist)
    patch_recente = patches[-1]

    print(f"Usando stats do patch {patch_recente} para prever o próximo...")

    modelo = joblib.load(ARQUIVO_MODELO)

    registros = []
    for _, row in df_hist[df_hist['patch'] == patch_recente].iterrows():
        nome_join = row['nome_join']
        delta, tendencia, pressao, wr_vs_media = calcular_features_temporais(
            df_hist, patches, patch_recente, nome_join
        )
        registros.append({
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
        })

    df_prever = pd.DataFrame(registros)
    X         = df_prever[FEATURES].fillna(0)

    probabilidades             = modelo.predict_proba(X)
    df_prever['Prob_Nerf %']   = (probabilidades[:, 1] * 100).round(1)
    df_prever['Prob_Buff %']   = (probabilidades[:, 2] * 100).round(1)

    print(f"\n{'='*60}")
    print(f"  🔴 TOP 10 CANDIDATOS A NERF — PRÓXIMO PATCH")
    print(f"{'='*60}")
    top_n = df_prever.sort_values('Prob_Nerf %', ascending=False).head(10)
    print(top_n[['Name', 'Win %', 'Presenca_Global %', 'Prob_Nerf %']].to_string(index=False))

    print(f"\n{'='*60}")
    print(f"  🟢 TOP 10 CANDIDATOS A BUFF — PRÓXIMO PATCH")
    print(f"{'='*60}")
    top_b = df_prever.sort_values('Prob_Buff %', ascending=False).head(10)
    print(top_b[['Name', 'Win %', 'Presenca_Global %', 'Prob_Buff %']].to_string(index=False))

    df_prever.to_csv(ARQUIVO_RESULT, index=False, sep=';')
    print(f"\nResultado salvo em: {ARQUIVO_RESULT}")