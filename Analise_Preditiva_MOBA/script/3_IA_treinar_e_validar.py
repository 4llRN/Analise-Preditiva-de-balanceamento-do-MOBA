"""
3_IA_treinar_e_validar.py
--------------------------
Treina o modelo e exibe os top candidatos a nerf/buff.

Uso: python script/3_IA_treinar_e_validar.py
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# ─────────────────────────────────────────────
# ▼ ATUALIZAR A CADA PATCH NOVO ▼
# ─────────────────────────────────────────────
PATCH_TESTE = '2612'
# ─────────────────────────────────────────────

ARQUIVO_TREINO = "dados/dataset_treino.csv"
ARQUIVO_TESTE  = "dados/dataset_teste.csv"
ARQUIVO_RESULT = "dados/resultado_validacao.csv"
ARQUIVO_MODELO = "script/modelo_moba.pkl"

FEATURES = [
    'Win %', 'Pick %', 'Ban %', 'Presenca_Global %',
    'delta_wr_1p', 'tendencia_3p', 'pressao_acumulada', 'wr_vs_media'
]

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  TREINAMENTO — PATCH {PATCH_TESTE}")
    print(f"{'='*60}\n")

    df_treino = pd.read_csv(ARQUIVO_TREINO, sep=';')
    df_teste  = pd.read_csv(ARQUIVO_TESTE,  sep=';')

    X_treino = df_treino[FEATURES].fillna(0)
    y_treino = df_treino['target']
    X_teste  = df_teste[FEATURES].fillna(0)

    print(f"Treino : {len(df_treino)} exemplos ({df_treino['patch'].nunique()} patches)")
    print(f"Teste  : {len(df_teste)} exemplos (patch {PATCH_TESTE})")

    dist = df_treino['target'].value_counts()
    print(f"Distribuição → Estáveis:{dist.get(0,0)} | Nerfs:{dist.get(1,0)} | Buffs:{dist.get(2,0)}")

    print("\nTreinando modelo...")
    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    modelo.fit(X_treino, y_treino)
    joblib.dump(modelo, ARQUIVO_MODELO)
    print(f"Modelo salvo em: {ARQUIVO_MODELO}")

    probabilidades = modelo.predict_proba(X_teste)
    df_teste = df_teste.copy()
    df_teste['Prob_Nerf %'] = (probabilidades[:, 1] * 100).round(1)
    df_teste['Prob_Buff %'] = (probabilidades[:, 2] * 100).round(1)

    print(f"\n{'='*60}")
    print(f"  🔴 TOP 10 CANDIDATOS A NERF — PATCH {PATCH_TESTE}")
    print(f"{'='*60}")
    top_n = df_teste.sort_values('Prob_Nerf %', ascending=False).head(10)
    print(top_n[['Name', 'Win %', 'Presenca_Global %', 'Prob_Nerf %']].to_string(index=False))

    print(f"\n{'='*60}")
    print(f"  🟢 TOP 10 CANDIDATOS A BUFF — PATCH {PATCH_TESTE}")
    print(f"{'='*60}")
    top_b = df_teste.sort_values('Prob_Buff %', ascending=False).head(10)
    print(top_b[['Name', 'Win %', 'Presenca_Global %', 'Prob_Buff %']].to_string(index=False))

    # Gráfico de importância
    importancia = modelo.feature_importances_
    indices     = np.argsort(importancia)[::-1]
    plt.figure(figsize=(10, 5))
    plt.title("Importância das Features")
    sns.barplot(x=importancia[indices], y=[FEATURES[i] for i in indices],
                hue=[FEATURES[i] for i in indices], palette="viridis", legend=False)
    plt.tight_layout()
    plt.savefig("dados/importancia_features.png", dpi=120)
    plt.close()

    df_teste.to_csv(ARQUIVO_RESULT, index=False, sep=';')
    print(f"\nResultado completo: {ARQUIVO_RESULT}")
    print(f"\n{'='*60}")
    print(f"  CONCLUÍDO!")
    print(f"{'='*60}")
    print("\nPróximo passo: python script/4_IA_prever_proximo.py")