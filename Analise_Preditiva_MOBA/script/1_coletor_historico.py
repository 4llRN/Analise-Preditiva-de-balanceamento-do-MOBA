"""
1_coletor_historico.py
-----------------------
Processa todos os HTMLs do U.GG salvos em 'dados/' e gera:
  dados/historico_completo.csv  → todos os patches empilhados

Convenção de nome: ugg_XXYY.html (ex: ugg_2612.html)

Uso: python script/1_coletor_historico.py
"""

import pandas as pd
import json
import re
import glob
import os
from pathlib import Path

PASTA_DADOS   = "dados"
PASTA_SAIDA   = "dados/historico"
ARQUIVO_FINAL = "dados/historico_completo.csv"

DE_PARA_NOMES = {
    "monkeyking" : "wukong",
    "nunuwillump": "nunu",
    "renataglasc": "renata",
    "ksante"     : "ksante",
    "belveth"    : "belveth",
}

def extrair_patch_do_nome(caminho):
    return Path(caminho).stem.split('_')[-1]

def processar_html(caminho_arquivo):
    patch_id = extrair_patch_do_nome(caminho_arquivo)
    print(f"  → Processando patch '{patch_id}' ({caminho_arquivo})...")

    with open(caminho_arquivo, "r", encoding="utf-8", errors="ignore") as f:
        conteudo = f.read()

    if 'window.__SSR_DATA__' not in conteudo:
        print(f"     [AVISO] Sem __SSR_DATA__. Pulando.")
        return None

    try:
        parte_json = conteudo.split('window.__SSR_DATA__ =')[1].strip()
        dados_ssr, _ = json.JSONDecoder().raw_decode(parte_json)
    except Exception as e:
        print(f"     [ERRO] Falha ao decodificar JSON: {e}. Pulando.")
        return None

    mapa_campeoes = {}
    for chave, valor in dados_ssr.items():
        if 'champion.json' in chave:
            for champ_key, champ_data in valor['data'].items():
                mapa_campeoes[int(champ_key)] = champ_data['name']
            break

    if not mapa_campeoes:
        print(f"     [AVISO] Mapa de campeões vazio. Pulando.")
        return None

    dados_brutos = []
    for chave, valor in dados_ssr.items():
        if 'champion_ranking' in chave:
            for rota, lista_campeoes in valor['data']['win_rates'].items():
                for champ in lista_campeoes:
                    champ_id  = int(champ['champion_id'])
                    nome      = mapa_campeoes.get(champ_id, f"Desconhecido_{champ_id}")
                    nome_limpo = re.sub(r'[^a-zA-Z0-9]', '', nome).lower()
                    nome_limpo = DE_PARA_NOMES.get(nome_limpo, nome_limpo)
                    dados_brutos.append({
                        'Name'    : nome,
                        'nome_join': nome_limpo,
                        'Win %'   : float(champ['win_rate']),
                        'Pick %'  : float(champ['pick_rate']),
                        'Ban %'   : float(champ['ban_rate']),
                    })
            break

    if not dados_brutos:
        print(f"     [AVISO] Sem dados de champion_ranking. Pulando.")
        return None

    df = pd.DataFrame(dados_brutos).groupby('nome_join').agg(
        Name   =('Name',   'first'),
        Win_pct=('Win %',  'mean'),
        Pick_pct=('Pick %', 'sum'),
        Ban_pct =('Ban %',  'max'),
    ).reset_index()

    df.rename(columns={'Win_pct':'Win %','Pick_pct':'Pick %','Ban_pct':'Ban %'}, inplace=True)
    df['Win %']  = df['Win %'].round(2)
    df['Pick %'] = df['Pick %'].round(2)
    df['Ban %']  = df['Ban %'].round(2)
    df['patch']  = patch_id

    print(f"     ✓ {len(df)} campeões extraídos.")
    return df

if __name__ == "__main__":
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    arquivos = sorted(glob.glob(f"{PASTA_DADOS}/ugg_*.html"))

    if not arquivos:
        print(f"[ERRO] Nenhum arquivo 'ugg_*.html' em '{PASTA_DADOS}/'.")
        exit(1)

    print(f"\n{'='*55}")
    print(f"  COLETOR HISTÓRICO — {len(arquivos)} arquivo(s)")
    print(f"{'='*55}\n")

    todos = []
    for arquivo in arquivos:
        df_patch = processar_html(arquivo)
        if df_patch is not None:
            df_patch.to_csv(f"{PASTA_SAIDA}/{extrair_patch_do_nome(arquivo)}_stats.csv", index=False, sep=';')
            todos.append(df_patch)

    if not todos:
        print("\n[ERRO] Nenhum patch processado.")
        exit(1)

    df_historico = pd.concat(todos, ignore_index=True)
    df_historico.to_csv(ARQUIVO_FINAL, index=False, sep=';')

    print(f"\n{'='*55}")
    print(f"  CONCLUÍDO! {len(todos)} patches | {len(df_historico)} linhas")
    print(f"  → {ARQUIVO_FINAL}")
    print(f"{'='*55}")
    print("\nPróximo passo: python script/2_gerar_gabarito.py")
