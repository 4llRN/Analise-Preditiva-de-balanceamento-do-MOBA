import pandas as pd
import requests
import re

# Dicionário de conversão para garantir que os nomes da Riot batam com os do U.GG
DE_PARA_NOMES = {
    "monkeyking": "wukong",
    "nunuwillump": "nunu",
    "renataglasc": "renata",
    "ksante": "ksante",
    "belveth": "belveth"
}

def obter_versao_atual():
    """Busca a lista de versões oficiais e retorna a mais recente."""
    url = "https://ddragon.leagueoflegends.com/api/versions.json"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    resposta = requests.get(url, headers=headers)
    if resposta.status_code == 200:
        return resposta.json()[0] 
    return None

def obter_dados_completos():
    versao = obter_versao_atual()
    if not versao:
        print("Não foi possível recuperar as versões da API.")
        return None
    
    url_ddragon = f'https://ddragon.leagueoflegends.com/cdn/{versao}/data/pt_BR/champion.json'
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    print(f"Buscando dados do Patch oficial: {versao}...")
    resposta = requests.get(url_ddragon, headers=headers)
    
    if resposta.status_code == 200:
        dados_json = resposta.json()['data']
        lista_preparada = []

        for nome_id, info in dados_json.items():
            # NORMALIZAÇÃO: Criando a chave de ligação (nome_join)
            nome_limpo = re.sub(r'[^a-zA-Z0-9]', '', nome_id).lower()
            nome_join = DE_PARA_NOMES.get(nome_limpo, nome_limpo)
            
            lista_preparada.append({
                'nome_join': nome_join,
                'Name_API': info['name'],
                'tags': info['tags'][0] if info['tags'] else 'None',
                'Patch': versao
            })

        df = pd.DataFrame(lista_preparada)
        print(f"Sucesso! {len(df)} campeões encontrados na API.")
        return df
    else:
        print(f"Erro {resposta.status_code} ao acessar o Data Dragon.")
        return None

# --- EXECUÇÃO ---
df_analise = obter_dados_completos()

if df_analise is not None:
    # A PEÇA QUE FALTAVA: Salvar fisicamente na pasta dados!
    df_analise.to_csv('dados/campeoes_info.csv', index=False, sep=';')
    print("\nArquivo 'campeoes_info.csv' salvo com sucesso na pasta dados!")
    print(df_analise.head())