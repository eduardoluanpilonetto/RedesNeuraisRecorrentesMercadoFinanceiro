import requests
import json

def fetch_stock_data(api_key, symbol):
    base_url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'outputsize': 'full',  # Pode ser 'compact' (últimos 100 dias) ou 'full' (todos os dados disponíveis)
        'apikey': 'd8d9810fe438400f9138837cd365c7dc7r', #Nesse trecho vai a minha API Key, pego no cadastro do site
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        return data['Time Series (Daily)']
    else:
        print(f"Erro ao buscar dados: {response.status_code}")
        return None
