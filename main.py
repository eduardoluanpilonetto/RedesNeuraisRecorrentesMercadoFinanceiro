import matplotlib.pyplot as plt
from data.data_fetcher import fetch_stock_data
from data.data_saver import save_data_to_json
from data.data_preprocessor import preprocess_data, save_preprocessed_data
from data.data_loader import load_preprocessed_data
from model_builder import build_lstm_model
import numpy as np
from sklearn.model_selection import train_test_split

def train_model(sequences, labels, batch_size, epochs, validation_split):
    # Convertendo as listas de sequências e rótulos em arrays numpy
    sequences = np.array(sequences)
    labels = np.array(labels)

    # Dividindo os dados em conjuntos de treinamento e validação
    x_train, x_val, y_train, y_val = train_test_split(sequences, labels, test_size=validation_split, random_state=42)

    # Obtendo o input_shape do modelo
    input_shape = x_train.shape[1:]

    # Construindo o modelo LSTM
    model = build_lstm_model(input_shape)

    # Treinando o modelo
    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))

    return model

def evaluate_model(model, x_test, y_test):
    # Fazendo previsões com o modelo treinado
    predictions = model.predict(x_test)

    # Calculando o erro médio absoluto (MAE)
    mae = np.mean(np.abs(predictions - y_test))

    return mae, predictions

def plot_predictions(y_true, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Preços Reais')
    plt.plot(predictions, label='Previsões')
    plt.xlabel('Período')
    plt.ylabel('Preço de Fechamento')
    plt.title('Comparação entre Preços Reais e Previsões do Modelo')
    plt.legend()
    plt.show()

def main():
    # Definir as configurações
    api_key = 'd8d9810fe438400f9138837cd365c7dc7r'
    symbol = 'MSFT'
    window_size = 20
    data_folder = 'data'

    # Passo 1: Buscar dados usando a API do Alpha Vantage
    stock_data = fetch_stock_data(api_key, symbol)

    if stock_data:
        # Passo 2: Armazenar os dados brutos em um arquivo JSON
        raw_data_filename = f'{data_folder}/raw/stock_data.json'
        save_data_to_json(stock_data, raw_data_filename)

        # Passo 3: Pré-processamento dos dados
        sequences, labels = preprocess_data(stock_data, window_size)

        # Passo 4: Armazenar os dados pré-processados em arquivos JSON
        processed_data_folder = f'{data_folder}/processed/'
        save_preprocessed_data(sequences, labels, processed_data_folder)

        # Passo 5: Carregar os dados pré-processados
        sequences, labels = load_preprocessed_data(data_folder)

        # Passo 6: Construir o modelo LSTM
        input_shape = (window_size, len(sequences[0][0]))
        model = build_lstm_model(input_shape)

	    # Passo 7: Treinar o modelo
        batch_size = 32
        epochs = 100
        validation_split = 0.15
        trained_model = train_model(sequences, labels, batch_size, epochs, validation_split)

        # Passo 8: Avaliar o modelo
        _, x_test, _, y_test = train_test_split(sequences, labels, test_size=validation_split, random_state=42)
        mae, predictions = evaluate_model(trained_model, x_test, y_test)

        # Passo 9: Plotar as previsões
        plot_predictions(y_test, predictions)

        print("Treinamento e Avaliação concluídos!")
    else:
        print("Erro ao buscar dados.")


if __name__ == "__main__":
    main()