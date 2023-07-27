import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import json

def preprocess_data(data, window_size):
    df = pd.DataFrame.from_dict(data, orient='index', columns=['open', 'high', 'low', 'close', 'volume'])
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df)

    sequences = []
    labels = []
    for i in range(len(normalized_data) - window_size):
        sequences.append(normalized_data[i : i + window_size])
        labels.append(normalized_data[i + window_size, 3])  # Usando o preço de fechamento como rótulo

    return sequences, labels

def save_preprocessed_data(sequences, labels, data_folder):
    with open(f'{data_folder}/processed/sequences.json', 'w') as f:
        json.dump(sequences, f)
    with open(f'{data_folder}/processed/labels.json', 'w') as f:
        json.dump(labels, f)