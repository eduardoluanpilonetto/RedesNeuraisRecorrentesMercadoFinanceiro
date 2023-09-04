import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import json
import numpy as np

def preprocess_data(data, window_size):
    df = pd.DataFrame.from_dict(data, orient='index', columns=['1. open', '2. high', '3. low', '4. close', '5. volume'])
    scaler = MinMaxScaler(feature_range=(0, 1000))
    normalized_data = scaler.fit_transform(df)

    sequences = []
    labels = []
    for i in range(len(normalized_data) - window_size):
        sequences.append(normalized_data[i : i + window_size])
        labels.append(normalized_data[i + window_size, 3])  # Usando o preço de fechamento como rótulo

    return sequences, labels

def convert_to_serializable(item):
    if isinstance(item, np.ndarray):
        return item.tolist()
    return item

def save_preprocessed_data(sequences, labels, data_folder):
    sequences_serializable = [convert_to_serializable(seq) for seq in sequences]
    labels_serializable = [convert_to_serializable(label) for label in labels]

    with open(f'{data_folder}sequences.json', 'w') as f:
        json.dump(sequences_serializable, f)
    with open(f'{data_folder}labels.json', 'w') as f:
        json.dump(labels_serializable, f)