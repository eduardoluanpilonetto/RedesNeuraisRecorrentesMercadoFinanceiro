import json

def load_preprocessed_data(data_folder):
    with open(f'{data_folder}/processed/sequences.json', 'r') as f:
        sequences = json.load(f)
    with open(f'{data_folder}/processed/labels.json', 'r') as f:
        labels = json.load(f)
    
    return sequences, labels