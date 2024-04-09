import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset

def get_tensors(dataset_path):
    train_df = pd.read_csv(dataset_path)
    eval_df = pd.read_csv(dataset_path)
    train_texts = train_df['sensor_data'].tolist()
    eval_texts = eval_df['sensor_data'].tolist()
    train_labels = train_df['label'].tolist()
    eval_labels = eval_df['label'].tolist()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    eval_encodings = tokenizer(eval_texts, truncation=True, padding=True)

    label_map = {"Aggressive": 0, "Slow": 1, "Normal": 2}
    train_labels_int = [label_map[label] for label in train_labels]
    eval_labels_int = [label_map[label] for label in eval_labels]

    train_dataset = TensorDataset(
      torch.tensor(train_encodings['input_ids'], dtype=torch.long),  
      torch.tensor(train_encodings['attention_mask'], dtype=torch.long),  
      torch.tensor(train_labels_int, dtype=torch.long)  
    )
    eval_dataset = TensorDataset(
      torch.tensor(eval_encodings['input_ids'], dtype=torch.long),  
      torch.tensor(eval_encodings['attention_mask'], dtype=torch.long),  
      torch.tensor(eval_labels_int, dtype=torch.long) 
    )
    return train_dataset, eval_dataset