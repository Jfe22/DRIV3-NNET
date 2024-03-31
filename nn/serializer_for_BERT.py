import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset

def get_tensors(dataset_path):
    # Read CSV files
    train_df = pd.read_csv(dataset_path)
    eval_df = pd.read_csv(dataset_path)

    # Preprocess data
    train_texts = train_df['sensor_data'].tolist()
    eval_texts = eval_df['sensor_data'].tolist()
    train_labels = train_df['label'].tolist()
    eval_labels = eval_df['label'].tolist()

    # Tokenization
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    eval_encodings = tokenizer(eval_texts, truncation=True, padding=True)

    # Create a mapping from label classes to integers
    label_map = {"Aggressive": 0, "Slow": 1, "Normal": 2}

    # Convert label strings to integers based on the mapping
    train_labels_int = [label_map[label] for label in train_labels]
    eval_labels_int = [label_map[label] for label in eval_labels]

    # Create train tensor 
    train_dataset = TensorDataset(
      torch.tensor(train_encodings['input_ids'], dtype=torch.long),  
      torch.tensor(train_encodings['attention_mask'], dtype=torch.long),  
      torch.tensor(train_labels_int, dtype=torch.long)  
    )

    # Create evaluation tensor (same as training for now)
    eval_dataset = TensorDataset(
      torch.tensor(eval_encodings['input_ids'], dtype=torch.long),  
      torch.tensor(eval_encodings['attention_mask'], dtype=torch.long),  
      torch.tensor(eval_labels_int, dtype=torch.long) 
    )

    return train_dataset, eval_dataset