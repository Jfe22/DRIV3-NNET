import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

def get_tensors(dataset_path):
  train_df = pd.read_csv(dataset_path)
  texts = train_df['sensor_data'].tolist()
  labels = train_df['label'].tolist()

  train_texts, train_labels, eval_texts, eval_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)

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

  print ("Train dataset size:", len(train_dataset))
  print ("Eval dataset size:", len(eval_dataset))
  return train_dataset, eval_dataset