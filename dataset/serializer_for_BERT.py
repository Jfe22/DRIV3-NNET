import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import TensorDataset

# Read CSV files
train_df = pd.read_csv('train.csv')
eval_df = pd.read_csv('eval.csv')

# Preprocess data
# Assuming train_df and eval_df have columns 'sensor_data' and 'label'
train_texts = train_df['sensor_data'].tolist()
eval_texts = eval_df['sensor_data'].tolist()
train_labels = train_df['label'].tolist()
eval_labels = eval_df['label'].tolist()

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
eval_encodings = tokenizer(eval_texts, truncation=True, padding=True)

# Create TensorDatasets
train_dataset = TensorDataset(
    torch.tensor(train_encodings['input_ids']),
    torch.tensor(train_encodings['attention_mask']),
    torch.tensor(train_labels)
)
eval_dataset = TensorDataset(
    torch.tensor(eval_encodings['input_ids']),
    torch.tensor(eval_encodings['attention_mask']),
    torch.tensor(eval_labels)
)
