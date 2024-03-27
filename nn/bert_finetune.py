from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
from serializer_for_BERT import train_dataset, eval_dataset
from transformers import DataCollatorWithPadding

class CustomDataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer)

    def __call__(self, features):
        if isinstance(features[0], (list, tuple)):
            # Handle list-like features
            input_ids = [feature[0] for feature in features]
            attention_masks = [feature[1] for feature in features]
            labels = [feature[2] for feature in features]
            
            return {
                'input_ids': torch.stack(input_ids, dim=0),
                'attention_mask': torch.stack(attention_masks, dim=0),
                'labels': torch.tensor(labels, dtype=torch.long)
            }
        else:
            # Handle dictionary-like features
            return {
                key: torch.tensor([getattr(f, key) for f in features]) for key in features[0].__dict__.keys()
            }


# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./output',
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=100,
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=500,
)

# Define custom data collators
data_collator = CustomDataCollator(tokenizer)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

# Fine-tune the model
trainer.train()