from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import torch
from serializer_for_BERT import get_tensors

def eval_driving(model_path, dataset_path):
  # Load pre-trained BERT model and tokenizer
  model = BertForSequenceClassification.from_pretrained(model_path)
  #tokenizer = BertTokenizer.from_pretrained(model_path)

  # Load evaluation dataset
  eval_dataset = get_tensors(dataset_path)[1]  # Assuming the second element of returned tuple is the eval dataset

  # Get predictions
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  model.eval()

  predicted_labels = []
  true_labels = []

  for input_ids, attention_mask, label in eval_dataset:
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)

    with torch.no_grad():
      outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    predicted_labels.append(predicted_label)
    true_labels.append(label.item())

  # Calculate classification report
  report = classification_report(true_labels, predicted_labels)

  print("Classification Report:")
  print(report)

  # Calculate additional metrics
  accuracy = (torch.tensor(predicted_labels) == torch.tensor(true_labels)).float().mean().item()
  precision = precision_score(true_labels, predicted_labels, average='macro')
  recall = recall_score(true_labels, predicted_labels, average='macro')
  f1 = f1_score(true_labels, predicted_labels, average='macro')

  print(f"Accuracy: {accuracy}")
  print(f"Precision: {precision}")
  print(f"Recall: {recall}")
  print(f"F1-score: {f1}")

  # Some more metrics 
  correct_count = sum(1 for pred, true in zip(predicted_labels, true_labels) if pred == true)
  total_count = len(eval_dataset)

  # Print evaluation summary
  print("Eval dataset size:", total_count)
  print("Correct answers:", correct_count)
  print("Wrong answers:", total_count - correct_count)