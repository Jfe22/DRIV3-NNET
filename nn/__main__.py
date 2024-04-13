from model_train import train_driving
from model_eval import eval_driving

dataset = "1sec_timeframe_dataset.csv"
model_name = "BERT_1sec_timeframe_30epochs"

train_driving(dataset, model_name, 30)
eval_driving(dataset, model_name) 