from model_train import train_driving
from model_eval import eval_driving

dataset = "auto_label_test1.csv"
model_name = "BERT_AUTO_LABEL_TRAIN_NOT_TEST"

train_driving(dataset, model_name, 30)
eval_driving(dataset, model_name) 