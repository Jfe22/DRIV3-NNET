from model_train import train_driving
from model_eval import eval_driving

dataset = "bmw_accelerometer_and_gyroscope_labeled_normalized_nodate_dataset.csv"
model_name = "BERT_accelerometer_and_gyroscope_normalized_nodate_30epochs_V2"

train_driving(dataset, model_name, 30)
eval_driving(dataset, model_name) 