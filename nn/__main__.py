from model_train import train_driving
from model_eval import eval_driving

dataset = "auto_label_data_augmented_1.csv"
model_name = "BERT_AUTO_LABEL_TRAIN_NOT_TEST_WITH_DATA_AUG"

train_driving(dataset, model_name, 30)
eval_driving(dataset, model_name) 