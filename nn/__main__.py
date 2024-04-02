from model_train import train_driving
from model_eval import eval_driving

train_driving('bmw_accelerometer_and_gyroscope_labeled_normalized_dataset.csv',
              'BERT_accelerometer_and_gyroscope_normalized_30epochs', 
              30)

eval_driving('bmw_accelerometer_and_gyroscope_labeled_normalized_dataset.csv',
             'BERT_accelerometer_and_gyroscope_normalized_30epochs') 