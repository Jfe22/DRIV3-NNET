from model_eval import eval_driving

eval_driving('../models/BERT_accelometer_normalized_30epochs', 
             '../dataset/bmw_accelerometer_only_labeled_normalized_dataset.csv')