import pandas as pd
import random

DEBUG = 0
concatenated_rows = []
concatenated_normalized_dataset = []
concatenated_raw_dataset = []
variance_dataset = []
window_size = 4
increment = 1

def get_arrays(window):
  arrayAccX = [i for i in window["accelerometerXAxis"]]
  arrayAccY = [i for i in window["accelerometerYAxis"]]
  arrayAccZ = [i for i in window["accelerometerZAxis"]]
  arrayGyroX = [i for i in window["gyroscopeXAxis"]]
  arrayGyroY = [i for i in window["gyroscopeYAxis"]]
  arrayGyroZ = [i for i in window["gyroscopeZAxis"]]
  arrayLabel = [i for i in window["label"]]
  label = arrayLabel[0]
  varianceAccY = max(arrayAccY) - min(arrayAccY)

  return arrayAccX, arrayAccY, arrayAccZ, arrayGyroX, arrayGyroY, arrayGyroZ, label 

#Use with normalized data
def set_label_mean(accY):
  nVals = 0
  sumAccY = 0

  for val in accY:
    nVals += 1
    sumAccY += val 
  meanAccY = sumAccY / nVals

  if (meanAccY > 0.8): return "Aggressive"
  return "Normal"

#Use with normalized data
def set_label_variance(accY):
  varAccY = max(accY) - min(accY)

  if (varAccY > 0.8): return "Aggressive"
  return "Normal"

#Use with non-normalized data
def set_label_sum(accX, accY, accZ):
  sumAcc = 0
  sumAcc += sum(accX)
  sumAcc += sum(accY)
  sumAcc += sum(accZ)
  ## PARA DAR SET DA LABEL, VALOR DE 4.3 SABEMOS A PRIORI QUE PODE SER USADO COMO TRESHOLD
  if (sumAcc > 4.3): return "Aggressive"
  return "Normal"

def get_windows(data, window_type):
  concatenated_4instarray_dataset = pd.DataFrame() 
  for i in range(len(data) - window_size + increment):
    window = data[i:i+window_size]
    accX, accY, accZ, gyroX, gyroY, gyroZ, label = get_arrays(window)
    label = set_label_sum(accX, accY, accZ)

    if window_type == 'array':
      new_row = pd.DataFrame({'accelerometerXAxis': [accX], 
                              'accelerometerYAxis': [accY],  
                              'accelerometerZAxis': [accZ],
                              'gyroscopeXAxis': [gyroX], 
                              'gyroscopeYAxis': [gyroY], 
                              'gyroscopeZAxis': [gyroZ], 
                              'label': [label]}, index=[0])
    elif window_type == 'sum':
      new_row = pd.DataFrame({'accelerometerXAxis': [sum(accX)], 
                              'accelerometerYAxis': [sum(accY)],  
                              'accelerometerZAxis': [sum(accZ)],
                              'gyroscopeXAxis': [sum(gyroX)], 
                              'gyroscopeYAxis': [sum(gyroY)], 
                              'gyroscopeZAxis': [sum(gyroZ)], 
                              'label': [label]}, index=[0])
    else:
      print("Invalid window type")
      return
    
    concatenated_4instarray_dataset = pd.concat([concatenated_4instarray_dataset, new_row], ignore_index=True)

    if DEBUG: print(concatenated_4instarray_dataset)
    if DEBUG: print(type(concatenated_4instarray_dataset))
  return concatenated_4instarray_dataset 


def read_normalized_data():
  for i in range(1, 11):
    df = pd.read_csv(f"datasets_for_pandas/tt_dataset_agg{i}.csv")
    if DEBUG: print(f'dataframe agg{i}: {df}')
    concatenated_normalized_dataset.append(df)

    df = pd.read_csv(f"datasets_for_pandas/tt_dataset_normal{i}.csv")
    if DEBUG: print(f'dataframe normal{i}: {df}')
    concatenated_normalized_dataset.append(df)

    f = pd.read_csv(f"datasets_for_pandas/tt_dataset_slow{i}.csv")
    if DEBUG: print(f'dataframe slow{i}: {df}')
    concatenated_normalized_dataset.append(df)

  dataframe = pd.concat(concatenated_normalized_dataset)
  return dataframe 

def read_non_normalized_data():
  for i in range(1, 11):
    df = pd.read_csv(f"datasets_for_pandas/tt_dataset_agg{i}_non_normalized.csv")
    concatenated_raw_dataset.append(df)

    df = pd.read_csv(f"datasets_for_pandas/tt_dataset_normal{i}_non_normalized.csv")
    concatenated_raw_dataset.append(df)

    f = pd.read_csv(f"datasets_for_pandas/tt_dataset_slow{i}_non_normalized.csv")
    concatenated_raw_dataset.append(df)

  dataframe = pd.concat(concatenated_raw_dataset)
  return dataframe 

def create_sum_labeled_dataset(att_type):
  concatenated_raw_dataset = read_non_normalized_data()
  if att_type == "sum": windowed_raw_dataset = get_windows(concatenated_raw_dataset, "sum")
  if att_type == "array": windowed_raw_dataset = get_windows(concatenated_raw_dataset, "array")
  return windowed_raw_dataset

def normalize_array_dataset(non_norm_data):
  norm_data = get_windows(read_normalized_data(), "array")
  norm_data['label'] = non_norm_data['label']
  return norm_data


# label: based on the sum of the accelerometer values
# attributes: arrays of size 4 (4 instances of time) for each sensor
def create_sum_label_array_att_dataset():
  non_norm_dataframe = create_sum_labeled_dataset("array")
  norm_final_dataframe = normalize_array_dataset(non_norm_dataframe)
  norm_final_dataframe.to_csv('datasets_for_training/tt_final_labels_from_accsum_post_norm.csv', index=False)
  # GENERATE THESE DATASETS TO COMPARE WITH THE FINAL DATASET
  #norm_without_c_labels = get_windows(read_normalized_data())
  #not_norm_with_c_labels = get_windows(read_non_normalized_data())
  #norm_without_c_labels.to_csv('datasets_for_training/tt_final_norm_without_c_labels.csv', index=False)
  #not_norm_with_c_labels.to_csv('datasets_for_training/tt_final_not_norm_with_c_labels.csv', index=False)


# label: based on the sum of the accelerometer values
# attributes: sum of non-normalized sensor data for the 4 instances of time
def create_sum_label_sum_att_dataset():
  non_norm_dataframe = create_sum_labeled_dataset("sum")
  non_norm_dataframe.to_csv('datasets_for_training/tt_final_labels_from_accsum_with_att_accsum.csv', index=False)

# label: based on the sum of the accelerometer values
# attributes: sum of non-normalized sensor data for the 4 instances of time (normalized post sum)
def create_sum_label_sum_att_normalized_dataset():
  non_norm_dataframe = create_sum_labeled_dataset("sum")
  non_label_columns = [col for col in non_norm_dataframe.columns if col != 'label']
  norm_dataframe = non_norm_dataframe.copy()
  # Min max normalization
  norm_dataframe[non_label_columns] = (norm_dataframe[non_label_columns] - norm_dataframe[non_label_columns].min()) / (norm_dataframe[non_label_columns].max() - norm_dataframe[non_label_columns].min())
  norm_dataframe.to_csv('datasets_for_training/tt_final_labels_from_accsum_with_att_accsum_normalized.csv', index=False)

#create_sum_label_array_att_dataset()
#create_sum_label_sum_att_dataset()
create_sum_label_sum_att_normalized_dataset()