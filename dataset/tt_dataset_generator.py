import pandas as pd
import random

DEBUG = 0
concatenated_rows = []
concatenated_raw_dataset = []
#concatenated_4instarray_dataset = []
variance_dataset = []
window_size = 4
increment = 1



# 3 opcoes distintas para representar os novos dados
  # - Soma de valores positivos
  # - Arrays com os 4 valores
  # - Variancia de cada sensor
def get_arrays(window):
  #print(window)

  arrayAccX = [i for i in window["accelerometerXAxis"]]
  arrayAccY = [i for i in window["accelerometerYAxis"]]
  arrayAccZ = [i for i in window["accelerometerZAxis"]]
  arrayGyroX = [i for i in window["gyroscopeXAxis"]]
  arrayGyroY = [i for i in window["gyroscopeYAxis"]]
  arrayGyroZ = [i for i in window["gyroscopeZAxis"]]
  arrayLabel = [i for i in window["label"]]
  label = arrayLabel[0]

  #print(arrayAccX)
  #print(arrayAccY)
  #print(arrayAccZ)
  #print(arrayGyroX)
  #print(arrayGyroY)
  #print(arrayGyroZ)

  return arrayAccX, arrayAccY, arrayAccZ, arrayGyroX, arrayGyroY, arrayGyroZ, label 

  


def get_variance(window):
  return



def get_windows(data):
  concatenated_4instarray_dataset = pd.DataFrame() 
  for i in range(len(data) - window_size + increment):
    window = data[i:i+window_size]
    accX, accY, accZ, gyroX, gyroY, gyroZ, label = get_arrays(window)

    ## Aqui vamos pegar no datafram que ja tinhamos e junstar os novos arrays
    ## mas tivemos prblemas com o lenght dos dados por passar arrays, 
    ## corrigimos com os [] a volta da var, para ele avaliar o array so como um valor
    ## no entanto, sera que isso vai afetar a preformacnce da rede,
    ## sera que devia criar um novo df para poder meter as coisas como a lenght que
    ## deviam ter????
    new_row = pd.DataFrame({'accelerometerXAxis': [accX], 'accelerometerYAxis': [accY],  'accelerometerZAxis': [accZ],
                 'gyroscopeXAxis': [gyroX], 'gyroscopeYAxis': [gyroY], 'gyroscopeZAxis': [gyroZ], 'label': [label]}, index=[0])
    concatenated_4instarray_dataset = pd.concat([concatenated_4instarray_dataset, new_row], ignore_index=True)
    #data = data.append({'accelerometerXAxis': accX, 'accelerometerYAxis': accY,  'accelerometerZAxis': accZ,
    #             'gyroscopeXAxis': gyroX, 'gyroscopeYAxis': gyroY, 'gyroscopeZAxis': gyroZ}, ignore_index=True)


    ## fazer aqui as cenas para criar um dataframe novo e meter as cenas iguais as de cima
    ## mas com os arrays sem os [] para poderem ter a lenght original


    print(concatenated_4instarray_dataset)
    #exit()
  #return data
  return concatenated_4instarray_dataset 


# Read all data files and concat them
for i in range(1, 11):
  df = pd.read_csv(f"datasets_for_pandas/tt_dataset_agg{i}.csv")
  if DEBUG: print(f'dataframe agg{i}: {df}')
  concatenated_raw_dataset.append(df)

  df = pd.read_csv(f"datasets_for_pandas/tt_dataset_normal{i}.csv")
  if DEBUG: print(f'dataframe normal{i}: {df}')
  concatenated_raw_dataset.append(df)

  df = pd.read_csv(f"datasets_for_pandas/tt_dataset_slow{i}.csv")
  if DEBUG: print(f'dataframe slow{i}: {df}')
  concatenated_raw_dataset.append(df)



# raw dataset concatenated
concatenated_raw_dataset = pd.concat(concatenated_raw_dataset)

# 
concatenated_handled_dataset = get_windows(concatenated_raw_dataset)
concatenated_handled_dataset.to_csv('datasets_for_training/tt_labeled_arrays_dataset.csv', index=False)
