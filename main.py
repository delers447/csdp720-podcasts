#! /usr/bin/python3

import numpy as np 
import pandas as pd 
import statistics
import time

############################################
#Member specific imports
import dan_train as dan
import tensorflow_decision_forests as tfdf

#import celia_train as celia

#import sagar_train as sagar

############################################
############################################
print("==== Loading Data ====")

train_file_path = "train.csv"
dataset_df = pd.read_csv(train_file_path)
#dataset_df = dataset_df.drop('id', axis=1)

def split_dataset(dataset, test_ratio=0.20):
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]

train_ds_pd, temp_ds_pd = split_dataset(dataset_df)
test_ds_pd, valid_ds_pd = split_dataset(temp_ds_pd, test_ratio=0.50)
train_ds_pd = train_ds_pd.drop('id', axis=1)
test_ds_pd = test_ds_pd.drop('id', axis=1)

print(f"we have {len(train_ds_pd)} in the training dataset, {len(test_ds_pd)} in the testing dataset, and {len(valid_ds_pd)} in the validation dataset.")
print(train_ds_pd.head(3))

############################################
############################################
print("==== Training Models ====")
#dan's
dan_model1 = dan.train_forest_1(train_ds_pd)
dan_model2 = dan.train_forest_2(train_ds_pd)
dan_model3 = dan.train_forest_3(train_ds_pd)

#celia's
#celia_modeX = celia.train(train_ds_pd)

#sagars's
#sagar_modeY = sagar.train(train_ds_pd)
############################################
############################################
print("==== Validating Models ====")
ids = valid_ds_pd.pop('id')
real_values = valid_ds_pd.pop('Listening_Time_minutes')

valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    valid_ds_pd,
    task = tfdf.keras.Task.REGRESSION)

#Dan's predictions
preds1 = dan_model1.predict(valid_ds)
preds2 = dan_model2.predict(valid_ds)
preds3 = dan_model3.predict(valid_ds)
#Celia's Predictions
#predsX = celia_modeX.predict(valid_ds)
#Sagar's Predictions
#predsY = sagar_modeY.predict(valid_ds)

pre_output = pd.DataFrame({'Id': ids,
                       'True Value' : real_values,
                       'dan_model1': preds1.squeeze(),
                       'dan_model2': preds2.squeeze(),
                       'dan_model3': preds3.squeeze()})
                       #'celia_modeX': predsX.squeeze()
                       #'sagar_modeX': predsY.squeeze()

ensemble = []
residuals = []
for index, row in pre_output.iterrows():
	ensemble_value = statistics.mean(list((row['dan_model1'], row['dan_model2'], row['dan_model3'])))
	ensemble.append(ensemble_value)
	residuals.append(row['True Value'] - ensemble_value)

output = pd.DataFrame({'Id': ids,
                       'True Value' : real_values,
                       'dan_model1': preds1.squeeze(),
                       'dan_model2': preds2.squeeze(),
                       'dan_model3': preds3.squeeze(),
                       #'celia_modeX': predsX.squeeze(),
                       #'sagar_modeX': predsY.squeeze(),
                       'Ensemble' : ensemble,
                       'Residual' : residuals})

print(output)
############################################
############################################
print("==== Exporting CSV ====")

with open("validation_output.csv", 'w') as f:
	f.write("Id,True Value,Ensemble,Residual,dan_model1,dan_model2,dan_model3\n")
	for index, row in output.iterrows():
		f.write(f"{row['Id']},{row['True Value']},{row['Ensemble']},{row['Residual']},{row['dan_model1']},{row['dan_model2']},{row['dan_model3']}\n")

############################################
############################################
print("==== Creating Submission ====")

competition_file_path = "test.csv"
competition_df = pd.read_csv(competition_file_path)
competition_ids = competition_df.pop('id')

competition_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    competition_df,
    task = tfdf.keras.Task.REGRESSION)

#Dan's predictions
competition_preds1 = dan_model1.predict(competition_ds)
competition_preds2 = dan_model2.predict(competition_ds)
competition_preds3 = dan_model3.predict(competition_ds)
#Celia's Predictions
#competition_predsX = celia_modeX.predict(valid_ds)
#Sagar's Predictions
#competition_predsY = sagar_modeY.predict(valid_ds)

competition_pre_output = pd.DataFrame({'Id': competition_ids,
                       'dan_model1': competition_preds1.squeeze(),
                       'dan_model2': competition_preds2.squeeze(),
                       'dan_model3': competition_preds3.squeeze()})
                       #'celia_modeX': predsX.squeeze()
                       #'sagar_modeX': predsY.squeeze()

competition_ensemble = []
for index, row in competition_pre_output.iterrows():
	competition_ensemble_value = statistics.mean(list((row['dan_model1'], row['dan_model2'], row['dan_model3'])))
	competition_ensemble.append(competition_ensemble_value)

competition_output = pd.DataFrame({'Id': competition_ids,
                                   'Ensemble' : competition_ensemble})

with open("submission.csv", 'w') as f:
	f.write("id,Listening_Time_minutes\n")
	for index, row in output.iterrows():
		f.write(f"{row['Id']},{row['Ensemble']}\n")

############################################
