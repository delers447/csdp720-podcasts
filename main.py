#! /usr/bin/python3

import numpy as np 
import pandas as pd 
import statistics
import time

############################################
#Member specific imports
import dan_train as dan
import tensorflow_decision_forests as tfdf

import celia_train as celia
import catboost as cd
from sklearn.preprocessing import LabelEncoder

#import sagar_train as sagar
import sagar_train as sagar
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.preprocessing import TargetEncoder


############################################
############################################
print("==== Loading Data ====")

train_file_path = "train.csv"
dataset_df = pd.read_csv(train_file_path)

def split_dataset(dataset, test_ratio=0.20):
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]

train_ds_pd, temp_ds_pd = split_dataset(dataset_df)
test_ds_pd, valid_ds_pd = split_dataset(temp_ds_pd, test_ratio=0.50)

dan_train = train_ds_pd.copy()
dan_test  = test_ds_pd.copy()
dan_valid = valid_ds_pd.copy()

celia_train = train_ds_pd.copy()
celia_test  = test_ds_pd.copy()
celia_valid = valid_ds_pd.copy()

sagar_train = train_ds_pd.copy()
sagar_test  = test_ds_pd.copy()
sagar_valid = valid_ds_pd.copy()

print(f"we have {len(train_ds_pd)} in the training dataset, {len(test_ds_pd)} in the testing dataset, and {len(valid_ds_pd)} in the validation dataset.")
print(train_ds_pd.head(3))

############################################
############################################
print("==== Training Models ====")
#dan's 
dan_train_ds_pd = dan_train.drop('id', axis=1)
dan_test_ds_pd  = dan_test.drop('id', axis=1)
print("===Training Dan's Model 1===")
dan_model1 = dan.train_forest_1(dan_train_ds_pd)
print("===Training Dan's Model 2===")
dan_model2 = dan.train_forest_2(dan_train_ds_pd)
print("===Training Dan's Model 3===")
dan_model3 = dan.train_forest_3(dan_train_ds_pd)

#celia's
print("===Training Celia's Model===")
celia_model = celia.get_model(celia_train, celia_test)

#sagars's
print("===Training Sagar's Model===")
sagar_model = sagar.get_model(sagar_train, sagar_test)

############################################
############################################
print("==== Validating Models ====")

#Dan's PreProcessing
ids = dan_valid.pop('id')
real_values = dan_valid.pop('Listening_Time_minutes')

dan_valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
                    dan_valid,
                    task = tfdf.keras.Task.REGRESSION)

#Dan's predictions
preds1 = dan_model1.predict(dan_valid_ds)
print("==== Dan's Predictions #1 ====")
for i in range(10):
    print(preds1[i])
time.sleep(5)

preds2 = dan_model2.predict(dan_valid_ds)
print("==== Dan's Predictions #2 ====")
for i in range(10):
    print(preds2[i])
time.sleep(5)

preds3 = dan_model3.predict(dan_valid_ds)
print("==== Dan's Predictions #3 ====")
for i in range(10):
    print(preds3[i])
time.sleep(5)

#Celia's PreProcessing
for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage']:
    median_val = celia_valid[col].median()
    celia_valid[col].fillna(median_val, inplace=True)
cat_cols = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
for col in cat_cols:
    le = LabelEncoder()
    celia_valid[col] = le.fit_transform(celia_valid[col])
celia_valid['popularity_ratio'] = (celia_valid['Host_Popularity_percentage'] + celia_valid['Guest_Popularity_percentage']) / 2
celia_valid['length_per_ad'] = celia_valid['Episode_Length_minutes'] / (celia_valid['Number_of_Ads'] + 1)
features = [col for col in celia_valid.columns if col not in ['id', 'Episode_Title', 'Listening_Time_minutes']]
celia_valid = celia_valid[features]

#Celia's Predictions
preds5 = celia_model.predict(celia_valid)
print("==== Celia's Predictions ====")
for i in range(10):
    print(preds5[i])
time.sleep(5)

#Sagar's PreProcessing
sagar_valid = sagar.preprocess_data(sagar_valid)
sagar_valid = sagar_valid.drop(columns=['id', 'Episode_Title', 'Listening_Time_minutes'])

#Sagar's Predictions
preds4 = sagar_model.predict(sagar_valid)
print("==== Sagar's Predictions ====")
for i in range(10):
    print(preds4[i])
time.sleep(5)

pre_output = pd.DataFrame({'Id': ids,
                       'True Value' : real_values,
                       'dan_model1': preds1.squeeze(),
                       'dan_model2': preds2.squeeze(),
                       'dan_model3': preds3.squeeze(),
                       'celia_model': preds5,
                       'sagar_model': preds4 })

ensemble = []
residuals = []
for index, row in pre_output.iterrows():
	ensemble_value = statistics.mean(list((row['dan_model1'], row['dan_model2'], row['dan_model3'], row['sagar_model'], row['celia_model'])))
	ensemble.append(ensemble_value)
	residuals.append(row['True Value'] - ensemble_value)

output = pd.DataFrame({'Id': ids,
                       'True Value' : real_values,
                       'dan_model1': preds1.squeeze(),
                       'dan_model2': preds2.squeeze(),
                       'dan_model3': preds3.squeeze(),
                       'celia_model': preds5,
                       'sagar_model': preds4,
                       'Ensemble' : ensemble,
                       'Residual' : residuals})

print(output)
############################################
############################################
print("==== Exporting CSV ====")

with open("validation_output.csv", 'w') as f:
	f.write("Id,True Value,Ensemble,Residual,dan_model1,dan_model2,dan_model3, sagar_model, celia_model\n")
	for index, row in output.iterrows():
		f.write(f"{row['Id']},{row['True Value']},{row['Ensemble']},{row['Residual']},{row['dan_model1']},{row['dan_model2']},{row['dan_model3']},{row['sagar_model']},{row['celia_model']}\n")

############################################
############################################
print("==== Creating Submission ====")

competition_file_path = "test.csv"
competition_df = pd.read_csv(competition_file_path)

#Dan Preprocessing
dan_competition_df = competition_df.copy()
competition_ids = dan_competition_df.pop('id')

competition_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    dan_competition_df,
    task = tfdf.keras.Task.REGRESSION)

#Dan's predictions
competition_preds1 = dan_model1.predict(competition_ds)
competition_preds2 = dan_model2.predict(competition_ds)
competition_preds3 = dan_model3.predict(competition_ds)

#Celia's Preprocessing
celia_competition_df = competition_df.copy()
for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage']:
    median_val = celia_competition_df[col].median()
    celia_competition_df[col].fillna(median_val, inplace=True)
cat_cols = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
for col in cat_cols:
    le = LabelEncoder()
    celia_competition_df[col] = le.fit_transform(celia_competition_df[col])
celia_competition_df['popularity_ratio'] = (celia_competition_df['Host_Popularity_percentage'] + celia_competition_df['Guest_Popularity_percentage']) / 2
celia_competition_df['length_per_ad'] = celia_competition_df['Episode_Length_minutes'] / (celia_competition_df['Number_of_Ads'] + 1)
features = [col for col in celia_competition_df.columns if col not in ['id', 'Episode_Title']]
celia_competition_df = celia_competition_df[features]

#Celia's Predictions
competition_preds5 = celia_model.predict(celia_competition_df)

#Sagar's Preprocessing:
sagar_competition_df = competition_df.copy()
sagars_valid_df = sagar.preprocess_data(sagar_competition_df)
sagars_valid_df = sagars_valid_df.drop(columns=['id', 'Episode_Title'])

#Sagar's Predictions
competition_preds4 = sagar_model.predict(sagars_valid_df)

competition_pre_output = pd.DataFrame({'Id': competition_ids,
                       'dan_model1': competition_preds1.squeeze(),
                       'dan_model2': competition_preds2.squeeze(),
                       'dan_model3': competition_preds3.squeeze(),
                       'celia_model': competition_preds5,
                       'sagar_model': competition_preds4 })

competition_ensemble = []
for index, row in competition_pre_output.iterrows():
	competition_ensemble_value = statistics.mean(list((row['dan_model1'], row['dan_model2'], row['dan_model3'],row['sagar_model'],row['celia_model'])))
	competition_ensemble.append(competition_ensemble_value)

competition_output = pd.DataFrame({'Id': competition_ids,
                                   'Ensemble' : competition_ensemble})

with open("submission.csv", 'w') as f:
	f.write("id,Listening_Time_minutes\n")
	for index, row in competition_output.iterrows():
		f.write(f"{row['Id']},{row['Ensemble']}\n")

############################################
print("Processing completed.")
