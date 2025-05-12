#! /usr/bin/python3


import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.preprocessing import TargetEncoder

def preprocess_data(df):
    df['Episode_Num'] = df['Episode_Title'].str[8:].astype('category')
    df.drop(columns=['Episode_Title'])

    categorical_cols = ['Episode_Num', 'Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
    for cat_col in categorical_cols:
        df[cat_col] = df[cat_col].astype('category')

    df['Is_Weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype('int')
    df['Is_High_Host_Popularity'] = (df['Host_Popularity_percentage'] > 70).astype(int)
    df['Is_High_Guest_Popularity'] = (df['Guest_Popularity_percentage'] > 70).astype(int)
    df['Host_Guest_Popularity_Gap'] = df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage']
    df['Ad_Density'] = df['Number_of_Ads'] / df['Episode_Length_minutes']
    df['Ad_Density'].replace([np.inf, -np.inf], np.nan, inplace=False)
    df['Is_Long_Episode'] = (df['Episode_Length_minutes'] > 60).astype(int)

    return df


def get_model(train_dataset, test_dataset):
	train_df = preprocess_data(train_dataset)
	test_df  = preprocess_data(test_dataset)

	x_train = train_df.drop(columns=['id', 'Episode_Title', 'Listening_Time_minutes'])
	y_train = train_df['Listening_Time_minutes']

	x_test = test_df.drop(columns=['id', 'Episode_Title', 'Listening_Time_minutes'])
	y_test = test_df['Listening_Time_minutes']

	model_params = {
		'n_estimators': 5000,
		'learning_rate': 0.01,
		'max_depth': 8,
		'subsample': 0.8,
		'colsample_bytree': 0.8,
		'objective': 'rmse',
		'random_state': 42,
		'device': 'gpu',
		'gpu_platform_id': 2,
		'gpu_device_id': 2
		}

	model = LGBMRegressor(**model_params)
	model.fit(x_train, y_train,
		eval_set=[(x_test, y_test)],
		callbacks=[early_stopping(stopping_rounds=2500),log_evaluation(period=500) ])

	return model
