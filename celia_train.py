#! /usr/bin/python3


import numpy as np 
import pandas as pd 
import catboost
from sklearn.preprocessing import LabelEncoder

def get_model(train, test):

	for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage']:
		median_val = train[col].median()
		train[col].fillna(median_val, inplace=True)
		test[col].fillna(median_val, inplace=True)

	cat_cols = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
	for col in cat_cols:
		le = LabelEncoder()
		train[col] = le.fit_transform(train[col])
		test[col] = le.transform(test[col])

	#New Features
	train['popularity_ratio'] = (train['Host_Popularity_percentage'] + train['Guest_Popularity_percentage']) / 2
	test['popularity_ratio'] = (test['Host_Popularity_percentage'] + test['Guest_Popularity_percentage']) / 2

	train['length_per_ad'] = train['Episode_Length_minutes'] / (train['Number_of_Ads'] + 1)
	test['length_per_ad'] = test['Episode_Length_minutes'] / (test['Number_of_Ads'] + 1)

	#Features
	features = [col for col in train.columns if col not in ['id', 'Episode_Title', 'Listening_Time_minutes']]
	x_train = train[features]
	y_train = train['Listening_Time_minutes']
	x_test = test[features]
	y_test = test['Listening_Time_minutes']

	cat_model = catboost.CatBoostRegressor(
		iterations=10000,
		learning_rate=0.01,
		depth=8,
		loss_function='RMSE',
		early_stopping_rounds=100,
		random_state=42,
		verbose=100)

	cat_model.fit(x_train, y_train, eval_set=(x_test, y_test))
	
	return cat_model