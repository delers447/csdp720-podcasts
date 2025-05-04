#! /usr/bin/python3

import tensorflow as tf
import tensorflow_decision_forests as tfdf

def train_forest_1(dataset):
	label = 'Listening_Time_minutes'
	train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label=label, task = tfdf.keras.Task.REGRESSION)

	rf1 = tfdf.keras.RandomForestModel(task = tfdf.keras.Task.REGRESSION)
	rf1.compile(metrics=["mse"]) 
	rf1.fit(x=train_ds)

	return rf1

def train_forest_2(dataset):
	label = 'Listening_Time_minutes'
	train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label=label, task = tfdf.keras.Task.REGRESSION)

	rf2 = tfdf.keras.GradientBoostedTreesModel(task = tfdf.keras.Task.REGRESSION)
	rf2.compile(metrics=["mse"]) 
	rf2.fit(x=train_ds)

	return rf2

def train_forest_3(dataset):
	label = 'Listening_Time_minutes'
	train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label=label, task = tfdf.keras.Task.REGRESSION)

	rf3 = tfdf.keras.CartModel(task = tfdf.keras.Task.REGRESSION)
	rf3.compile(metrics=["mse"]) 
	rf3.fit(x=train_ds)

	return rf3