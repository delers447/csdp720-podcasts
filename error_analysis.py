#! /usr/bin/python3

from matplotlib import pyplot as plt
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

file = "validation_output.csv"

values = []
ensemble = []
residual = []
dan1 = []
dan2 = []
dan3 = []
celia = []
sagar = []

with open(file, 'r') as f:
	for i, line in enumerate(f):
		#print(line)
		if i == 0:
			continue
		id_,true_value_,ensemble_,residual_,dan_model1_,dan_model2_,dan_model3_, sagar_model_, celia_model_ = line.split(',')
		values.append(float(true_value_))
		ensemble.append(float(ensemble_))
		residual.append(float(residual_)) 
		dan1.append(float(dan_model1_)) 
		dan2.append(float(dan_model2_)) 
		dan3.append(float(dan_model3_))
		celia.append(float(celia_model_)) 
		sagar.append(float(sagar_model_))

def calculate_residuals(values, estimates):
	#residual = value - estimation
	results = []
	for value, estimate in zip(values, estimates):
		results.append(value - estimate)
	return results

ensemble_residuals = calculate_residuals(values, ensemble)
dan1_residuals = calculate_residuals(values, dan1)
dan2_resduals = calculate_residuals(values, dan2)
dan3_residuals = calculate_residuals(values, dan3)
celia_residuals = calculate_residuals(values, celia)
sagar_residuals = calculate_residuals(values, sagar)


def make_histogram():
	graph_data = [ensemble_residuals, dan1_residuals, dan2_resduals, dan3_residuals, celia_residuals, sagar_residuals]
	labels = ['ensemble_residuals', 'dan1_residuals', 'dan2_resduals', 'dan3_residuals', 'celia_residuals', 'sagar_residuals']
	fig, axs = plt.subplots(2, 3 )

	bins = 20
	xmin, xmax = -75, 75
	ymin, ymax = 0, 25000

	axs[0, 0].hist(graph_data[0], bins=bins)
	axs[0, 0].set_title(labels[0])
	axs[0, 0].set_xlim([xmin, xmax])
	axs[0, 0].set_ylim([ymin, ymax])

	axs[0, 1].hist(graph_data[1], bins=bins)
	axs[0, 1].set_title(labels[1])
	axs[0, 1].set_xlim([xmin, xmax])
	axs[0, 1].set_ylim([ymin, ymax])

	axs[0, 2].hist(graph_data[2], bins=bins)
	axs[0, 2].set_title(labels[2])
	axs[0, 2].set_xlim([xmin, xmax])
	axs[0, 2].set_ylim([ymin, ymax])

	axs[1, 0].hist(graph_data[3], bins=bins)
	axs[1, 0].set_title(labels[3])
	axs[1, 0].set_xlim([xmin, xmax])
	axs[1, 0].set_ylim([ymin, ymax])

	axs[1, 1].hist(graph_data[4], bins=bins)
	axs[1, 1].set_title(labels[4])
	axs[1, 1].set_xlim([xmin, xmax])
	axs[1, 1].set_ylim([ymin, ymax])

	axs[1, 2].hist(graph_data[5], bins=bins)
	axs[1, 2].set_title(labels[5])
	axs[1, 2].set_xlim([xmin, xmax])
	axs[1, 2].set_ylim([ymin, ymax])
	
	plt.show()


def make_boxplot():
	graph_data = [ensemble_residuals, dan1_residuals, dan2_resduals, dan3_residuals, celia_residuals, sagar_residuals]
	fig = plt.figure(figsize =(10, 7))
	ax = fig.add_subplot(111)
	ax.boxplot(graph_data, #patch_artist = True,
	                 orientation='horizontal', showfliers=False)
	ax.set_yticklabels(['ensemble_residuals', 'dan1_residuals', 'dan2_resduals', 'dan3_residuals', 'celia_residuals', 'sagar_residuals'])
	plt.title("Residuals")
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	plt.show()

def conduct_hypothesis_test():

	f_statistic, p_value = f_oneway(ensemble_residuals, dan1_residuals, dan2_resduals, dan3_residuals, celia_residuals, sagar_residuals)

	print(f"F-statistic: {f_statistic}")
	print(f"P-value: {p_value}")

	all_scores = ensemble_residuals + dan1_residuals + dan2_resduals + dan3_residuals + celia_residuals + sagar_residuals
	all_groups = (
	    ['Ensemble'] * len(ensemble_residuals) +
	    ['Dan1'] * len(dan1_residuals) +
	    ['Dan2'] * len(dan2_resduals) +
	    ['Dan3'] * len(dan3_residuals) +
	    ['Celia'] * len(celia_residuals) +
	    ['Sagar'] * len(sagar_residuals))

	tukey = pairwise_tukeyhsd(endog=all_scores, groups=all_groups, alpha=0.05)
	print(tukey)

def print_mse():
	print(f"Ensemble's MSE: {sum(e**2 for e in ensemble_residuals) / len(ensemble_residuals)}")
	print(f"Dan1's MSE: {sum(e**2 for e in dan1_residuals) / len(dan1_residuals)}")
	print(f"Dan2's MSE: {sum(e**2 for e in dan2_resduals) / len(dan2_resduals)}")
	print(f"Dan3's MSE: {sum(e**2 for e in dan3_residuals) / len(dan3_residuals)}")
	print(f"Celia's MSE: {sum(e**2 for e in celia_residuals) / len(celia_residuals)}")
	print(f"Sagar's MSE: {sum(e**2 for e in sagar_residuals) / len(sagar_residuals)}")

make_boxplot()
make_histogram()
conduct_hypothesis_test()
print_mse()