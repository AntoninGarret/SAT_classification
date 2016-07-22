from numpy import genfromtxt
import numpy as np
from matplotlib import pyplot as plt
import pandas
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
   
def find_feature(childrens, index):
	for j in range(len(childrens)):
		for i in range(2):
			if childrens[j][i] == index:
				return j
	print("not found")
    
def rec_cluster(childrens, node, sample_size):
	if node[0] < sample_size:
		left_son = [node[0]]
	else:
		left_son = rec_cluster(childrens, childrens[node[0] - sample_size], sample_size)
	if node[1] < sample_size:
		right_son = [node[1]]
	else:
		right_son = rec_cluster(childrens, childrens[node[1] - sample_size], sample_size)
	return left_son + right_son
    
def find_cluster(childrens, start, height, sample_size):
	res = start
	for i in range(height - 1):
		res = find_feature(childrens, res + sample_size)
	cluster = rec_cluster(childrens, childrens[res], sample_size)
	cluster.sort()
	return cluster
	
def find_feature_cluster(children, feature, height, sample_size):
	return find_cluster(children, find_feature(children, feature), height, sample_size)
	
def measure_to_distance(measure):
	n = len(measure)
	distance = [[np.abs(measure[i] - measure[j]) for i in range(j - 1)] for j in range(n)]
	condensed_distance = []
	for i in range(n):
		condensed_distance += distance[i]
	return condensed_distance
	
		


BENCH_DATA = genfromtxt('Sequential_Application_SATUNSAT_track_results_wo_names.csv', delimiter=',')
data_average = [np.average(i) for i in BENCH_DATA]
inverse_points = [np.sum(1/i) for i in BENCH_DATA]
data_std = [np.std(i) for i in BENCH_DATA]


#BENCH_DATA = BENCH_DATA.transpose()
print(BENCH_DATA.shape)
#print(np.isnan(BENCH_DATA).any())

linkage_matrix = linkage(measure_to_distance(data_std), method = 'average')

plt.title('SAT Clustering Dendrogram unstructed')
dendrogram(linkage_matrix, leaf_font_size = 12)
plt.show()
oo
