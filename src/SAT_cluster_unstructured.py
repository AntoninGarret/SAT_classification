from numpy import genfromtxt
import numpy as np
from matplotlib import pyplot as plt
import pandas
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage

def plot_dendrogram(model, measure, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])
    #distance = measure_to_distance(children , measure)

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

   
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
	
def measure_to_distance(children, measure):
	return [np.abs(measure[node[0]] - measure[node[1]]) for node in children]
		


BENCH_DATA = genfromtxt('Sequential_Application_SATUNSAT_track_wo_names.csv', delimiter=',')
data_average = [np.average(i) for i in BENCH_DATA]
inverse_points = [np.sum(1/i) for i in BENCH_DATA]
#data_split = [np."ecarttype"(i) for in in BENCH_DATA]


#BENCH_DATA = BENCH_DATA.transpose()
print(BENCH_DATA.shape)
#print(np.isnan(BENCH_DATA).any())

ward = AgglomerativeClustering(linkage='average')

#print(ward.fit_predict(BENCH_DATA))
ward.fit(BENCH_DATA)
#print(ward.children_)
#print(find_feature_cluster(ward.children_, 0, 2, 300))

purple_cluster = find_feature_cluster(ward.children_, 221, 3, 300)
yellow_cluster = find_feature_cluster(ward.children_, 239, 2, 300)
purple_cluster_bis = find_feature_cluster(ward.children_, 255, 3, 300)
black_cluster = find_feature_cluster(ward.children_, 170, 2, 300)
red_cluster = find_feature_cluster(ward.children_, 249, 2, 300)
green_cluster = find_feature_cluster(ward.children_, 187, 1, 300)

print(yellow_cluster)
plt.title('SAT Clustering Dendrogram unstructured')
plot_dendrogram(ward, data_average, leaf_font_size = 12)
#plt.savefig('SAT_Clustering_Dendrogram_unstructured.png')
plt.show()
