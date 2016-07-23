from numpy import genfromtxt
import numpy as np
from matplotlib import pyplot as plt
import pandas
from sklearn.cluster import FeatureAgglomeration
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

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
		


BENCH_DATA = genfromtxt('Sequential_Application_SATUNSAT_track_wo_names.csv', delimiter=',')
#BENCH_DATA = BENCH_DATA.transpose()
print(BENCH_DATA.shape)
#print(np.isnan(BENCH_DATA).any())

ward = FeatureAgglomeration(linkage='average')

#print(ward.fit_predict(BENCH_DATA))
ward.fit(BENCH_DATA)
#print(ward.children_)
#print(find_feature_cluster(ward.children_, 0, 2, 300))

plt.title('SAT Feature_Agglomeration')
plot_dendrogram(ward, leaf_font_size = 12)
plt.savefig('SAT_Feature_Agglomeration.png')
plt.show()
