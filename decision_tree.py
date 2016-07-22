from sklearn import tree
from numpy import genfromtxt
from sklearn.externals.six import StringIO  
import pydot 

Bench_data = genfromtxt('Sequential_Application_SATUNSAT_track_results2.csv', delimiter=',')
#Bench_data = Bench_data.transpose()
print(Bench_data.shape)
Bench_labels = [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0
,0,0,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1
,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,0,0,0,0,1,0,0,1,1,1,1,0,1,0,1,1,0,1,1,0,1
,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0
,0,1,0,0,1,0,0,0,1,0,1,1,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0
,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1
,0,1,1,1,1,1,0,0,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0
,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,1,0,0,1,0,0,1,1,1
,1,0,0,0]


dt = tree.DecisionTreeClassifier()
dt = dt.fit(Bench_data, Bench_labels)

from sklearn.externals.six import StringIO
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(dt, out_file=f)
dot_data = StringIO() 
tree.export_graphviz(dt, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("iris.pdf") 

