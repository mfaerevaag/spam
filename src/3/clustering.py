from pylab import *
from scipy.io import loadmat
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# load libs
import sys
sys.path.insert(0, "../../lib")
from toolbox import clusterplot
from load_data import *

# load data
X, y, N, M = load_data(False)

# Perform hierarchical/agglomerative clustering on data matrix
Method = 'average' # single, complete
Metric = 'euclidean'

Z = linkage(X, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = 2
cls = fcluster(Z, criterion='maxclust', t=Maxclust)
figure(1)
clusterplot(X, cls.reshape(cls.shape[0],1), y=y)

# Display dendrogram
max_display_levels=6
figure(2)
dendrogram(Z, truncate_mode='level', p=max_display_levels)

show()
