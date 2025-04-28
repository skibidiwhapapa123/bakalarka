import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform

# Step 1: Define the graph
edges = [(1,2), (2,3), (3,1), (3,4), (4,5), (5,6), (6,4)]
G = nx.Graph()
G.add_edges_from(edges)

# Step 2: Get edge list and prepare neighborhood sets
edge_list = list(G.edges())
edge_idx = {e: i for i, e in enumerate(edge_list)}

# For each edge, get the set of nodes connected to its endpoints (inclusive)
def edge_neighbors(edge):
    u, v = edge
    neighbors = set(G[u]) | set(G[v]) | {u, v}
    return neighbors

neighbor_sets = [edge_neighbors(e) for e in edge_list]

# Step 3: Compute pairwise Jaccard distances between edges
def jaccard_distance(set1, set2):
    return 1 - len(set1 & set2) / len(set1 | set2)

n = len(edge_list)
distance_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        dist = jaccard_distance(neighbor_sets[i], neighbor_sets[j])
        distance_matrix[i, j] = distance_matrix[j, i] = dist

# Step 4: Hierarchical clustering (single and complete)
condensed_dist = squareform(distance_matrix)

linkage_single = linkage(condensed_dist, method='single')
linkage_complete = linkage(condensed_dist, method='complete')

# Step 5: Plot dendrograms
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
dendrogram(linkage_single, labels=[str(e) for e in edge_list])
plt.title("Single Linkage Clustering of Edges")

plt.subplot(1, 2, 2)
dendrogram(linkage_complete, labels=[str(e) for e in edge_list])
plt.title("Complete Linkage Clustering of Edges")

plt.tight_layout()
plt.show()
