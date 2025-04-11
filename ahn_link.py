import networkx as nx
from collections import defaultdict
from itertools import combinations


class HLC:
    def __init__(self, adj, edges):
        self.adj = adj
        self.edges = edges

    def edge_similarity(self, e1, e2, w=None):
        i, j = e1
        k, l = e2

        if len(set([i, j, k, l])) > 3:
            return 0.0

        u = j if i in (k, l) else i
        v = l if k in (i, j) else k

        Nu = self.adj[u] - {v}
        Nv = self.adj[v] - {u}
        I = Nu & Nv

        if w is None:
            return len(I) / ((len(Nu) * len(Nv)) ** 0.5) if Nu and Nv else 0.0
        else:
            sum_w = sum(w.get(tuple(sorted((u, n))), 1.0) for n in I)
            denom = (sum(w.get(tuple(sorted((u, n))), 1.0) for n in Nu) *
                     sum(w.get(tuple(sorted((v, n))), 1.0) for n in Nv)) ** 0.5
            return sum_w / denom if denom else 0.0

    def partition_density(self, cid2edges):
        m = sum(len(edges) for edges in cid2edges.values())
        D = 0
        for edges in cid2edges.values():
            if len(edges) <= 1:
                continue
            nodes = set()
            for i, j in edges:
                nodes.add(i)
                nodes.add(j)
            n = len(nodes)
            l = len(edges)
            D += (l * (l - n + 1)) / ((n - 2) * (n - 1)) if n > 2 else 0
        return (2 / m) * D if m else 0

    def single_linkage(self, threshold=None, w=None, dendro_flag=False):
        similarities = []
        edge_index = {tuple(sorted(edge)): idx for idx, edge in enumerate(self.edges)}

        for node in self.adj:
            neighbors = list(self.adj[node])
            for u, v in combinations(neighbors, 2):
                e1 = tuple(sorted((node, u)))
                e2 = tuple(sorted((node, v)))
                if e1 in edge_index and e2 in edge_index:
                    sim = self.edge_similarity(e1, e2, w)
                    similarities.append((sim, edge_index[e1], edge_index[e2]))

        similarities.sort(reverse=True)

        parent = list(range(len(self.edges)))  # parent list for union-find
        size = [1] * len(self.edges)  # to store the size of each cluster

        # To keep track of the next available cluster index for merges
        next_cluster_idx = len(self.edges)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            nonlocal next_cluster_idx
            x_root = find(x)
            y_root = find(y)
            if x_root != y_root:
                # Union by size, attach the smaller tree under the larger one
                if size[x_root] < size[y_root]:
                    x_root, y_root = y_root, x_root
                parent[y_root] = x_root
                size[x_root] += size[y_root]
                # Create a new cluster for this merged set
                new_cluster_idx = next_cluster_idx
                next_cluster_idx += 1
                return new_cluster_idx
            return None

        list_D = []
        best_D = 0
        best_S = 0
        best_partition = None
        linkage = []

        D = 0  # Ensure that D has a default value

        for sim, i, j in similarities:
            if threshold is not None and sim < threshold:
                break
            new_cluster_idx = union(i, j)
            if new_cluster_idx is not None:
                # Update the linkage with 4 columns: (distance, node1, node2, size_of_cluster)
                linkage.append((sim, i, j, size[find(i)]))
                if threshold is None:
                    cid2edges = defaultdict(list)
                    for idx, p in enumerate(parent):
                        cid = find(p)
                        cid2edges[cid].append(self.edges[idx])
                    D = self.partition_density(cid2edges)  # Assign D here

                # Update best partition based on density
                if D > best_D:
                    best_D = D
                    best_S = sim
                    best_partition = dict(cid2edges)

        if threshold is not None:
            cid2edges = defaultdict(list)
            for idx, p in enumerate(parent):
                cid = find(p)
                cid2edges[cid].append(self.edges[idx])
            edge2cid = {tuple(sorted(e)): cid for cid, edges in cid2edges.items() for e in edges}
            cid2nodes = {cid: list(set(i for e in edges for i in e)) for cid, edges in cid2edges.items()}
            return edge2cid, None, None, None, dict(cid2edges), cid2nodes

        if best_partition is None:
            best_partition = defaultdict(list)
            for idx, edge in enumerate(self.edges):
                best_partition[idx].append(edge)

        edge2cid = {tuple(sorted(e)): cid for cid, edges in best_partition.items() for e in edges}
        cid2nodes = {cid: list(set(i for e in edges for i in e)) for cid, edges in best_partition.items()}

        return edge2cid, best_S, best_D, list_D, dict(best_partition), cid2nodes, linkage

    



def run_hlc_on_nx_graph(G, threshold=None, dendro_flag=False):
    """
    Run HLC on a networkx graph `G`.

    Parameters:
        G: networkx.Graph or networkx.DiGraph
        threshold: optional float – similarity threshold for cutting dendrogram
        dendro_flag: bool – if True, store full dendrogram

    Returns:
        edge2cid: dict mapping edges to community IDs
        S_best, D_best: floats, best threshold and partition density
        list_D: list of (threshold, D) values
        cid2edges: dict of community ID to list of edges
        cid2nodes: dict of community ID to list of nodes
    """
    adj = {i: set(G.neighbors(i)) for i in G.nodes()}
    edges = [(i, j) for i, j in G.edges() if i < j]

    is_weighted = nx.is_weighted(G)
    ij2wij = None
    if is_weighted:
        ij2wij = {tuple(sorted((i, j))): d['weight'] for i, j, d in G.edges(data=True)}

    hlc = HLC(adj, edges)

    if threshold is not None:
        return hlc.single_linkage(threshold=threshold, dendro_flag=dendro_flag)
    else:
        return hlc.single_linkage(dendro_flag=dendro_flag)
