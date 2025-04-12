import networkx as nx
from collections import defaultdict
from itertools import combinations


class HLC:
    def __init__(self, adj, edges):
        self.adj = adj
        self.edges = edges

    def edge_similarity(self, e1, e2):
        i, j = e1
        k, l = e2

        if len({i, j, k, l}) > 3:
            return 0.0

        u = j if i in (k, l) else i
        v = l if k in (i, j) else k

        Nu = self.adj[u] | {u}
        Nv = self.adj[v] | {v}
        I = Nu & Nv
        U = Nu | Nv

        return len(I) / len(U) if U else 0.0

    def partition_density(self, cid2edges):
        m = sum(len(edges) for edges in cid2edges.values())
        D = 0
        for edges in cid2edges.values():
            if len(edges) <= 1:
                continue
            nodes = {i for e in edges for i in e}
            n = len(nodes)
            l = len(edges)
            D += (l * (l - n + 1)) / ((n - 2) * (n - 1)) if n > 2 else 0
        return (2 / m) * D if m else 0

    def single_linkage(self, threshold=None):
        similarities = []
        edge_index = {tuple(sorted(edge)): idx for idx, edge in enumerate(self.edges)}
        
        for node in self.adj:
            neighbors = list(self.adj[node])
            for u, v in combinations(neighbors, 2):
                e1 = tuple(sorted((node, u)))
                e2 = tuple(sorted((node, v)))
                if e1 in edge_index and e2 in edge_index:
                    sim = self.edge_similarity(e1, e2)
                    similarities.append((sim, edge_index[e1], edge_index[e2]))

        similarities.sort(reverse=True)

        parent = list(range(len(self.edges)))
        size = [1] * len(self.edges)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            x_root = find(x)
            y_root = find(y)
            if x_root != y_root:
                if size[x_root] < size[y_root]:
                    x_root, y_root = y_root, x_root
                parent[y_root] = x_root
                size[x_root] += size[y_root]
                return True
            return False

        best_D = 0
        best_S = 0
        best_partition = None

        for sim, i, j in similarities:
            if threshold is not None and sim < threshold:
                break
            if union(i, j):
                if threshold is None:
                    cid2edges = defaultdict(list)
                    for idx, p in enumerate(parent):
                        cid = find(p)
                        cid2edges[cid].append(self.edges[idx])
                    D = self.partition_density(cid2edges)
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
            cid2nodes = {cid: list({i for e in edges for i in e}) for cid, edges in cid2edges.items()}
            return edge2cid, dict(cid2edges), cid2nodes

        if best_partition is None:
            best_partition = defaultdict(list)
            for idx, edge in enumerate(self.edges):
                best_partition[idx].append(edge)

        edge2cid = {tuple(sorted(e)): cid for cid, edges in best_partition.items() for e in edges}
        cid2nodes = {cid: list({i for e in edges for i in e}) for cid, edges in best_partition.items()}
        print("# D_max = %f\n# S_max = %f" % (best_D, best_S))
        return edge2cid, best_S, best_D, dict(best_partition), cid2nodes


def run_hlc_on_nx_graph(G, threshold=None):
    """
    Run HLC on a NetworkX graph G.

    Parameters:
        G: networkx.Graph (must be unweighted)
        threshold: optional float – similarity threshold for cutting dendrogram

    Returns:
        edge2cid: dict mapping edges to community IDs
        S_best, D_best: floats, best threshold and partition density
        cid2edges: dict of community ID to list of edges
        cid2nodes: dict of community ID to list of nodes
    """
    adj = {i: set(G.neighbors(i)) for i in G.nodes()}
    edges = list(G.edges())
    hlc = HLC(adj, edges)

    return hlc.single_linkage(threshold=threshold)
