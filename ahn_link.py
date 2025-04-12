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

        if len(set([i, j, k, l])) > 3:
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
            nodes = set()
            for i, j in edges:
                nodes.add(i)
                nodes.add(j)
            n = len(nodes)
            l = len(edges)
            if n > 2:
                D += (l * (l - n + 1)) / ((n - 2) * (n - 1))
        return (2 / m) * D if m else 0

    def cluster_similarity(self, c1, c2, sim_matrix, linkage):
        sims = [
            sim_matrix[(min(e1, e2), max(e1, e2))]
            for e1 in c1 for e2 in c2
            if (min(e1, e2), max(e1, e2)) in sim_matrix
        ]
        if not sims:
            return 0.0
        if linkage == "single":
            return max(sims)
        elif linkage == "complete":
            return min(sims)
        elif linkage == "average":
            return sum(sims) / len(sims)
        return 0.0

    def linkage_clustering(self, threshold=None, linkage="single"):
        edge_index = {tuple(sorted(e)): idx for idx, e in enumerate(self.edges)}
        sim_matrix = {}

        print(linkage)

        for node in self.adj:
            for u, v in combinations(self.adj[node], 2):
                e1 = tuple(sorted((node, u)))
                e2 = tuple(sorted((node, v)))
                if e1 in edge_index and e2 in edge_index:
                    i, j = edge_index[e1], edge_index[e2]
                    sim = self.edge_similarity(e1, e2)
                    sim_matrix[(min(i, j), max(i, j))] = sim

        clusters = {i: {i} for i in range(len(self.edges))}

        best_D = 0
        best_S = 0
        best_partition = None

        while True:
            best_pair = None
            best_sim = -1
            cluster_ids = list(clusters.keys())

            for i in range(len(cluster_ids)):
                for j in range(i + 1, len(cluster_ids)):
                    c1, c2 = cluster_ids[i], cluster_ids[j]
                    sim = self.cluster_similarity(clusters[c1], clusters[c2], sim_matrix, linkage)
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (c1, c2)

            if best_pair is None or (threshold is not None and best_sim < threshold):
                break

            c1, c2 = best_pair
            clusters[c1].update(clusters[c2])
            del clusters[c2]

            # Evaluate partition density
            cid2edges = defaultdict(list)
            for cid, edge_ids in clusters.items():
                for eid in edge_ids:
                    cid2edges[cid].append(self.edges[eid])

            D = self.partition_density(cid2edges)
            if D > best_D:
                best_D = D
                best_S = best_sim
                best_partition = dict(cid2edges)

        if best_partition is None:
            best_partition = defaultdict(list)
            for idx, edge in enumerate(self.edges):
                best_partition[idx].append(edge)

        edge2cid = {tuple(sorted(e)): cid for cid, edges in best_partition.items() for e in edges}
        cid2nodes = {cid: list({i for e in edges for i in e}) for cid, edges in best_partition.items()}
        print("# D_max = %f\n# S_max = %f" % (best_D, best_S))
        return edge2cid, best_S, best_D, best_partition, cid2nodes


def link_communities(G, threshold=None, linkage="single"):
    """
    Run HLC on a networkx graph `G`.

    Parameters:
        G: networkx.Graph
        threshold: float – optional similarity threshold for stopping
        linkage: str – 'single', 'complete', or 'average'

    Returns:
        edge2cid: dict mapping edges to community IDs
        best_S, best_D: float
        cid2edges: dict of community ID to list of edges
        cid2nodes: dict of community ID to list of nodes
    """
    adj = {i: set(G.neighbors(i)) for i in G.nodes()}
    edges = list(G.edges())
    hlc = HLC(adj, edges)
    return hlc.linkage_clustering(threshold=threshold, linkage=linkage)
