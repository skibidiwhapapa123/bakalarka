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

    def linkage_clustering(self, linkage='single', threshold=None):
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
        cluster_edges = {i: {i} for i in range(len(self.edges))}
        edge_sims = {frozenset((i, j)): sim for sim, i, j in similarities}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def cluster_similarity(c1, c2):
            sims = []
            for e1 in cluster_edges[c1]:
                for e2 in cluster_edges[c2]:
                    if e1 == e2:
                        continue
                    key = frozenset((e1, e2))
                    if key in edge_sims:
                        sims.append(edge_sims[key])
            if not sims:
                return 0.0
            if linkage == 'single':
                return max(sims)
            elif linkage == 'complete':
                return min(sims)
            elif linkage == 'average':
                return sum(sims) / len(sims)
            else:
                raise ValueError(f"Unknown linkage method: {linkage}")

        best_D = 0
        best_S = 0
        best_partition = None

        for sim, i, j in similarities:
            ci = find(i)
            cj = find(j)
            if ci == cj:
                continue
            sim_score = cluster_similarity(ci, cj)
            if threshold is not None and sim_score < threshold:
                break
            # Merge cj into ci
            parent[cj] = ci
            cluster_edges[ci].update(cluster_edges[cj])
            del cluster_edges[cj]

            if threshold is None:
                cid2edges = defaultdict(list)
                for idx in range(len(self.edges)):
                    cid = find(idx)
                    cid2edges[cid].append(self.edges[idx])
                D = self.partition_density(cid2edges)
                if D > best_D:
                    best_D = D
                    best_S = sim_score
                    best_partition = dict(cid2edges)

        if threshold is not None:
            cid2edges = defaultdict(list)
            for idx in range(len(self.edges)):
                cid = find(idx)
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



def link_communities(G, threshold=None, linkage="single"):
    """
    Run HLC on a networkx graph G.

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
    return hlc.linkage_clustering(linkage=linkage, threshold=threshold)