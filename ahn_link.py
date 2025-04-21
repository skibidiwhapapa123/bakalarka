import heapq
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
            if n > 2:
                D += (l * (l - n + 1)) / ((n - 2) * (n - 1))
        return (2 / m) * D if m else 0

    def single_linkage(self, threshold=None):
        edge_index = {tuple(sorted(edge)): idx for idx, edge in enumerate(self.edges)}
        num_edges = len(self.edges)

        # Initial clusters
        clusters = {i: {i} for i in range(num_edges)}
        parent = list(range(num_edges))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        # Compute initial edge similarities
        similarities = []
        edge_sims = {}
        for node in self.adj:
            neighbors = list(self.adj[node])
            for u, v in combinations(neighbors, 2):
                e1 = tuple(sorted((node, u)))
                e2 = tuple(sorted((node, v)))
                if e1 in edge_index and e2 in edge_index:
                    i, j = edge_index[e1], edge_index[e2]
                    sim = self.edge_similarity(e1, e2)
                    if sim > 0:
                        key = frozenset((i, j))
                        edge_sims[key] = sim
                        heapq.heappush(similarities, (-sim, i, j))  # max-heap via negative sim

        best_D = 0
        best_S = 0
        best_partition = None

        while similarities:
            neg_sim, i, j = heapq.heappop(similarities)
            sim = -neg_sim

            ci = find(i)
            cj = find(j)
            if ci == cj:
                continue

            if threshold is not None and sim < threshold:
                break

            # Merge cj into ci
            parent[cj] = ci
            clusters[ci].update(clusters[cj])
            del clusters[cj]

            # Update similarities between new cluster and all others
            for ck in list(clusters.keys()):
                if ck == ci:
                    continue
                min_sim = float('-inf')
                for e1 in clusters[ci]:
                    for e2 in clusters[ck]:
                        key = frozenset((e1, e2))
                        if key in edge_sims:
                            min_sim = max(min_sim, edge_sims[key])
                if min_sim != float('-inf'):
                    heapq.heappush(similarities, (-min_sim, ci, ck))

            # Track best partition density if threshold not set
            if threshold is None:
                cid2edges = defaultdict(list)
                for idx in range(num_edges):
                    cid = find(idx)
                    cid2edges[cid].append(self.edges[idx])
                D = self.partition_density(cid2edges)
                if D > best_D:
                    best_D = D
                    best_S = sim
                    best_partition = dict(cid2edges)

        # Final partition (threshold-based or best-D)
        if threshold is not None:
            cid2edges = defaultdict(list)
            for idx in range(num_edges):
                cid = find(idx)
                cid2edges[cid].append(self.edges[idx])
        else:
            cid2edges = best_partition or defaultdict(list)
            if not cid2edges:
                for idx, edge in enumerate(self.edges):
                    cid2edges[idx].append(edge)

        edge2cid = {tuple(sorted(e)): cid for cid, edges in cid2edges.items() for e in edges}
        cid2nodes = {cid: list({i for e in edges for i in e}) for cid, edges in cid2edges.items()}

        if threshold is None:
            print("# D_max = %f\n# S_max = %f" % (best_D, best_S))
            return edge2cid, best_S, best_D, dict(cid2edges), cid2nodes
        else:
            return edge2cid, dict(cid2edges), cid2nodes

    def complete_linkage(self, threshold=None):
        edge_index = {tuple(sorted(edge)): idx for idx, edge in enumerate(self.edges)}
        num_edges = len(self.edges)

        # Initial clusters
        clusters = {i: {i} for i in range(num_edges)}
        parent = list(range(num_edges))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        # Compute initial edge similarities
        similarities = []
        edge_sims = {}
        for node in self.adj:
            neighbors = list(self.adj[node])
            for u, v in combinations(neighbors, 2):
                e1 = tuple(sorted((node, u)))
                e2 = tuple(sorted((node, v)))
                if e1 in edge_index and e2 in edge_index:
                    i, j = edge_index[e1], edge_index[e2]
                    sim = self.edge_similarity(e1, e2)
                    if sim > 0:
                        key = frozenset((i, j))
                        edge_sims[key] = sim
                        heapq.heappush(similarities, (-sim, i, j))  # max-heap via negative sim

        best_D = 0
        best_S = 0
        best_partition = None

        while similarities:
            neg_sim, i, j = heapq.heappop(similarities)
            sim = -neg_sim

            ci = find(i)
            cj = find(j)
            if ci == cj:
                continue

            if threshold is not None and sim < threshold:
                break

            # Merge cj into ci
            parent[cj] = ci
            clusters[ci].update(clusters[cj])
            del clusters[cj]

            # Update similarities between new cluster and all others
            for ck in list(clusters.keys()):
                if ck == ci:
                    continue
                min_sim = float('-inf')
                for e1 in clusters[ci]:
                    for e2 in clusters[ck]:
                        key = frozenset((e1, e2))
                        if key in edge_sims:
                            min_sim = min(min_sim, edge_sims[key])
                if min_sim != float('-inf'):
                    heapq.heappush(similarities, (-min_sim, ci, ck))

            # Track best partition density if threshold not set
            if threshold is None:
                cid2edges = defaultdict(list)
                for idx in range(num_edges):
                    cid = find(idx)
                    cid2edges[cid].append(self.edges[idx])
                D = self.partition_density(cid2edges)
                if D > best_D:
                    best_D = D
                    best_S = sim
                    best_partition = dict(cid2edges)

        # Final partition (threshold-based or best-D)
        if threshold is not None:
            cid2edges = defaultdict(list)
            for idx in range(num_edges):
                cid = find(idx)
                cid2edges[cid].append(self.edges[idx])
        else:
            cid2edges = best_partition or defaultdict(list)
            if not cid2edges:
                for idx, edge in enumerate(self.edges):
                    cid2edges[idx].append(edge)

        edge2cid = {tuple(sorted(e)): cid for cid, edges in cid2edges.items() for e in edges}
        cid2nodes = {cid: list({i for e in edges for i in e}) for cid, edges in cid2edges.items()}

        if threshold is None:
            print("# D_max = %f\n# S_max = %f" % (best_D, best_S))
            return edge2cid, best_S, best_D, dict(cid2edges), cid2nodes
        else:
            return edge2cid, dict(cid2edges), cid2nodes



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
    if linkage == "single":
        return hlc.single_linkage(threshold=threshold)
    else:
        return hlc.complete_linkage(threshold=threshold)
