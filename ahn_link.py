import heapq
from collections import defaultdict
from itertools import combinations
 
import heapq
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform

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

    def average_linkage(self, threshold=None):
        print(">>> Starting average-linkage clustering")
        print(f">>> Number of edges: {len(self.edges)}")

        edge_index = {tuple(sorted(edge)): idx for idx, edge in enumerate(self.edges)}
        num_edges = len(self.edges)

        clusters = {i: {i} for i in range(num_edges)}
        parent = list(range(num_edges))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        similarities = []
        edge_sims = {}
        active_pairs = set()

        print(">>> Computing edge similarities...")
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
                        heapq.heappush(similarities, (-sim, i, j))
                        active_pairs.add((min(i, j), max(i, j)))
                        print(f"  > Similarity({e1}, {e2}) = {sim:.4f}")

        best_D = 0
        best_S = 0
        best_partition = None

        print(">>> Starting clustering process...")
        while similarities:
            sim, i, j = heapq.heappop(similarities)
            sim = -sim
            ci = find(i)
            cj = find(j)

            if ci == cj:
                continue

            if threshold is not None and 1 - sim < threshold:
                print(f"  > Stopping: similarity {sim:.4f} below threshold {threshold}")
                break

            print(f"\n[Merge Step] Merging clusters {ci} and {cj} with similarity {sim:.4f}")
            parent[cj] = ci
            clusters[ci].update(clusters[cj])
            del clusters[cj]

            for ck in list(clusters.keys()):
                if ck == ci:
                    continue
                total_sim = 0.0
                count = 0
                for e1 in clusters[ci]:
                    for e2 in clusters[ck]:
                        key = frozenset((e1, e2))
                        if key in edge_sims:
                            total_sim += edge_sims[key]
                            count += 1
                if count > 0:
                    avg_sim = total_sim / count
                    pair = (min(ci, ck), max(ci, ck))
                    if pair not in active_pairs:
                        active_pairs.add(pair)
                        heapq.heappush(similarities, (-avg_sim, ci, ck))
                        print(f"    > Updated average similarity between clusters {ci} and {ck}: {avg_sim:.4f}")

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
            print("\n# Final Results:")
            print("# ---------------------")
            print(f"# D_max = {best_D:.6f}")
            print(f"# S_max = {best_S:.6f}")
            print(f"# Number of communities: {len(cid2edges)}")
            print("# ---------------------")
            return edge2cid, best_S, best_D, dict(cid2edges), cid2nodes
        else:
            print(f"\n# Threshold mode: Number of communities = {len(cid2edges)}")
            return edge2cid, dict(cid2edges), cid2nodes


    def single_linkage(self, threshold=None):
        print(">>> Starting single-linkage clustering")
        print(f">>> Number of edges: {len(self.edges)}")

        edge_index = {tuple(sorted(edge)): idx for idx, edge in enumerate(self.edges)}
        num_edges = len(self.edges)

        # Initial clusters (one edge per cluster)
        clusters = {i: {i} for i in range(num_edges)}
        parent = list(range(num_edges))  # Disjoint-set parent array

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        # Compute initial edge similarities
        similarities = []
        edge_sims = {}

        print(">>> Computing edge similarities...")
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
                        heapq.heappush(similarities, (-sim, i, j))
                        print(f"  > Similarity({e1}, {e2}) = {sim:.4f}")
        print("All similarities in the heap:")
        for sim, i, j in similarities:
            print(f"Similarity between nodes {i} and {j}: {-sim}")

        print(f">>> Total edge similarity pairs: {len(edge_sims)}")

        best_D = 0
        best_S = 0
        best_partition = None

        print(">>> Starting clustering process...")
        while similarities:
            neg_sim, i, j = heapq.heappop(similarities)
            sim = -neg_sim
            ci = find(i)
            cj = find(j)

            if ci == cj:
                print(f"  > Skipping pair ({i}, {j}) - already in the same cluster")
                continue

            if threshold is not None and sim < threshold:
                print(f"  > Stopping: similarity {sim:.4f} below threshold {threshold}")
                break

            print(f"\n[Merge Step]")
            print(f"Merging clusters {ci} and {cj} with similarity {sim:.4f}")
            print(f" - Cluster {ci} edges: {[self.edges[e] for e in clusters[ci]]}")
            print(f" - Cluster {cj} edges: {[self.edges[e] for e in clusters[cj]]}")
            parent[cj] = ci
            clusters[ci].update(clusters[cj])
            del clusters[cj]

            # Update similarities between the new cluster and others
            for ck in list(clusters.keys()):
                if ck == ci:
                    continue
                max_sim = float('-inf')
                for e1 in clusters[ci]:
                    for e2 in clusters[ck]:
                        key = frozenset((e1, e2))
                        if key in edge_sims:
                            max_sim = max(max_sim, edge_sims[key])
                if max_sim != float('-inf'):
                    heapq.heappush(similarities, (max_sim, ci, ck))
                    print(f"    > Updated similarity between clusters {ci} and {ck}: {max_sim:.4f}")
                    print(f"      - Cluster {ci} edges: {[self.edges[e] for e in clusters[ci]]}")
                    print(f"      - Cluster {ck} edges: {[self.edges[e] for e in clusters[ck]]}")

            # Track best partition by density
            if threshold is None:
                cid2edges = defaultdict(list)
                for idx in range(num_edges):
                    cid = find(idx)
                    cid2edges[cid].append(self.edges[idx])
                D = self.partition_density(cid2edges)
                print(f"    > Partition density D = {D:.4f}")
                if D > best_D:
                    print("    > New best partition found.")
                    best_D = D
                    best_S = sim
                    best_partition = dict(cid2edges)

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
            print("\n# Final Results:")
            print("# ---------------------")
            print(f"# D_max = {best_D:.6f}")
            print(f"# S_max = {best_S:.6f}")
            print(f"# Number of communities: {len(cid2edges)}")
            print("# ---------------------")
            return edge2cid, best_S, best_D, dict(cid2edges), cid2nodes
        else:
            print(f"\n# Threshold mode: Number of communities = {len(cid2edges)}")
            return edge2cid, dict(cid2edges), cid2nodes



    def complete_linkage(self, threshold=None):
        print(">>> Starting complete-linkage clustering")
        print(f">>> Number of edges: {len(self.edges)}")

        edge_index = {tuple(sorted(edge)): idx for idx, edge in enumerate(self.edges)}
        num_edges = len(self.edges)

        # Initial clusters (one edge per cluster)
        clusters = {i: {i} for i in range(num_edges)}
        parent = list(range(num_edges))  # Disjoint-set parent array

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        similarities = []
        edge_sims = {}
        active_pairs = set()

        print(">>> Computing edge similarities...")
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
                        heapq.heappush(similarities, (sim, i, j))
                        active_pairs.add((min(i, j), max(i, j)))
                        print(f"  > Similarity({e1}, {e2}) = {sim:.4f}")

        print(f">>> Total edge similarity pairs: {len(edge_sims)}")

        best_D = 0
        best_S = 0
        best_partition = None

        print(">>> Starting clustering process...")
        while similarities:
            sim, i, j = heapq.heappop(similarities)
            ci = find(i)
            cj = find(j)
            pair = (min(ci, cj), max(ci, cj))

            if ci == cj or pair not in active_pairs:
                continue

            active_pairs.remove(pair)

            if threshold is not None and 1 - sim < threshold:
                print(f"  > Stopping: similarity {sim:.4f} below threshold {threshold}")
                break

            print(f"\n[Merge Step]")
            print(f"Merging clusters {ci} and {cj} with similarity {sim:.4f}")
            print(f" - Cluster {ci} edges: {[self.edges[e] for e in clusters[ci]]}")
            print(f" - Cluster {cj} edges: {[self.edges[e] for e in clusters[cj]]}")
            parent[cj] = ci
            clusters[ci].update(clusters[cj])
            del clusters[cj]

            # Update similarities between new cluster and all other clusters
            for ck in list(clusters.keys()):
                if ck == ci:
                    continue
                max_sim = float('-inf')
                for e1 in clusters[ci]:
                    for e2 in clusters[ck]:
                        key = frozenset((e1, e2))
                        if key in edge_sims:
                            max_sim = min(max_sim, edge_sims[key])
                if max_sim != float('-inf'):
                    pair = (min(ci, ck), max(ci, ck))
                    if pair not in active_pairs:
                        active_pairs.add(pair)
                        heapq.heappush(similarities, (max_sim, ci, ck))
                        print(f"    > Updated similarity between clusters {ci} and {ck}: {max_sim:.4f}")
                        print(f"      - Cluster {ci} edges: {[self.edges[e] for e in clusters[ci]]}")
                        print(f"      - Cluster {ck} edges: {[self.edges[e] for e in clusters[ck]]}")

            # Track best partition by density
            if threshold is None:
                cid2edges = defaultdict(list)
                for idx in range(num_edges):
                    cid = find(idx)
                    cid2edges[cid].append(self.edges[idx])
                D = self.partition_density(cid2edges)
                print(f"    > Partition density D = {D:.4f}")
                if D > best_D:
                    print("    > New best partition found.")
                    best_D = D
                    best_S = sim
                    best_partition = dict(cid2edges)

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
            print("\n# Final Results:")
            print("# ---------------------")
            print(f"# D_max = {best_D:.6f}")
            print(f"# S_max = {best_S:.6f}")
            print(f"# Number of communities: {len(cid2edges)}")
            print("# ---------------------")
            return edge2cid, best_S, best_D, dict(cid2edges), cid2nodes
        else:
            print(f"\n# Threshold mode: Number of communities = {len(cid2edges)}")
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
    elif linkage == "complete":
        return hlc.complete_linkage(threshold=threshold)
    else:
        return hlc.average_linkage(threshold=threshold)

