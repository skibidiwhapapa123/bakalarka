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
        print("Initializing single linkage clustering...\n")

        edge_index = {tuple(sorted(edge)): idx for idx, edge in enumerate(self.edges)}
        num_edges = len(self.edges)
        print(f"Total number of edges: {num_edges}")

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

        print("\nComputing initial edge similarities...")
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
                        print(f"Similarity between {e1} (#{i}) and {e2} (#{j}) = {sim:.4f}")

        print(f"\nTotal initial similarities calculated: {len(similarities)}")

        best_D = 0
        best_S = 0
        best_partition = None

        print("\nStarting clustering loop...\n")
        while similarities:
            neg_sim, i, j = heapq.heappop(similarities)
            sim = -neg_sim
            print(f"Processing pair (#{i}, #{j}) with similarity {sim:.4f}")

            ci = find(i)
            cj = find(j)

            if ci == cj:
                print(f"  Edges already in the same cluster (#{ci}). Skipping.")
                continue

            if threshold is not None and sim < threshold:
                print(f"  Similarity {sim:.4f} below threshold {threshold}. Stopping.")
                break

            print(f"  Merging cluster #{cj} into cluster #{ci}")
            parent[cj] = ci
            clusters[ci].update(clusters[cj])
            del clusters[cj]

            print(f"  New cluster #{ci} size: {len(clusters[ci])}")

            # Update similarities between the new cluster and all others
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
                    heapq.heappush(similarities, (-max_sim, ci, ck))
                    print(f"  Updated similarity between cluster #{ci} and #{ck}: {max_sim:.4f}")

            # Track best partition by density
            if threshold is None:
                cid2edges = defaultdict(list)
                for idx in range(num_edges):
                    cid = find(idx)
                    cid2edges[cid].append(self.edges[idx])
                D = self.partition_density(cid2edges)
                print(f"  Current partition density: {D:.4f}")
                if D > best_D:
                    best_D = D
                    best_S = sim
                    best_partition = dict(cid2edges)
                    print(f"  >> New best partition found! D = {D:.4f}, S = {sim:.4f}")

        print("\nClustering complete.\n")

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
            print("# Final Results:")
            print("# ---------------------")
            print("# D_max = %f" % best_D)
            print("# S_max = %f" % best_S)
            print("# Number of communities: %d" % len(cid2edges))
            print("# ---------------------")
            return edge2cid, best_S, best_D, dict(cid2edges), cid2nodes
        else:
            print("# Threshold mode: Number of communities = %d" % len(cid2edges))
            return edge2cid, dict(cid2edges), cid2nodes


    def complete_linkage(self, threshold=None):
        print("Initializing complete linkage clustering...\n")



        # Map each edge to a unique index
        edge_index = {tuple(sorted(edge)): idx for idx, edge in enumerate(self.edges)}
        num_edges = len(self.edges)
        print(f"Total number of edges: {num_edges}")

        # Initialize each edge in its own cluster
        clusters = {i: {i} for i in range(num_edges)}
        parent = list(range(num_edges))  # Disjoint-set parent array

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        similarities = []
        edge_sims = {}

        print("\nComputing initial edge similarities...")
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
                        print(f"Similarity between {e1} (#{i}) and {e2} (#{j}) = {sim:.4f}")

        print(f"\nTotal initial similarities calculated: {len(similarities)}")

        best_D = 0
        best_S = 0
        best_partition = None

        print(heapq)

        print("\nStarting clustering loop...\n")
        while similarities:
            sim, i, j = heapq.heappop(similarities)
            ci = find(i)
            cj = find(j)

            print(f"Processing pair (#{i}, #{j}) with similarity {sim:.4f}")

            if ci == cj:
                print(f"  Edges already in the same cluster (#{ci}). Skipping.")
                continue

            if threshold is not None and sim > threshold:
                print(f"  Similarity {sim:.4f} below threshold {threshold}. Stopping.")
                break

            print(f"  Merging cluster #{cj} into cluster #{ci}")
            parent[cj] = ci
            clusters[ci].update(clusters[cj])
            del clusters[cj]

            print(f"  New cluster #{ci} size: {len(clusters[ci])}")

            # Update similarities with other clusters
            for ck in list(clusters.keys()):
                if ck == ci:
                    continue
                min_sim = float('inf')
                found_sim = False
                for e1 in clusters[ci]:
                    for e2 in clusters[ck]:
                        key = frozenset((e1, e2))
                        if key in edge_sims:
                            min_sim = min(min_sim, edge_sims[key])
                            found_sim = True
                if found_sim:
                    heapq.heappush(similarities, (min_sim, ci, ck))
                    print(f"  Updated similarity between cluster #{ci} and #{ck}: {min_sim:.4f}")

            if threshold is None:
                cid2edges = defaultdict(list)
                for idx in range(num_edges):
                    cid = find(idx)
                    cid2edges[cid].append(self.edges[idx])
                D = self.partition_density(cid2edges)
                print(f"  Current partition density: {D:.4f}")
                if D > best_D:
                    best_D = D
                    best_S = sim
                    best_partition = dict(cid2edges)
                    print(f"  >> New best partition found! D = {D:.4f}, S = {sim:.4f}")

        print("\nClustering complete.\n")

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
            print("# Final Results:")
            print("# ---------------------")
            print("# D_max = %f" % best_D)
            print("# S_max = %f" % best_S)
            print("# Number of communities: %d" % len(cid2edges))
            print("# ---------------------")
            return edge2cid, best_S, best_D, dict(cid2edges), cid2nodes
        else:
            print("# Threshold mode: Number of communities = %d" % len(cid2edges))
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
