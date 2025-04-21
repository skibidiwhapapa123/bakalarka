import networkx as nx

def shen_modularity_verbose(G, community_list):
    m = G.number_of_edges()
    print(f"Počet hran (m): {m}")
    if m == 0:
        return 0

    degrees = dict(G.degree())
    print("Stupně vrcholů:", degrees)

    O = {}
    for cid, community in enumerate(community_list):
        for node in community:
            O[node] = O.get(node, 0) + 1
    print("Počet komunit pro každý uzel (O):", O)

    Q = 0.0
    nodes = list(G.nodes())
    for i in nodes:
        for j in nodes:
            A_ij = 1 if G.has_edge(i, j) else 0
            k_i = degrees[i]
            k_j = degrees[j]
            O_i = O.get(i, 1)
            O_j = O.get(j, 1)

            shared = 0
            for community in community_list:
                if i in community and j in community:
                    shared += 1

            if shared > 0:
                delta_Q =  shared * (A_ij - (k_i * k_j) / (2 * m)) / (O_i * O_j)
                Q += delta_Q
                print(f"({i}, {j}): A_ij={A_ij}, shared={shared}, "
                      f"k_i={k_i}, k_j={k_j}, O_i={O_i}, O_j={O_j} "
                      f"=> delta_Q={delta_Q:.4f}")

    Q /= (2 * m)
    print(f"\nShen modularita: {Q:.4f}")
    return Q

# Definice grafu
G = nx.Graph()
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)])

# Komunity
community_list = [
    {1, 2, 3},
    {1, 3, 4},
]

# Výpočet s výpisem
shen_modularity_verbose(G, community_list)
