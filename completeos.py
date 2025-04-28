import networkx as nx

# Load the Karate Club graph
G = nx.karate_club_graph()


print(len(G.nodes()))

# Assign integer community IDs
club_to_id = {'Mr. Hi': 1, 'Officer': 2}

# Write edge list to network.dat
with open("network.dat", "w") as f:
    for u, v in G.edges():
        f.write(f"{u+1}\t{v+1}\n")  # +1 to match LFR-style 1-based indexing

# Write community assignments to community.dat
with open("community.dat", "w") as f:
    for node in G.nodes():
        club = G.nodes[node]['club']
        community_id = club_to_id[club]
        f.write(f"{node+1}\t{community_id}\n")  # +1 for 1-based indexing
