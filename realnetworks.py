import networkx as nx

zkc = nx.karate_club_graph()
gt_membership = [zkc.nodes[v]['club'] for v in zkc.nodes()]

community_1 = {v for v, club in zip(zkc.nodes(), gt_membership) if club == 'Mr. Hi'}
community_2 = {v for v, club in zip(zkc.nodes(), gt_membership) if club == 'Officer'}

# Print the sets of nodes for each community
print("Community 1 (Mr. Hi):", community_1)
print("Community 2 (Officer):", community_2)