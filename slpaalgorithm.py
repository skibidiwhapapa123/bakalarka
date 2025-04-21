import numpy as np
from collections import defaultdict

def find_communities(G, T, r):
    memory = {i: {i: 1} for i in G.nodes()}

    # Step 1: Initial community assignment based on memory
    for _ in range(T):
        listeners_order = list(G.nodes())
        np.random.shuffle(listeners_order)

        for listener in listeners_order:
            speakers = list(G[listener])
            if not speakers:
                continue

            labels = defaultdict(int)
            for speaker in speakers:
                total = sum(memory[speaker].values())
                label_probs = [freq / total for freq in memory[speaker].values()]
                chosen_label = list(memory[speaker].keys())[np.random.multinomial(1, label_probs).argmax()]
                labels[chosen_label] += 1

            accepted_label = max(labels, key=labels.get)
            memory[listener][accepted_label] = memory[listener].get(accepted_label, 0) + 1

    # Step 2: Remove labels with a frequency lower than threshold r
    for node, mem in memory.items():
        to_delete = [label for label, freq in mem.items() if freq / float(T + 1) < r]
        for label in to_delete:
            del mem[label]

    # Step 3: Create initial communities from memory
    communities = defaultdict(set)
    for node, mem in memory.items():
        for label in mem.keys():
            communities[label].add(node)

    # Step 4: Remove nested communities
    community_list = list(communities.values())
    to_delete = set()
    for i, comm1 in enumerate(community_list):
        for j, comm2 in enumerate(community_list):
            if i != j and comm1.issubset(comm2):
                to_delete.add(frozenset(comm1))

    # Remove the nested communities
    final_communities = [comm for comm in community_list if frozenset(comm) not in to_delete]
    
    return final_communities
