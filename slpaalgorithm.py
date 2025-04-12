import numpy as np
from collections import defaultdict

def find_communities(G, T, r):
    memory = {i: {i: 1} for i in G.nodes()}

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

    for node, mem in memory.items():
        to_delete = [label for label, freq in mem.items() if freq / float(T + 1) < r]
        for label in to_delete:
            del mem[label]

    communities = defaultdict(set)
    for node, mem in memory.items():
        for label in mem.keys():
            communities[label].add(node)

    return [frozenset(nodes) for nodes in communities.values()]