import networkx as nx
import random
from collections import defaultdict


class DemonAlgorithm:
    def __init__(self, epsilon=0.25, min_community_size=3, weighted=False):
        self.epsilon = epsilon
        self.min_community_size = min_community_size
        self.weighted = weighted

    def execute(self, G):
        """
        Execute Demon algorithm on a NetworkX graph G.
        Returns a list of frozensets where each frozenset represents a community.
        """
        self.G = G

        # Initialize nodes with their own communities
        for n in self.G.nodes():
            self.G.nodes[n]['communities'] = [n]

        all_communities = {}

        total_nodes = len(list(nx.nodes(self.G)))
        actual = 0
        for ego in nx.nodes(self.G):
            percentage = float(actual * 100) / total_nodes
            actual += 1

            # Ego network (subgraph around each node)
            ego_minus_ego = nx.ego_graph(self.G, ego, 1, False)
            community_to_nodes = self.__overlapping_label_propagation(ego_minus_ego, ego)

            # Merging phase: merge small communities into larger ones
            for c in community_to_nodes.keys():
                if len(community_to_nodes[c]) > self.min_community_size:
                    actual_community = community_to_nodes[c]
                    all_communities = self.__merge_communities(all_communities, actual_community)

        # Convert all communities to frozensets
        final_communities = [frozenset(community) for community in all_communities.values()]


        return final_communities

    def __overlapping_label_propagation(self, ego_minus_ego, ego, max_iteration=100):
        """
        Propagates labels in the ego network and finds overlapping communities.
        """
        t = 0
        old_node_to_coms = {}

        while t < max_iteration:
            t += 1
            node_to_coms = {}

            nodes = list(nx.nodes(ego_minus_ego))
            random.shuffle(nodes)

            for n in nodes:
                label_freq = {}
                n_neighbors = list(nx.neighbors(ego_minus_ego, n))

                if len(n_neighbors) < 1:
                    continue

                # Frequency count of labels from neighbors
                for nn in n_neighbors:
                    communities_nn = [nn]
                    if nn in old_node_to_coms:
                        communities_nn = old_node_to_coms[nn]

                    for nn_c in communities_nn:
                        if nn_c in label_freq:
                            v = label_freq.get(nn_c)
                            if self.weighted:
                                label_freq[nn_c] = v + ego_minus_ego.edge[nn][n]['weight']
                            else:
                                label_freq[nn_c] = v + 1
                        else:
                            if self.weighted:
                                label_freq[nn_c] = ego_minus_ego.edge[nn][n]['weight']
                            else:
                                label_freq[nn_c] = 1

                # Randomly assign labels if it's the first iteration
                if t == 1:
                    if n_neighbors:
                        r_label = random.sample(list(label_freq.keys()), 1)
                        ego_minus_ego.nodes[n]['communities'] = r_label
                        old_node_to_coms[n] = r_label
                    continue

                # Majority voting for label assignment
                else:
                    labels = []
                    max_freq = -1
                    for l, c in label_freq.items():
                        if c > max_freq:
                            max_freq = c
                            labels = [l]
                        elif c == max_freq:
                            labels.append(l)

                    node_to_coms[n] = labels

                    if n not in old_node_to_coms or set(node_to_coms[n]) != set(old_node_to_coms[n]):
                        old_node_to_coms[n] = node_to_coms[n]
                        ego_minus_ego.nodes[n]['communities'] = labels

            t += 1

        # Build communities based on propagated labels
        community_to_nodes = {}
        for n in nx.nodes(ego_minus_ego):
            if len(list(nx.neighbors(ego_minus_ego, n))) == 0:
                ego_minus_ego.nodes[n]['communities'] = [n]

            c_n = ego_minus_ego.nodes[n]['communities']
            for c in c_n:
                if c in community_to_nodes:
                    community_to_nodes[c].append(n)
                else:
                    community_to_nodes[c] = [n, ego]

        return community_to_nodes

    def __merge_communities(self, communities, actual_community):
   
        # If the community is already present, return
        if tuple(actual_community) in communities:
            return communities

        inserted = False
        for test_community in communities.items():
            union = self.__generalized_inclusion(actual_community, test_community[0])
            if union is not None:
                communities.pop(test_community[0])  # Remove the old community
                communities[tuple(sorted(union))] = union  # Merge the communities
                inserted = True
                break

        if not inserted:
            communities[tuple(sorted(actual_community))] = actual_community  # Add the new community

        return communities

    def __generalized_inclusion(self, c1, c2):
        """
        Determines if two communities should be merged based on overlap.
        """
        intersection = set(c2) & set(c1)
        smaller_set = min(len(c1), len(c2))

        if len(intersection) == 0:
            return None

        if smaller_set != 0:
            res = float(len(intersection)) / float(smaller_set)

        if res >= self.epsilon:
            return set(c2) | set(c1)
        return None
