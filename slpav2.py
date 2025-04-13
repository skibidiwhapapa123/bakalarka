import networkx as nx
import random
from collections import defaultdict

class SLPA:
    def __init__(self, graph, max_iterations=10, threshold=0.01):
        self.graph = graph
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.node_communities = {node: set([node]) for node in graph.nodes}  # Start with each node in its own community
        self.communities_count = defaultdict(int)
        
    def run(self):
        # Step 1: Initial random messages
        messages = {node: set([random.randint(0, 1000)]) for node in self.graph.nodes}
        
        for iteration in range(self.max_iterations):
            # Step 2: Message passing phase
            new_messages = {}
            for node in self.graph.nodes:
                neighbors = list(self.graph.neighbors(node))
                neighbor_messages = []
                for neighbor in neighbors:
                    # Collect messages from neighbors
                    neighbor_messages.extend(messages[neighbor])
                
                # Add current node's own community message
                neighbor_messages.extend(messages[node])

                # Step 3: Voting phase
                if neighbor_messages:
                    new_message = max(set(neighbor_messages), key=neighbor_messages.count)
                else:
                    new_message = random.randint(0, 1000)
                new_messages[node] = {new_message}
                
            # Step 4: Community update phase
            for node in new_messages:
                self.node_communities[node].update(new_messages[node])
                
            # Step 5: Remove nested communities (based on majority)
            for node, communities in self.node_communities.items():
                if len(communities) > 1:
                    # Remove nested communities, keeping the "dominant" one
                    dominant_community = max(communities, key=lambda comm: self.communities_count[comm])
                    self.node_communities[node] = {dominant_community}
                
            # Update communities count
            for node in self.node_communities:
                for community in self.node_communities[node]:
                    self.communities_count[community] += 1
            
            # Check for convergence
            max_diff = max(len(self.node_communities[node].symmetric_difference(new_messages[node])) 
                           for node in self.node_communities)
            
            if max_diff < self.threshold:
                print(f"Converged after {iteration+1} iterations.")
                break
            messages = new_messages  # Update messages for the next iteration
            
        return self.node_communities
