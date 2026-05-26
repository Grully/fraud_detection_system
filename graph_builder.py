# graph_builder.py
import networkx as nx

class TransactionGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def get_out_degree(self, node):
        return self.graph.out_degree(node) if self.graph.has_node(node) else 0

    def get_in_degree(self, node):
        return self.graph.in_degree(node) if self.graph.has_node(node) else 0

    def get_unique_receivers(self, sender):
        if not self.graph.has_node(sender):
            return 0
        receivers = set()
        for _, out_receiver in self.graph.out_edges(sender, data=False):
            receivers.add(out_receiver)
        return len(receivers)

    def get_avg_amount_sent(self, sender):
        if not self.graph.has_node(sender):
            return 0
        amounts = []
        for _, _, edge_data in self.graph.out_edges(sender, data=True):
            amounts.append(edge_data.get('amount', 0))
        return sum(amounts) / len(amounts) if amounts else 0