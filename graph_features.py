# graph_features.py
import pandas as pd
from graph_builder import TransactionGraph

def compute_graph_features(df, time_window=24):
    print("Computing graph features...")
    df_sorted = df.sort_values('step').reset_index(drop=True)
    graph = TransactionGraph()
    graph_features = []
    total = len(df_sorted)
    for idx, row in df_sorted.iterrows():
        if idx % 100000 == 0:
            print(f"Processed {idx}/{total} transactions...")
        sender = row['nameOrig']
        receiver = row['nameDest']
        features = {}
        features['sender_degree'] = graph.get_out_degree(sender)
        features['receiver_degree'] = graph.get_in_degree(receiver)
        features['sender_unique_receivers'] = graph.get_unique_receivers(sender)
        features['sender_avg_amount'] = graph.get_avg_amount_sent(sender)
        has_sent_before = False
        if graph.graph.has_node(sender):
            for _, out_receiver in graph.graph.out_edges(sender, data=False):
                if out_receiver == receiver:
                    has_sent_before = True
                    break
        features['is_new_receiver'] = 0 if has_sent_before else 1
        if features['sender_avg_amount'] > 0:
            features['amount_ratio_to_avg'] = row['amount'] / features['sender_avg_amount']
        else:
            features['amount_ratio_to_avg'] = 1.0
        graph_features.append(features)
        if not graph.graph.has_node(sender):
            graph.graph.add_node(sender)
        if not graph.graph.has_node(receiver):
            graph.graph.add_node(receiver)
        graph.graph.add_edge(sender, receiver, amount=row['amount'])
    features_df = pd.DataFrame(graph_features)
    result_df = df_sorted.copy()
    for col in features_df.columns:
        result_df[col] = features_df[col].values
    print(f"Added {len(features_df.columns)} graph features")
    return result_df