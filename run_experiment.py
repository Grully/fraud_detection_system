#!/usr/bin/env python3
import pandas as pd
import numpy as np
import warnings
import joblib
import os
import time

warnings.filterwarnings('ignore')

from data_loader import load_data, get_data_info
from preprocess import prepare_features, get_feature_columns
from graph_features import compute_graph_features
from model_baseline import train_baseline_model
from model_graph import train_graph_model
from evaluate import compare_models, save_comparison_report, plot_roc_curve, plot_precision_recall_curve

def main():
    print("="*60)
    print("FRAUD DETECTION SYSTEM")
    print("Model training and result saving")
    print("="*60)

    start_time = time.time()

    print("\nSTEP 1: Loading data")
    df = load_data()
    get_data_info(df)

    print("\nSTEP 2: Data preprocessing")
    df_processed = prepare_features(df)

    required_cols = ['nameOrig', 'nameDest', 'isFraud', 'step', 'amount']
    for col in required_cols:
        if col in df_processed.columns:
            df_processed = df_processed.dropna(subset=[col])

    print(f"   After cleaning: {len(df_processed)} rows")

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    print("\nSTEP 3: Training baseline model")
    base_features = get_feature_columns(df_processed)

    baseline_model, baseline_metrics, baseline_preds = train_baseline_model(
        df_processed, base_features
    )

    X_test_base, y_test, y_pred_base, y_proba_base = baseline_preds

    print(f"\n   Test set saved: {len(y_test)} rows")
    print(f"   Fraud rate in test: {y_test.mean()*100:.4f}%")

    print("\nSTEP 4: Computing graph features")
    df_with_graph = compute_graph_features(df_processed)

    graph_cache_path = os.path.join(data_dir, "df_with_graph.pkl")
    joblib.dump(df_with_graph, graph_cache_path)
    print(f"   Graph features cached: {graph_cache_path}")

    print("\nSTEP 5: Training graph model")

    graph_features_list = [
        'sender_degree', 'receiver_degree', 'sender_unique_receivers',
        'sender_avg_amount', 'is_new_receiver', 'amount_ratio_to_avg'
    ]

    available_graph_features = [f for f in graph_features_list if f in df_with_graph.columns]
    if len(available_graph_features) != len(graph_features_list):
        print(f"   Missing features: {set(graph_features_list) - set(available_graph_features)}")

    graph_model, graph_metrics, graph_preds = train_graph_model(
        df_with_graph, base_features, available_graph_features
    )

    X_test_graph, y_test_graph, y_pred_graph, y_proba_graph = graph_preds

    if len(y_test) == len(y_test_graph) and np.array_equal(y_test, y_test_graph):
        print(f"\n   Test sets match")
    else:
        print(f"\n   WARNING: Test sets differ!")

    print("\nSTEP 6: Saving test data")

    test_data = {
        'X_test_base': X_test_base,
        'X_test_graph': X_test_graph,
        'y_test': y_test,
        'base_features': base_features,
        'graph_features': available_graph_features,
        'random_state': 42,
        'test_size': 0.3,
        'n_samples': len(df_processed),
        'n_fraud': int(y_test.sum()),
        'fraud_rate': y_test.mean()
    }

    test_data_path = os.path.join(data_dir, "test_data.pkl")
    joblib.dump(test_data, test_data_path)
    print(f"   Test data saved: {test_data_path}")
    print(f"   Test set size: {len(y_test)}")
    print(f"   Fraud rate in test: {y_test.mean()*100:.4f}%")
    print(f"   Fraud count in test: {int(y_test.sum())}")

    print("\nSTEP 7: Model comparison")
    improvements = compare_models(baseline_metrics, graph_metrics)

    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    save_comparison_report(baseline_metrics, graph_metrics, improvements,
                           os.path.join(reports_dir, "comparison_report.txt"))

    plot_roc_curve(y_test, y_proba_base,
                   title="ROC Curve - Baseline Model",
                   save_path=os.path.join(reports_dir, "roc_baseline.png"))
    plot_roc_curve(y_test, y_proba_graph,
                   title="ROC Curve - Graph Model",
                   save_path=os.path.join(reports_dir, "roc_graph.png"))

    plot_precision_recall_curve(y_test, y_proba_base,
                                title="Precision-Recall Curve - Baseline Model",
                                save_path=os.path.join(reports_dir, "pr_baseline.png"))
    plot_precision_recall_curve(y_test, y_proba_graph,
                                title="Precision-Recall Curve - Graph Model",
                                save_path=os.path.join(reports_dir, "pr_graph.png"))

    elapsed_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"TRAINING COMPLETED in {elapsed_time:.2f} seconds")
    print("="*60)

    print("\nRESULTS:")
    print(f"   Baseline: Precision={baseline_metrics['precision']:.4f}, Recall={baseline_metrics['recall']:.4f}, F1={baseline_metrics['f1']:.4f}, AUC={baseline_metrics['roc_auc']:.4f}")
    print(f"   Graph: Precision={graph_metrics['precision']:.4f}, Recall={graph_metrics['recall']:.4f}, F1={graph_metrics['f1']:.4f}, AUC={graph_metrics['roc_auc']:.4f}")
    print(f"   F1 improvement: +{improvements['f1']:.2f}%")

    metrics_summary = {
        'baseline': baseline_metrics,
        'graph': graph_metrics,
        'improvements': improvements,
        'test_size': len(y_test),
        'test_fraud_rate': float(y_test.mean()),
        'runtime': elapsed_time
    }
    joblib.dump(metrics_summary, os.path.join(reports_dir, "metrics_summary.pkl"))

if __name__ == "__main__":
    main()