# evaluate.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import numpy as np
import os

def plot_roc_curve(y_test, y_proba, title="ROC Curve", save_path=None):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = np.trapz(tpr, fpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random (AUC = 0.5)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return auc

def plot_precision_recall_curve(y_test, y_proba, title="PR Curve", save_path=None):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'g-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix", save_path=None):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legit', 'Fraud'],
                yticklabels=['Legit', 'Fraud'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def compare_models(baseline_metrics, graph_metrics):
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    metrics_names = ['precision', 'recall', 'f1', 'roc_auc']
    print(f"\n{'Metric':<15} {'Baseline':<15} {'Graph':<15} {'Improvement':<15}")
    print("-" * 60)
    improvements = {}
    for metric in metrics_names:
        baseline = baseline_metrics[metric]
        graph = graph_metrics[metric]
        improvement = ((graph - baseline) / baseline) * 100 if baseline > 0 else 0
        improvements[metric] = improvement
        arrow = "+" if improvement > 0 else "-" if improvement < 0 else "="
        print(f"{metric.upper():<15} {baseline:<15.4f} {graph:<15.4f} {arrow} {improvement:+.2f}%")
    return improvements

def save_comparison_report(baseline_metrics, graph_metrics, improvements, save_path=None):
    report = []
    report.append("="*60)
    report.append("MODEL COMPARISON REPORT")
    report.append("="*60)
    report.append("\nBaseline model:")
    for metric, value in baseline_metrics.items():
        report.append(f"  {metric.upper()}: {value:.4f}")
    report.append("\nGraph model:")
    for metric, value in graph_metrics.items():
        report.append(f"  {metric.upper()}: {value:.4f}")
    report.append("\nImprovement:")
    for metric, imp in improvements.items():
        report.append(f"  {metric.upper()}: {imp:+.2f}%")
    report_text = "\n".join(report)
    print(report_text)
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
    return report_text