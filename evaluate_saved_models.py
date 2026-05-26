#!/usr/bin/env python3
"""
Скрипт для оценки уже сохранённых моделей (без переобучения).
Использует ТУ ЖЕ тестовую выборку, что и при обучении.
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import joblib
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, confusion_matrix
)
def plot_roc_curve(y_test, y_proba, title="ROC Curve", save_path=None):
    """Строит ROC-кривую."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = np.trapz(tpr, fpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Случайная модель (AUC = 0.5)')
    plt.xlabel('Доля ложных срабатываний (FPR)')
    plt.ylabel('Доля верно обнаруженных (TPR)')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   График сохранён: {save_path}")
    plt.show()
    return auc
def plot_precision_recall_curve(y_test, y_proba, title="PR Curve", save_path=None):
    """Строит Precision-Recall кривую."""
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'g-', linewidth=2)
    plt.xlabel('Полнота (Recall)')
    plt.ylabel('Точность (Precision)')
    plt.title(title)
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   График сохранён: {save_path}")
    plt.show()
def plot_confusion_matrix(y_test, y_pred, title="Матрица ошибок", save_path=None):
    """Строит матрицу ошибок."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Легитимные', 'Мошеннические'],
                yticklabels=['Легитимные', 'Мошеннические'])
    plt.title(title)
    plt.ylabel('Истинная метка')
    plt.xlabel('Предсказанная метка')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   График сохранён: {save_path}")
    plt.show()
def evaluate_saved_models():
    """Загружает модели и тестовую выборку, выполняет оценку."""
    print("="*60)
    print("ОЦЕНКА СОХРАНЁННЫХ МОДЕЛЕЙ")
    print("="*60)
    start_time = time.time()
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    baseline_model_path = os.path.join(models_dir, "baseline_model.pkl")
    graph_model_path = os.path.join(models_dir, "graph_model.pkl")
    test_data_path = os.path.join(data_dir, "test_data.pkl")
    # Проверка наличия файлов
    missing = []
    if not os.path.exists(baseline_model_path):
        missing.append("baseline_model.pkl")
    if not os.path.exists(graph_model_path):
        missing.append("graph_model.pkl")
    if not os.path.exists(test_data_path):
        missing.append("test_data.pkl")
    if missing:
        print(f"\n Отсутствуют файлы: {missing}")
        print("Сначала запустите run_experiment.py для обучения и сохранения моделей.")
        return
    # Загрузка
    print("\n Загрузка данных...")
    baseline_model = joblib.load(baseline_model_path)
    graph_model = joblib.load(graph_model_path)
    test_data = joblib.load(test_data_path)
    x_test_base = test_data['x_test_base']
    x_test_graph = test_data['x_test_graph']
    y_test = test_data['y_test']
    print(f"   Тестовая выборка: {len(y_test)} строк")
    print(f"   Доля мошенничества: {y_test.mean()*100:.4f}%")
    print(f"   Количество мошеннических транзакций в тесте: {int(y_test.sum())}")
    # Диагностика: проверяем размерности
    print(f"\n Размерности:")
    print(f"   x_test_base: {x_test_base.shape}")
    print(f"   x_test_graph: {x_test_graph.shape}")
    print(f"   y_test: {y_test.shape}")
    # Предсказания
    print("\n Вычисление предсказаний...")
    y_proba_base = baseline_model.predict_proba(x_test_base)[:, 1]
    y_pred_base = (y_proba_base >= 0.5).astype(int)
    y_proba_graph = graph_model.predict_proba(x_test_graph)[:, 1]
    y_pred_graph = (y_proba_graph >= 0.5).astype(int)
    # Диагностика вероятностей
    print(f"\n Статистика вероятностей:")
    print(f"   Базовая модель: min={y_proba_base.min():.4f}, max={y_proba_base.max():.4f}, mean={y_proba_base.mean():.4f}")
    print(f"   Графовая модель: min={y_proba_graph.min():.4f}, max={y_proba_graph.max():.4f}, mean={y_proba_graph.mean():.4f}")
    # Метрики
    baseline_metrics = {
        'precision': precision_score(y_test, y_pred_base),
        'recall': recall_score(y_test, y_pred_base),
        'f1': f1_score(y_test, y_pred_base),
        'roc_auc': roc_auc_score(y_test, y_proba_base)
    }
    graph_metrics = {
        'precision': precision_score(y_test, y_pred_graph),
        'recall': recall_score(y_test, y_pred_graph),
        'f1': f1_score(y_test, y_pred_graph),
        'roc_auc': roc_auc_score(y_test, y_proba_graph)
    }
    # Вывод результатов
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ")
    print("="*50)
    print("\n Базовая модель (Random Forest):")
    print(f"   Точность (Precision): {baseline_metrics['precision']:.4f} ({baseline_metrics['precision']*100:.2f}%)")
    print(f"   Полнота (Recall):     {baseline_metrics['recall']:.4f} ({baseline_metrics['recall']*100:.2f}%)")
    print(f"   F1-мера:              {baseline_metrics['f1']:.4f} ({baseline_metrics['f1']*100:.2f}%)")
    print(f"   ROC-AUC:              {baseline_metrics['roc_auc']:.4f} ({baseline_metrics['roc_auc']*100:.2f}%)")
    print("\n Графовая модель (RF + графовые признаки):")
    print(f"   Точность (Precision): {graph_metrics['precision']:.4f} ({graph_metrics['precision']*100:.2f}%)")
    print(f"   Полнота (Recall):     {graph_metrics['recall']:.4f} ({graph_metrics['recall']*100:.2f}%)")
    print(f"   F1-мера:              {graph_metrics['f1']:.4f} ({graph_metrics['f1']*100:.2f}%)")
    print(f"   ROC-AUC:              {graph_metrics['roc_auc']:.4f} ({graph_metrics['roc_auc']*100:.2f}%)")
    # Сравнение
    print("\n" + "-"*40)
    print("СРАВНЕНИЕ")
    print("-"*40)
    improvements = {}
    for metric in ['precision', 'recall', 'f1', 'roc_auc']:
        base = baseline_metrics[metric]
        graph = graph_metrics[metric]
        imp = ((graph - base) / base) * 100 if base > 0 else 0
        improvements[metric] = imp
        arrow = "+" if imp > 0 else "-" if imp < 0 else "="
        print(f"{metric.upper():<10}: {base:.4f} → {graph:.4f} {arrow} {imp:+.2f}%")
    # Визуализация
    print("\n Построение графиков...")
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    plot_roc_curve(y_test, y_proba_base,
                   title="ROC-кривая - Базовая модель",
                   save_path=os.path.join(reports_dir, "roc_baseline_eval.png"))
    plot_roc_curve(y_test, y_proba_graph,
                   title="ROC-кривая - Графовая модель",
                   save_path=os.path.join(reports_dir, "roc_graph_eval.png"))
    plot_precision_recall_curve(y_test, y_proba_base,
                                title="Precision-Recall - Базовая модель",
                                save_path=os.path.join(reports_dir, "pr_baseline_eval.png"))
    plot_precision_recall_curve(y_test, y_proba_graph,
                                title="Precision-Recall - Графовая модель",
                                save_path=os.path.join(reports_dir, "pr_graph_eval.png"))
    plot_confusion_matrix(y_test, y_pred_base,
                          title="Матрица ошибок - Базовая модель",
                          save_path=os.path.join(reports_dir, "cm_baseline_eval.png"))
    plot_confusion_matrix(y_test, y_pred_graph,
                          title="Матрица ошибок - Графовая модель",
                          save_path=os.path.join(reports_dir, "cm_graph_eval.png"))
    # Сохранение отчёта
    report_path = os.path.join(reports_dir, "evaluation_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ОТЧЁТ ОБ ОЦЕНКЕ СОХРАНЁННЫХ МОДЕЛЕЙ\n")
        f.write("="*40 + "\n\n")
        f.write(f"Тестовая выборка: {len(y_test)} строк\n")
        f.write(f"Доля мошенничества: {y_test.mean()*100:.4f}%\n\n")
        f.write("Базовая модель:\n")
        for k, v in baseline_metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\nГрафовая модель:\n")
        for k, v in graph_metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\nУлучшение:\n")
        for k, v in improvements.items():
            f.write(f"  {k}: {v:+.2f}%\n")
    print(f"   Отчёт сохранён: {report_path}")
    elapsed = time.time() - start_time
    print(f"\n Время выполнения: {elapsed:.2f} сек")
    print("="*60)
    # Итоговый вывод
    if improvements['roc_auc'] > 5:
        print("\n Графовая модель показывает значительное улучшение!")
    elif improvements['roc_auc'] > 0:
        print("\n Графовая модель показывает небольшое улучшение.")
    else:
        print("\n Графовая модель не показала улучшения. Проверьте корректность данных.")
if __name__ == "__main__":
    evaluate_saved_models()