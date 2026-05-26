#!/usr/bin/env python3
"""
Главный скрипт для запуска полного эксперимента.
Сохраняет модели и тестовую выборку для последующей оценки.
"""
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
    print("СИСТЕМА ОБНАРУЖЕНИЯ МОШЕННИЧЕСКИХ ТРАНЗАКЦИЙ")
    print("Обучение моделей и сохранение результатов")
    print("="*60)
    start_time = time.time()
    # ========== 1. ЗАГРУЗКА ДАННЫХ ==========
    print("\n ЭТАП 1: Загрузка данных")
    df = load_data()
    get_data_info(df)
    # Для ускорения можно использовать подвыборку (раскомментировать при необходимости)
    # df = df.head(200000)
    # print(f"   Используется подвыборка: {len(df)} строк")
    # ========== 2. ПРЕДОБРАБОТКА ==========
    print("\n ЭТАП 2: Предобработка данных")
    df_processed = prepare_features(df)
    required_cols = ['nameOrig', 'nameDest', 'isFraud', 'step', 'amount']
    for col in required_cols:
        if col in df_processed.columns:
            df_processed = df_processed.dropna(subset=[col])
    print(f"   После очистки: {len(df_processed)} строк")
    # Создаём папки
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    # ========== 3. БАЗОВАЯ МОДЕЛЬ ==========
    print("\n ЭТАП 3: Обучение базовой модели")
    base_features = get_feature_columns(df_processed)
    baseline_model, baseline_metrics, baseline_preds = train_baseline_model(
        df_processed, base_features
    )
    # Извлекаем тестовые данные
    x_test_base, y_test, y_pred_base, y_proba_base = baseline_preds
    print(f"\n   Тестовая выборка сохранена: {len(y_test)} строк")
    print(f"   Доля мошенничества в тесте: {y_test.mean()*100:.4f}%")
    # ========== 4. ВЫЧИСЛЕНИЕ ГРАФОВЫХ ПРИЗНАКОВ ==========
    print("\n ЭТАП 4: Вычисление графовых признаков")
    df_with_graph = compute_graph_features(df_processed)
    # Кэшируем графовые признаки
    graph_cache_path = os.path.join(data_dir, "df_with_graph.pkl")
    joblib.dump(df_with_graph, graph_cache_path)
    print(f"   Графовые признаки кэшированы: {graph_cache_path}")
    # ========== 5. ГРАФОВАЯ МОДЕЛЬ ==========
    print("\n ЭТАП 5: Обучение графовой модели")
    graph_features_list = [
        'sender_degree', 'receiver_degree', 'sender_unique_receivers',
        'sender_avg_amount', 'is_new_receiver', 'amount_ratio_to_avg'
    ]
    # Проверяем наличие всех колонок
    available_graph_features = [f for f in graph_features_list if f in df_with_graph.columns]
    if len(available_graph_features) != len(graph_features_list):
        print(f"   Отсутствуют признаки: {set(graph_features_list) - set(available_graph_features)}")
    graph_model, graph_metrics, graph_preds = train_graph_model(
        df_with_graph, base_features, available_graph_features
    )
    # Извлекаем тестовые данные графовой модели
    x_test_graph, y_test_graph, y_pred_graph, y_proba_graph = graph_preds
    # Проверяем, что y_test совпадает (должно быть True)
    if len(y_test) == len(y_test_graph) and np.array_equal(y_test, y_test_graph):
        print(f"\n   Тестовые выборки совпадают")
    else:
        print(f"\n   ВНИМАНИЕ: Тестовые выборки различаются! Это проблема.")
    # ========== 6. СОХРАНЕНИЕ ТЕСТОВОЙ ВЫБОРКИ ==========
    print("\n ЭТАП 6: Сохранение тестовой выборки")
    test_data = {
        'x_test_base': x_test_base,
        'x_test_graph': x_test_graph,
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
    print(f"   Тестовая выборка сохранена: {test_data_path}")
    print(f"   Размер тестовой выборки: {len(y_test)}")
    print(f"   Доля мошенничества в тесте: {y_test.mean()*100:.4f}%")
    print(f"   Количество мошеннических транзакций в тесте: {int(y_test.sum())}")
    # ========== 7. СРАВНЕНИЕ И ВИЗУАЛИЗАЦИЯ ==========
    print("\n ЭТАП 7: Сравнение результатов")
    improvements = compare_models(baseline_metrics, graph_metrics)
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    save_comparison_report(baseline_metrics, graph_metrics, improvements,
                           os.path.join(reports_dir, "comparison_report.txt"))
    # Строим графики
    plot_roc_curve(y_test, y_proba_base,
                   title="ROC-кривая - Базовая модель",
                   save_path=os.path.join(reports_dir, "roc_baseline.png"))
    plot_roc_curve(y_test, y_proba_graph,
                   title="ROC-кривая - Графовая модель",
                   save_path=os.path.join(reports_dir, "roc_graph.png"))
    plot_precision_recall_curve(y_test, y_proba_base,
                                title="Precision-Recall кривая - Базовая модель",
                                save_path=os.path.join(reports_dir, "pr_baseline.png"))
    plot_precision_recall_curve(y_test, y_proba_graph,
                                title="Precision-Recall кривая - Графовая модель",
                                save_path=os.path.join(reports_dir, "pr_graph.png"))
    # ========== 8. ИТОГИ ==========
    elapsed_time = time.time() - start_time
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ МОДЕЛЕЙ ЗАВЕРШЕНО")
    print(f"Время выполнения: {elapsed_time:.2f} секунд")
    print("="*60)
    print("\n РЕЗУЛЬТАТЫ:")
    print(f"   Базовая модель: Precision={baseline_metrics['precision']:.4f}, Recall={baseline_metrics['recall']:.4f}, F1={baseline_metrics['f1']:.4f}, AUC={baseline_metrics['roc_auc']:.4f}")
    print(f"   Графовая модель: Precision={graph_metrics['precision']:.4f}, Recall={graph_metrics['recall']:.4f}, F1={graph_metrics['f1']:.4f}, AUC={graph_metrics['roc_auc']:.4f}")
    print(f"   Улучшение F1: +{improvements['f1']:.2f}%")
    # Сохраняем метрики в отдельный файл для отчёта
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