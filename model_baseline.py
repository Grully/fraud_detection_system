"""
Модуль обучения базовой модели (Random Forest).
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
def train_baseline_model(df, feature_cols, target_col='isFraud', test_size=0.3, random_state=42):
    """
    Обучает базовую модель Random Forest.
    Parameters:
    -----------
    df : pd.DataFrame
        Данные с признаками и целевой переменной
    feature_cols : list
        Список колонок-признаков
    target_col : str
        Название целевой переменной
    test_size : float
        Доля тестовой выборки
    random_state : int
        Seed для воспроизводимости
    Returns:
    --------
    tuple: (модель, метрики, (x_test, y_test, y_pred, y_proba))
    """
    print("\n" + "="*50)
    print("БАЗОВАЯ МОДЕЛЬ: Random Forest")
    print("="*50)
    # Подготовка данных
    X = df[feature_cols].values
    y = df[target_col].values
    print(f"Размер выборки: {len(X)}")
    print(f"Количество признаков: {len(feature_cols)}")
    print(f"Признаки: {feature_cols}")
    # Дисбаланс классов
    fraud_rate = y.mean() * 100
    print(f"Доля мошенничества: {fraud_rate:.4f}%")
    # Разделение на train/test
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Обучающая выборка: {len(x_train)} (фрод: {y_train.mean()*100:.4f}%)")
    print(f"Тестовая выборка: {len(x_test)} (фрод: {y_test.mean()*100:.4f}%)")
    # Обучение модели
    print("\n Обучение модели...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(x_train, y_train)
    # Предсказания
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    # Метрики
    metrics = {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    print("\n РЕЗУЛЬТАТЫ:")
    for metric, value in metrics.items():
        print(f"   {metric.upper()}: {value:.4f} ({value*100:.2f}%)")
    # Сохранение модели
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "baseline_model.pkl")
    joblib.dump(model, model_path)
    print(f"\n Модель сохранена: {model_path}")
    # Возвращаем x_test для сохранения
    return model, metrics, (x_test, y_test, y_pred, y_proba)