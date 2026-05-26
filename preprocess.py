"""
Модуль предобработки данных.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
def encode_categorical(df):
    """
    Кодирует категориальные признаки.
    Для PaySim это колонка 'type' (тип транзакции).
    """
    if 'type' in df.columns:
        le = LabelEncoder()
        df['type_encoded'] = le.fit_transform(df['type'])
        print(f"   Типы транзакций: {dict(zip(le.classes_, range(len(le.classes_))))}")
    return df
def extract_temporal_features(df):
    """
    Извлекает временные признаки из колонки 'step'.
    step - часы от начала симуляции (1 step = 1 час)
    """
    if 'step' in df.columns:
        # Время суток (0-23, по модулю 24)
        df['hour_of_day'] = df['step'] % 24
        # День недели (0-6, 7 дней * 24 часа = 168 шагов в неделю)
        df['day_of_week'] = (df['step'] // 24) % 7
        # Выходной или будний день
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        print(f"   Временные признаки добавлены: hour_of_day, day_of_week, is_weekend")
    return df
def prepare_features(df):
    """
    Подготавливает признаки для обучения.
    ВАЖНО: Балансы (oldbalanceOrg, newbalanceOrig и т.д.) НЕ используются,
    так как для мошеннических транзакций они обнулены (см. описание датасета).
    """
    # Копируем, чтобы не изменять оригинал
    df_processed = df.copy()
    print("\n Предобработка данных...")
    # 1. Кодирование категорий
    df_processed = encode_categorical(df_processed)
    # 2. Временные признаки
    df_processed = extract_temporal_features(df_processed)
    # 3. Базовые числовые признаки
    # Используем только amount (балансы НЕ используем!)
    numeric_cols = ['amount']
    for col in numeric_cols:
        if col in df_processed.columns:
            # Логарифмируем сумму для уменьшения влияния выбросов
            df_processed[f'{col}_log'] = np.log1p(df_processed[col])
    return df_processed
def get_feature_columns(df):
    """
    Возвращает список колонок, используемых как признаки.
    """
    feature_cols = []
    # Категориальные закодированные
    if 'type_encoded' in df.columns:
        feature_cols.append('type_encoded')
    # Временные
    for col in ['hour_of_day', 'day_of_week', 'is_weekend']:
        if col in df.columns:
            feature_cols.append(col)
    # Числовые
    if 'amount_log' in df.columns:
        feature_cols.append('amount_log')
    elif 'amount' in df.columns:
        feature_cols.append('amount')
    return feature_cols
