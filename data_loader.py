"""
Модуль загрузки данных PaySim.
"""
import pandas as pd
import os
def load_data(file_path=None):
    """
    Загружает датасет PaySim.
    Parameters:
    -----------
    file_path : str, optional
        Путь к CSV файлу. Если None, ищет в папке data/
    Returns:
    --------
    pd.DataFrame : Загруженные данные
    """
    if file_path is None:
        # Ищем файл в папке data
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        possible_names = [
            "PS_20174392719_1491204439457_log.csv",
            "paysim.csv",
            "PS_20174392719_1491204439457_log.csv"
        ]
        for name in possible_names:
            candidate = os.path.join(data_dir, name)
            if os.path.exists(candidate):
                file_path = candidate
                break
    if file_path is None or not os.path.exists(file_path):
        raise FileNotFoundError(
            "Датасет не найден! Поместите файл PS_20174392719_1491204439457_log.csv "
            "в папку data/ или укажите путь явно."
        )
    print(f" Загрузка данных из: {file_path}")
    df = pd.read_csv(file_path)
    print(f"   Загружено {len(df):,} строк, {len(df.columns)} колонок")
    return df
def get_data_info(df):
    """
    Выводит основную информацию о датасете.
    """
    print("\n" + "="*50)
    print("ИНФОРМАЦИЯ О ДАННЫХ")
    print("="*50)
    print(f"Размер: {df.shape}")
    print(f"Типы колонок:\n{df.dtypes}")
    print(f"\nПропуски:\n{df.isnull().sum()}")
    if 'isFraud' in df.columns:
        fraud_rate = df['isFraud'].mean() * 100
        print(f"\nДоля мошеннических транзакций: {fraud_rate:.4f}%")
    return df
