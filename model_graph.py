# model_graph.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

def train_graph_model(df, base_features, graph_features, target_col='isFraud', test_size=0.3, random_state=42):
    print("\n" + "="*50)
    print("GRAPH MODEL: Random Forest + graph features")
    print("="*50)
    missing = [f for f in graph_features if f not in df.columns]
    if missing:
        print(f"Warning: Missing features: {missing}")
        graph_features = [f for f in graph_features if f in df.columns]
    all_features = base_features + graph_features
    X = df[all_features].values
    y = df[target_col].values
    fraud_rate = y.mean() * 100
    print(f"Fraud rate: {fraud_rate:.4f}%")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    print("\nRESULTS:")
    for metric, value in metrics.items():
        print(f"   {metric.upper()}: {value:.4f} ({value*100:.2f}%)")
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "graph_model.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel saved: {model_path}")
    return model, metrics, (X_test, y_test, y_pred, y_proba)