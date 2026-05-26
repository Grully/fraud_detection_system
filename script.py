import os

files = [
    "data_loader.py", "preprocess.py", "graph_builder.py",
    "graph_features.py", "model_baseline.py", "model_graph.py",
    "evaluate.py", "run_experiment.py", "evaluate_saved_models.py"
]

for filename in files:
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        continue

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Удаляем все пустые строки
    cleaned = [line for line in lines if line.strip() != '']

    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(cleaned)
    print(f"Cleaned: {filename}")