import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_model(preds, true_labels, City):
    if true_labels is None or true_labels.isnull().all():
        print(f"No true labels available for city: {City}")
        return {"city": City, "precision": None, "recall": None, "f1": None}

    precision = precision_score(true_labels, preds)
    recall = recall_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)

    print(f"{City} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    return {"city": City, "precision": precision, "recall": recall, "f1": f1}

def save_results(df, metrics):
    df.to_csv("results/anomaly_detection_results.csv", index=False)
    pd.DataFrame(metrics).to_csv("results/metrics.csv", index=False)
    print("✅ Results and metrics saved.")
