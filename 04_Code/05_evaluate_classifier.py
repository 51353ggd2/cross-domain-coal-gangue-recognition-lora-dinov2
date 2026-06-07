"""
Evaluate binary coal-gangue classification results.

This script expects a CSV file containing:
- image_id
- label_id
- prediction_id
- probability_coal
- probability_gangue
"""

from pathlib import Path
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate(prediction_csv, output_dir):
    prediction_csv = Path(prediction_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(prediction_csv)
    y_true = df["label_id"].astype(int)
    y_pred = df["prediction_id"].astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="binary", zero_division=0),
    }

    pd.DataFrame([metrics]).to_csv(output_dir / "classification_metrics.csv", index=False)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    matrix_df = pd.DataFrame(matrix, index=["true_coal", "true_gangue"], columns=["pred_coal", "pred_gangue"])
    matrix_df.to_csv(output_dir / "confusion_matrix.csv")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_csv", default="../05_Result_Data/predictions_template.csv")
    parser.add_argument("--output_dir", default="../05_Result_Data/Generated_Outputs")
    args = parser.parse_args()

    result = evaluate(args.prediction_csv, args.output_dir)
    print(result)
