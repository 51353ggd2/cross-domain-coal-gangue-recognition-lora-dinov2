"""
Generate publication-ready figures from source result tables.

This script uses result templates from the supplementary material.
Replace the template rows with real experimental data before plotting.
"""

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_baseline_results(csv_file, output_path):
    df = pd.read_csv(csv_file)
    if df.empty:
        return

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(df["method"], df["f1_score"])
    axis.set_ylabel("F1-score")
    axis.set_xlabel("Method")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_training_curve(csv_file, output_path):
    df = pd.read_csv(csv_file)
    if df.empty:
        return

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(df["epoch"], df["training_loss"], marker="o")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Training loss")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", default="../05_Result_Data")
    parser.add_argument("--figure_dir", default="../06_Figures_and_Source_Data/Generated_Figures")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    plot_baseline_results(result_dir / "baseline_results_template.csv", figure_dir / "baseline_f1_scores.png")
    plot_training_curve(result_dir / "training_curve_template.csv", figure_dir / "training_curve.png")
