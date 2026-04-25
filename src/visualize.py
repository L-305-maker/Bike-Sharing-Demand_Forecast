from pathlib import Path
import matplotlib.pyplot as plt

RESULT_DIR = Path("result")
RESULT_DIR.mkdir(exist_ok=True)

def save_model_comparison(results_df):
    plt.figure(figsize=(10, 6))
    plt.barh(results_df["model"], results_df["test_r2"])
    plt.xlabel("Test R2")
    plt.title("Model Performance Comparison")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()



def save_feature_importance(importance_series, title: str, filename: str, xlabel: str = "Importance"):
    plt.figure(figsize=(8, 6))
    importance_series.sort_values().plot(kind="barh")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()



def save_true_vs_predicted_plot(y_true, y_pred):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.xlabel("True Demand")
    plt.ylabel("Predicted Demand")
    plt.title("True vs Predicted")
    min_value = min(y_true.min(), y_pred.min())
    max_value = max(y_true.max(), y_pred.max())
    plt.plot([min_value, max_value], [min_value, max_value])
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "true_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close()



def save_residual_distribution(residuals):
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=50)
    plt.title("Residual Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "residual_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
