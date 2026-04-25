from pathlib import Path

import pandas as pd

from src.data_loader import build_dataset
from src.evaluate import show_permutation_importance
from src.model import train_model
from src.visualize import (
    save_feature_importance,
    save_model_comparison,
    save_residual_distribution,
    save_true_vs_predicted_plot,
)

RESULT_DIR = Path("result")
RESULT_DIR.mkdir(exist_ok=True)


def main():
    X_train, X_test, y_train, y_test, X_encoded, y = build_dataset(
        "data/hour.csv",
        use_time_split=True,
    )

    trained_models, search_details, results = train_model(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    results_df = pd.DataFrame(results).sort_values(by="test_r2", ascending=False)

    print("\n===== Model Results =====")
    print(results_df)
    results_df.to_csv(RESULT_DIR / "model_results.csv", index=False)

    tuning_rows = []
    for model_name, info in search_details.items():
        print(f"\n{model_name} best params: {info['best_params']}")
        print(f"{model_name} best CV score: {info['best_cv_score']}")
        tuning_rows.append(
            {
                "model": model_name,
                "best_params": str(info["best_params"]),
                "best_cv_score": info["best_cv_score"],
            }
        )
    pd.DataFrame(tuning_rows).to_csv(RESULT_DIR / "tuning_summary.csv", index=False)

    rf_model = trained_models["Random Forest"]
    rf_importance = pd.Series(
        rf_model.feature_importances_,
        index=X_train.columns,
    ).sort_values(ascending=False)

    print("\n==== Random Forest Feature Importance ====")
    print(rf_importance)
    rf_importance.to_csv(RESULT_DIR / "rf_feature_importance.csv", header=["importance"])

    xgb_importance = show_permutation_importance(
        trained_models["XGBoost"],
        X_test,
        y_test,
        X_train.columns,
        "XGBoost",
    )
    xgb_importance.to_csv(RESULT_DIR / "xgb_permutation_importance.csv", header=["importance"])

    save_model_comparison(results_df)
    save_feature_importance(
        rf_importance,
        "Random Forest Feature Importance",
        "rf_feature_importance.png",
    )
    save_feature_importance(
        xgb_importance,
        "XGBoost Permutation Importance",
        "xgb_permutation_importance.png",
    )

    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    y_pred = best_model.predict(X_test)

    save_true_vs_predicted_plot(y_test, y_pred)

    residuals = y_test - y_pred
    save_residual_distribution(residuals)

    print(f"\nBest model: {best_model_name}")
    print(f"All result files have been saved to: {RESULT_DIR.resolve()}")


if __name__ == "__main__":
    main()
