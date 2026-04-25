import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance


def evaluate(model, X_train, X_test, y_train, y_test, name: str):
    """Return a unified regression evaluation dictionary."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = test_mse ** 0.5
    gap = train_r2 - test_r2

    return {
        "model": name,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "test_mae": test_mae,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "gap": gap,
    }


def show_permutation_importance(model, X_test, y_test, feature_names, model_name, n_repeats: int = 5):
    """Calculate and return permutation importance as a sorted Series."""
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=42,
        scoring="r2",
    )
    importance = pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False)
    print(f"\n==== {model_name} Permutation Importance ====")
    print(importance)
    return importance