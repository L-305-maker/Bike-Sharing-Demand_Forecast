from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from .evaluate import evaluate


def train_model(X_train, X_test, y_train, y_test):
    """Train multiple regression models and return trained models, tuning details, and evaluation results."""
    results = []
    trained_models = {}
    search_details = {}

    # Time-aware cross-validation for this time-dependent demand prediction task.
    tscv_5 = TimeSeriesSplit(n_splits=5)
    tscv_3 = TimeSeriesSplit(n_splits=3)

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    trained_models["Linear Regression"] = lr
    results.append(evaluate(lr, X_train, X_test, y_train, y_test, "Linear Regression"))

    # 2. Decision Tree Regressor
    tree = DecisionTreeRegressor(
        max_depth=9,
        min_samples_leaf=15,
        min_samples_split=15,
        random_state=42,
    )
    tree.fit(X_train, y_train)
    trained_models["Decision Tree"] = tree
    results.append(evaluate(tree, X_train, X_test, y_train, y_test, "Decision Tree"))

    # 3. Ridge Regression
    ridge_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ]
    )
    param_ridge = {"ridge__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    ridge_grid = GridSearchCV(
        ridge_pipe,
        param_grid=param_ridge,
        cv=tscv_5,
        scoring="r2",
        n_jobs=1,
    )
    ridge_grid.fit(X_train, y_train)
    ridge_best = ridge_grid.best_estimator_
    trained_models["Ridge"] = ridge_best
    search_details["Ridge"] = {
        "best_params": ridge_grid.best_params_,
        "best_cv_score": float(ridge_grid.best_score_),
    }
    results.append(evaluate(ridge_best, X_train, X_test, y_train, y_test, "Ridge"))

    # 4. Lasso Regression
    lasso_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lasso", Lasso(max_iter=50000)),
        ]
    )
    param_lasso = {"lasso__alpha": [0.001, 0.01, 0.1, 1]}
    lasso_grid = GridSearchCV(
        lasso_pipe,
        param_grid=param_lasso,
        cv=tscv_5,
        scoring="r2",
        n_jobs=1,
    )
    lasso_grid.fit(X_train, y_train)
    lasso_best = lasso_grid.best_estimator_
    trained_models["Lasso"] = lasso_best
    search_details["Lasso"] = {
        "best_params": lasso_grid.best_params_,
        "best_cv_score": float(lasso_grid.best_score_),
    }
    results.append(evaluate(lasso_best, X_train, X_test, y_train, y_test, "Lasso"))

    # 5. Polynomial + Ridge
    poly_ridge_pipe = Pipeline(
        [
            ("poly", PolynomialFeatures(include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ]
    )
    param_poly_ridge = {
        "poly__degree": [1, 2],
        "ridge__alpha": [0.01, 0.1, 1, 10, 100, 1000],
    }
    poly_ridge_grid = GridSearchCV(
        poly_ridge_pipe,
        param_grid=param_poly_ridge,
        cv=tscv_5,
        scoring="r2",
        n_jobs=1,
    )
    poly_ridge_grid.fit(X_train, y_train)
    poly_ridge_best = poly_ridge_grid.best_estimator_
    trained_models["Poly + Ridge"] = poly_ridge_best
    search_details["Poly + Ridge"] = {
        "best_params": poly_ridge_grid.best_params_,
        "best_cv_score": float(poly_ridge_grid.best_score_),
    }
    results.append(evaluate(poly_ridge_best, X_train, X_test, y_train, y_test, "Poly + Ridge"))

    # 6. Polynomial + Lasso
    poly_lasso_pipe = Pipeline(
        [
            ("poly", PolynomialFeatures(include_bias=False)),
            ("scaler", StandardScaler()),
            ("lasso", Lasso(max_iter=50000)),
        ]
    )
    param_poly_lasso = {
        "poly__degree": [1, 2],
        "lasso__alpha": [0.01, 0.1, 1],
    }
    poly_lasso_grid = GridSearchCV(
        poly_lasso_pipe,
        param_grid=param_poly_lasso,
        cv=tscv_5,
        scoring="r2",
        n_jobs=1,
    )
    poly_lasso_grid.fit(X_train, y_train)
    poly_lasso_best = poly_lasso_grid.best_estimator_
    trained_models["Poly + Lasso"] = poly_lasso_best
    search_details["Poly + Lasso"] = {
        "best_params": poly_lasso_grid.best_params_,
        "best_cv_score": float(poly_lasso_grid.best_score_),
    }
    results.append(evaluate(poly_lasso_best, X_train, X_test, y_train, y_test, "Poly + Lasso"))

    # 7. Random Forest Regressor
    rf = RandomForestRegressor(random_state=42)
    param_rf = {
        "n_estimators": [100, 150, 200],
        "max_depth": [5, 9, 12],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [2, 5, 10],
    }
    rf_grid = GridSearchCV(
        rf,
        param_grid=param_rf,
        cv=tscv_3,
        scoring="r2",
        n_jobs=1,
    )
    rf_grid.fit(X_train, y_train)
    rf_best = rf_grid.best_estimator_
    trained_models["Random Forest"] = rf_best
    search_details["Random Forest"] = {
        "best_params": rf_grid.best_params_,
        "best_cv_score": float(rf_grid.best_score_),
    }
    results.append(evaluate(rf_best, X_train, X_test, y_train, y_test, "Random Forest"))

    # 8. XGBoost Regressor
    xgb = XGBRegressor(
        random_state=42,
        objective="reg:squarederror",
        n_jobs=1,
    )
    param_xgb = {
        "n_estimators": [100, 150, 200, 300],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.03, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
    }
    xgb_random = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_xgb,
        n_iter=20,
        cv=tscv_3,
        scoring="r2",
        n_jobs=1,
        random_state=42,
    )
    xgb_random.fit(X_train, y_train)
    xgb_best = xgb_random.best_estimator_
    trained_models["XGBoost"] = xgb_best
    search_details["XGBoost"] = {
        "best_params": xgb_random.best_params_,
        "best_cv_score": float(xgb_random.best_score_),
    }
    results.append(evaluate(xgb_best, X_train, X_test, y_train, y_test, "XGBoost"))

    return trained_models, search_details, results
