import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def data_load(file_path: str):
    df = pd.read_csv(file_path)
    return df


def data_check(data:pd.DataFrame):
    print("==== Basic Data Information ====")
    print(data.info())
    print("==== First Five Rows ====")
    print(data.head())
    print("==== Missing Values ====")
    print(data.isnull().sum())
    print("==== Duplicated Rows ====")
    print(data.duplicated().sum())
    print("==== Descriptive Statistics ====")
    print(data.describe())


def data_processing(data:pd.DataFrame)->pd.DataFrame:
    data = data.copy()
    if "instant" in data.columns:
        data = data.drop("instant", axis=1)

    data["dteday"] = pd.to_datetime(data["dteday"])

    data["year"] = data["dteday"].dt.year
    data["month"] = data["dteday"].dt.month
    data["day"] = data["dteday"].dt.day
    data["dayofweek"] = data["dteday"].dt.dayofweek

    data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(int)

    if "hr" in data.columns:
        data["is_peak_hour"] = data["hr"].isin([7, 8, 9, 17, 18, 19]).astype(int)
        data["hr_sin"] = np.sin(2 * np.pi * data["hr"] / 24)
        data["hr_cos"] = np.cos(2 * np.pi * data["hr"] / 24)

    # 7. 构造月份周期特征
    data["month_sin"] = np.sin(2 * np.pi * data["mnth"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["mnth"] / 12)
    data = data.drop("dteday", axis=1)

    return data


def split_features(data:pd.DataFrame):
    leakage_cols = ["casual", "registered"]
    target_col = "cnt"

    drop_cols = leakage_cols + [target_col]
    X = data.drop(drop_cols, axis=1)
    y = data[target_col]

    return X, y


def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    对类别特征进行 One-Hot 编码
    """

    X = X.copy()

    categorical_cols = [
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "year",
        "month",
        "dayofweek",
        "is_weekend",
        "is_peak_hour"
    ]

    categorical_cols = [col for col in categorical_cols if col in X.columns]

    X_encoded = pd.get_dummies(
        X,
        columns=categorical_cols,
        drop_first=True
    )

    return X_encoded


def time_based_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):


    split_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def random_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )


def build_dataset(file_path: str, use_time_split: bool = True):

    data = data_load(file_path)

    print("Raw data shape:", data.shape)

    data_check(data)

    data_processed = data_processing(data)

    print("\nProcessed data shape:", data_processed.shape)

    X, y = split_features(data_processed)

    print("Feature X shape:", X.shape)
    print("Target y shape:", y.shape)

    X_encoded = encode_features(X)

    print("Encoded X shape:", X_encoded.shape)

    if use_time_split:
        X_train, X_test, y_train, y_test = time_based_split(X_encoded, y)
    else:
        X_train, X_test, y_train, y_test = random_split(X_encoded, y)

    print("\nX_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    return X_train, X_test, y_train, y_test, X_encoded, y

