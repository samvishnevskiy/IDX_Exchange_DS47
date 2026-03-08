import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import BallTree
from xgboost import XGBRegressor

plt.style.use("ggplot")


def load_data():
    """Load and merge all CRMLSSold CSV files."""
    files = sorted(glob.glob("CRMLSSold20*.csv"))

    df_list = []
    for f in files:
        try:
            df_list.append(pd.read_csv(f, low_memory=False))
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not df_list:
        raise ValueError("No CSV files were loaded successfully.")

    full_df = pd.concat(df_list, ignore_index=True)
    print(f">> Total raw records: {full_df.shape[0]}")
    return full_df


def clean_data(df):
    """Filter scope, convert types, and remove invalid rows."""
    df = df.copy()

    df = df[
        (df["PropertyType"] == "Residential") &
        (df["PropertySubType"] == "SingleFamilyResidence")
    ].copy()

    possible_date_cols = ["CloseDate", "CloseDateTime", "CloseDateUTC"]
    date_col = None
    for c in possible_date_cols:
        if c in df.columns:
            date_col = c
            break

    if date_col is None:
        raise ValueError(
            "No close date column found. Expected one of: CloseDate, CloseDateTime, CloseDateUTC"
        )

    df["CloseDateParsed"] = pd.to_datetime(df[date_col], errors="coerce")

    numeric_cols = [
        "ClosePrice",
        "LivingArea",
        "Latitude",
        "Longitude",
        "BedroomsTotal",
        "BathroomsTotalInteger",
        "YearBuilt",
        "LotSizeSquareFeet",
    ]

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=numeric_cols + ["CloseDateParsed"]).copy()
    df = df[(df["ClosePrice"] >= 10000) & (df["LivingArea"] >= 100)].copy()
    df = df[df["ClosePrice"] <= 5000000].copy()

    df["YearMonth"] = df["CloseDateParsed"].dt.to_period("M").astype(str)

    print(f">> Data cleaned. Rows remaining: {df.shape[0]}")
    return df


def split_forward_holdout(df, train_months, test_month):
    """Split data into train and test sets by month."""
    train_df = df[df["YearMonth"].isin(train_months)].copy()
    test_df = df[df["YearMonth"] == test_month].copy()

    print(f">> Train months: {train_months}")
    print(f">> Test month : {test_month}")
    print(f">> Train size : {train_df.shape[0]}")
    print(f">> Test size  : {test_df.shape[0]}")

    if train_df.empty or test_df.empty:
        raise ValueError("Train or test split is empty. Check the YearMonth values.")

    return train_df, test_df


def add_spatial_features(train_df, test_df, k=5):
    """Create leakage-free spatial lag feature from training sales."""
    train_df = train_df.copy()
    test_df = test_df.copy()

    if len(train_df) < 2:
        raise ValueError("Training data is too small to compute spatial lag features.")

    print(">> Engineering leakage-free SpatialLag_Price...")

    train_df["lat_rad"] = np.radians(train_df["Latitude"])
    train_df["lon_rad"] = np.radians(train_df["Longitude"])
    test_df["lat_rad"] = np.radians(test_df["Latitude"])
    test_df["lon_rad"] = np.radians(test_df["Longitude"])

    tree = BallTree(train_df[["lat_rad", "lon_rad"]].values, metric="haversine")

    k_train = min(k + 1, len(train_df))
    _, ind_train = tree.query(train_df[["lat_rad", "lon_rad"]].values, k=k_train)

    train_prices = train_df["ClosePrice"].values
    train_neighbor_indices = ind_train[:, 1:] if k_train > 1 else ind_train
    train_neighbor_prices = train_prices[train_neighbor_indices]
    train_df["SpatialLag_Price"] = np.mean(train_neighbor_prices, axis=1)

    k_test = min(k, len(train_df))
    _, ind_test = tree.query(test_df[["lat_rad", "lon_rad"]].values, k=k_test)

    test_neighbor_prices = train_prices[ind_test]
    test_df["SpatialLag_Price"] = np.mean(test_neighbor_prices, axis=1)

    return train_df, test_df


def add_zip_median_feature(train_df, test_df):
    """Add ZIP-level median price based on training data only."""
    train_df = train_df.copy()
    test_df = test_df.copy()

    if "PostalCode" not in train_df.columns:
        print(">> PostalCode not found. Skipping ZIP_MedianPrice.")
        return train_df, test_df

    train_df["PostalCode"] = train_df["PostalCode"].astype(str)
    test_df["PostalCode"] = test_df["PostalCode"].astype(str)

    zip_median = train_df.groupby("PostalCode")["ClosePrice"].median()
    global_median = train_df["ClosePrice"].median()

    train_df["ZIP_MedianPrice"] = train_df["PostalCode"].map(zip_median).fillna(global_median)
    test_df["ZIP_MedianPrice"] = test_df["PostalCode"].map(zip_median).fillna(global_median)

    print(">> Added ZIP_MedianPrice feature.")
    return train_df, test_df


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    eps = 1e-9
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps)))


def mdape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    eps = 1e-9
    return np.median(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps)))


def run_model(train_df, test_df):
    """Train XGBoost and evaluate on the holdout month."""
    base_features = [
        "BedroomsTotal",
        "BathroomsTotalInteger",
        "LivingArea",
        "LotSizeSquareFeet",
        "YearBuilt",
        "Latitude",
        "Longitude",
        "SpatialLag_Price",
    ]

    if "ZIP_MedianPrice" in train_df.columns and "ZIP_MedianPrice" in test_df.columns:
        base_features.append("ZIP_MedianPrice")

    features = [f for f in base_features if f in train_df.columns and f in test_df.columns]

    X_train = train_df[features].copy()
    X_test = test_df[features].copy()

    y_train_price = train_df["ClosePrice"].values
    y_test_price = test_df["ClosePrice"].values

    y_train_log = np.log1p(y_train_price)
    y_test_log = np.log1p(y_test_price)

    print(">> Training XGBoost model...")

    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(X_train, y_train_log)

    pred_log = model.predict(X_test)
    pred_price = np.expm1(pred_log)

    metrics = {
        "R2_price": r2_score(y_test_price, pred_price),
        "R2_log_price": r2_score(y_test_log, pred_log),
        "MAE": mean_absolute_error(y_test_price, pred_price),
        "MAPE": mape(y_test_price, pred_price),
        "MdAPE": mdape(y_test_price, pred_price),
    }

    return model, features, metrics, y_test_price, pred_price


def plot_importance(model, features, filename="feature_importance.png"):
    """Save feature importance plot."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(indices)), importances[indices], align="center", color="#348ABD")
    plt.xticks(range(len(indices)), [features[i] for i in indices], rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f">> Saved '{filename}'")


def plot_actual_vs_predicted(y_true, y_pred, filename="actual_vs_predicted.png"):
    """Save predicted vs actual scatter plot."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.25, s=12)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("Actual Close Price")
    plt.ylabel("Predicted Close Price")
    plt.title("Predicted vs Actual Close Price")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f">> Saved '{filename}'")


def evaluate_period(df, train_months, test_month, label, plot_file):
    train_df, test_df = split_forward_holdout(df, train_months, test_month)
    train_df, test_df = add_spatial_features(train_df, test_df, k=5)
    train_df, test_df = add_zip_median_feature(train_df, test_df)

    model, features, metrics, y_test_price, pred_price = run_model(train_df, test_df)

    print("\n" + "=" * 45)
    print(f" {label}")
    print("=" * 45)
    print(f" R² (price)     : {metrics['R2_price']:.4f}")
    print(f" R² (log price) : {metrics['R2_log_price']:.4f}")
    print(f" MAE            : ${metrics['MAE']:,.0f}")
    print(f" MAPE           : {metrics['MAPE'] * 100:.2f}%")
    print(f" MdAPE          : {metrics['MdAPE'] * 100:.2f}%")

    plot_importance(model, features, filename=plot_file)
    plot_actual_vs_predicted(
        y_test_price,
        pred_price,
        filename=plot_file.replace("feature_importance", "actual_vs_predicted"),
    )

    return model, features, metrics


if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)

    evaluate_period(
        df,
        ["2025-06", "2025-07", "2025-08", "2025-09", "2025-10", "2025-11"],
        "2025-12",
        "December 2025 as test set",
        "feature_importance_dec.png",
    )

    if "2026-01" in df["YearMonth"].unique():
        evaluate_period(
            df,
            ["2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"],
            "2026-01",
            "January 2026 as test set",
            "feature_importance_jan.png",
        )
    else:
        print("\n>> January 2026 data not found in merged files. Skipping January evaluation.")



