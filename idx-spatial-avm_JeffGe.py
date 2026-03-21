## Model Explanation: I used XGBoost as the main model because housing prices are highly non-linear and depend on interactions between features like size, rooms, and location. Tree-based models capture these patterns much better than linear models.
## I trained the model on log-transformed prices to handle the right-skewed distribution and to focus on percentage-based errors.

## To capture location, I engineered two features: Spatial lag price, which reflects nearby home prices, ZIP median price, which captures the overall price level of each area  

Both features are built using training data only to avoid data leakage.

import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

plt.style.use("ggplot")

RANDOM_STATE = 42
MIN_TRAIN_MONTHS = 6
SPATIAL_K = 5

FORBIDDEN_FEATURES = {
    "ListPrice",
    "OriginalListPrice",
}

OUTPUT_DIR = Path("outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"
METRICS_DIR = OUTPUT_DIR / "metrics"

for folder in [OUTPUT_DIR, PLOTS_DIR, TABLES_DIR, METRICS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def load_data(pattern="CRMLSSold20*.csv"):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    print(f">> Found {len(files)} datasets. Merging...")

    frames = []
    for f in files:
        try:
            temp = pd.read_csv(f, low_memory=False)
            temp["source_file"] = os.path.basename(f)
            frames.append(temp)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not frames:
        raise ValueError("No CSV files were loaded successfully.")

    df = pd.concat(frames, ignore_index=True)
    print(f">> Total raw records: {df.shape[0]}")
    return df


def pick_first_existing(df, candidates, required=True):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"None of these columns were found: {candidates}")
    return None


def clean_postal_code(series):
    s = series.astype("string").str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.extract(r"(\d{5})", expand=False)
    return s


def base_clean(df):
    df = df.copy()

    df = df[
        (df["PropertyType"] == "Residential") &
        (df["PropertySubType"] == "SingleFamilyResidence")
    ].copy()

    date_col = pick_first_existing(
        df,
        ["CloseDate", "CloseDateTime", "CloseDateUTC"]
    )

    zip_col = pick_first_existing(
        df,
        ["PostalCode", "ZIP", "ZipCode", "Postal_Code"],
        required=False
    )

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

    for col in numeric_cols:
        if col in df.columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")
        else:
            raise ValueError(f"Missing required numeric column: {col}")

    df.loc[:, "CloseDateParsed"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=numeric_cols + ["CloseDateParsed"]).copy()

    df = df[(df["ClosePrice"] > 0) & (df["LivingArea"] > 0)].copy()

    df.loc[:, "YearMonth"] = df["CloseDateParsed"].dt.to_period("M")

    if zip_col is not None:
        df.loc[:, "ZIP_CLEAN"] = clean_postal_code(df[zip_col])
    else:
        df.loc[:, "ZIP_CLEAN"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="string")

    print(f">> Rows after scope + type cleaning: {df.shape[0]}")
    return df


def get_latest_test_month(df):
    months = sorted(df["YearMonth"].dropna().unique())
    if not months:
        raise ValueError("No valid YearMonth values found.")
    return months[-1]


def get_train_months(df, test_month, min_train_months=6, use_all_history=False):
    months = sorted(df["YearMonth"].dropna().unique())
    prior_months = [m for m in months if m < test_month]

    if len(prior_months) < min_train_months:
        raise ValueError(
            f"Need at least {min_train_months} months before {test_month}, "
            f"but only found {len(prior_months)}."
        )

    if use_all_history:
        return prior_months

    return prior_months[-min_train_months:]


def split_forward_holdout(df, test_month=None, min_train_months=6, use_all_history=False):
    if test_month is None:
        test_month = get_latest_test_month(df)

    if isinstance(test_month, str):
        test_month = pd.Period(test_month, freq="M")

    train_months = get_train_months(
        df,
        test_month=test_month,
        min_train_months=min_train_months,
        use_all_history=use_all_history,
    )

    train_df = df[df["YearMonth"].isin(train_months)].copy()
    test_df = df[df["YearMonth"] == test_month].copy()

    print(f">> Train months: {[str(m) for m in train_months]}")
    print(f">> Test month : {str(test_month)}")
    print(f">> Train size before trim: {train_df.shape[0]}")
    print(f">> Test size before trim : {test_df.shape[0]}")

    if train_df.empty or test_df.empty:
        raise ValueError("Train or test split is empty.")

    return train_df, test_df, train_months, test_month


def trim_closeprice_split_local(df, lower_q=0.005, upper_q=0.995):
    df = df.copy()

    n_before = len(df)
    lo = df["ClosePrice"].quantile(lower_q)
    hi = df["ClosePrice"].quantile(upper_q)

    df = df[(df["ClosePrice"] >= lo) & (df["ClosePrice"] <= hi)].copy()
    n_after = len(df)

    info = {
        "n_before": int(n_before),
        "n_after": int(n_after),
        "lower_threshold": float(lo),
        "upper_threshold": float(hi),
        "trimmed_rows": int(n_before - n_after),
        "trim_rate": float((n_before - n_after) / max(n_before, 1)),
    }
    return df, info


def add_engineered_numeric_features(df):
    df = df.copy()

    current_year = int(df["CloseDateParsed"].dt.year.max())

    df.loc[:, "HomeAge"] = current_year - df["YearBuilt"]
    df.loc[:, "HomeAge"] = df["HomeAge"].clip(lower=0)

    df.loc[:, "BathsPerBedroom"] = df["BathroomsTotalInteger"] / np.maximum(df["BedroomsTotal"], 1)
    df.loc[:, "LotLivingRatio"] = df["LotSizeSquareFeet"] / np.maximum(df["LivingArea"], 1)
    df.loc[:, "LogLivingArea"] = np.log1p(df["LivingArea"])
    df.loc[:, "LogLotSize"] = np.log1p(df["LotSizeSquareFeet"])

    return df


def add_spatial_lag_feature(train_df, target_df, k=5):
    train_df = train_df.copy()
    target_df = target_df.copy()

    if len(train_df) < 2:
        raise ValueError("Training data is too small for spatial lag.")

    train_coords = np.radians(train_df[["Latitude", "Longitude"]].to_numpy(dtype=np.float64))
    target_coords = np.radians(target_df[["Latitude", "Longitude"]].to_numpy(dtype=np.float64))

    tree = BallTree(train_coords, metric="haversine")
    train_prices = train_df["ClosePrice"].to_numpy(dtype=np.float64)

    if target_df.index.equals(train_df.index):
        k_train = min(k + 1, len(train_df))
        _, ind = tree.query(target_coords, k=k_train)
        neighbor_idx = ind[:, 1:] if k_train > 1 else ind
    else:
        k_test = min(k, len(train_df))
        _, ind = tree.query(target_coords, k=k_test)
        neighbor_idx = ind

    neighbor_prices = train_prices[neighbor_idx]
    target_df.loc[:, "SpatialLag_Price"] = np.mean(neighbor_prices, axis=1)

    return target_df


def add_zip_median_feature(train_df, target_df):
    train_df = train_df.copy()
    target_df = target_df.copy()

    if "ZIP_CLEAN" not in train_df.columns:
        train_df.loc[:, "ZIP_MedianPrice"] = train_df["ClosePrice"].median()
        target_df.loc[:, "ZIP_MedianPrice"] = train_df["ClosePrice"].median()
        return train_df, target_df

    zip_median = train_df.groupby("ZIP_CLEAN", dropna=True)["ClosePrice"].median()
    global_median = train_df["ClosePrice"].median()

    train_df.loc[:, "ZIP_MedianPrice"] = train_df["ZIP_CLEAN"].map(zip_median).fillna(global_median)
    target_df.loc[:, "ZIP_MedianPrice"] = target_df["ZIP_CLEAN"].map(zip_median).fillna(global_median)

    return train_df, target_df


def prepare_features(train_df, test_df, spatial_k=5):
    train_df = add_engineered_numeric_features(train_df)
    test_df = add_engineered_numeric_features(test_df)

    train_df = add_spatial_lag_feature(train_df, train_df, k=spatial_k)
    test_df = add_spatial_lag_feature(train_df, test_df, k=spatial_k)

    train_df, test_df = add_zip_median_feature(train_df, test_df)

    feature_cols = [
        "BedroomsTotal",
        "BathroomsTotalInteger",
        "LivingArea",
        "LotSizeSquareFeet",
        "YearBuilt",
        "Latitude",
        "Longitude",
        "HomeAge",
        "BathsPerBedroom",
        "LotLivingRatio",
        "LogLivingArea",
        "LogLotSize",
        "SpatialLag_Price",
        "ZIP_MedianPrice",
    ]

    forbidden_in_use = [c for c in feature_cols if c in FORBIDDEN_FEATURES]
    if forbidden_in_use:
        raise ValueError(f"Forbidden features found in model input: {forbidden_in_use}")

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_train = train_df["ClosePrice"].to_numpy(dtype=np.float64)
    y_test = test_df["ClosePrice"].to_numpy(dtype=np.float64)

    return train_df, test_df, X_train, X_test, y_train, y_test, feature_cols


def build_models():
    models = {}

    models["dummy_median"] = TransformedTargetRegressor(
        regressor=DummyRegressor(strategy="median"),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    linear_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])
    models["linear_regression"] = TransformedTargetRegressor(
        regressor=linear_pipe,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    rf_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])
    models["random_forest"] = TransformedTargetRegressor(
        regressor=rf_pipe,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    if XGBOOST_AVAILABLE:
        xgb_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                n_estimators=800,
                learning_rate=0.03,
                max_depth=6,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.05,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
                objective="reg:squarederror",
            )),
        ])
        models["xgboost"] = TransformedTargetRegressor(
            regressor=xgb_pipe,
            func=np.log1p,
            inverse_func=np.expm1,
        )

    return models


def evaluate_predictions(y_true, y_pred):
    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(np.maximum(y_pred, 0))

    return {
        "R2_price": float(r2_score(y_true, y_pred)),
        "R2_log_price": float(r2_score(y_true_log, y_pred_log)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def fit_and_compare_models(X_train, y_train, X_test, y_test):
    models = build_models()

    rows = []
    fitted = {}
    predictions = {}

    for name, model in models.items():
        print(f">> Training {name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        fitted[name] = model
        predictions[name] = pred

        metrics = evaluate_predictions(y_test, pred)
        metrics["Model"] = name
        rows.append(metrics)

    results = pd.DataFrame(rows).sort_values(by=["R2_price", "MAPE"], ascending=[False, True]).reset_index(drop=True)
    return fitted, predictions, results


def get_inner_model(fitted_model):
    inner = fitted_model.regressor_
    if isinstance(inner, Pipeline):
        return inner.named_steps["model"]
    return inner


def plot_actual_vs_predicted(y_true, y_pred, title, filename):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.25, s=12)

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")

    plt.xlabel("Actual Close Price")
    plt.ylabel("Predicted Close Price")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_residuals(y_true, y_pred, title, filename):
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, residuals, alpha=0.25, s=12)
    plt.axhline(0, linestyle="--")

    plt.xlabel("Actual Close Price")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_model_comparison(results_df, test_month, filename):
    df = results_df.copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(df["Model"], df["R2_price"])
    axes[0].set_title(f"R² by Model ({test_month})")
    axes[0].set_ylabel("R²")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(df["Model"], df["MAPE"] * 100)
    axes[1].set_title(f"MAPE by Model ({test_month})")
    axes[1].set_ylabel("MAPE (%)")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_feature_importance(fitted_model, feature_cols, title, filename):
    model = get_inner_model(fitted_model)

    if not hasattr(model, "feature_importances_"):
        return

    importance = model.feature_importances_
    order = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(order)), importance[order])
    plt.xticks(range(len(order)), [feature_cols[i] for i in order], rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_results_table(results_df, test_month):
    out = results_df.copy()
    out["MAPE_pct"] = out["MAPE"] * 100

    csv_path = TABLES_DIR / f"model_results_{test_month}.csv"
    out.to_csv(csv_path, index=False)
    return csv_path


def run_single_holdout(df, test_month=None, use_all_history=False):
    train_df, test_df, train_months, test_month = split_forward_holdout(
        df,
        test_month=test_month,
        min_train_months=MIN_TRAIN_MONTHS,
        use_all_history=use_all_history,
    )

    train_df, train_trim = trim_closeprice_split_local(train_df)
    test_df, test_trim = trim_closeprice_split_local(test_df)

    print(f">> Train size after trim: {train_df.shape[0]}")
    print(f">> Test size after trim : {test_df.shape[0]}")

    train_df, test_df, X_train, X_test, y_train, y_test, feature_cols = prepare_features(
        train_df,
        test_df,
        spatial_k=SPATIAL_K,
    )

    fitted_models, predictions, results_df = fit_and_compare_models(X_train, y_train, X_test, y_test)

    test_month_str = str(test_month)
    csv_path = save_results_table(results_df, test_month_str)

    best_model_name = results_df.iloc[0]["Model"]
    best_pred = predictions[best_model_name]

    plot_actual_vs_predicted(
        y_test,
        best_pred,
        title=f"Predicted vs Actual ({best_model_name}, {test_month_str})",
        filename=PLOTS_DIR / f"actual_vs_predicted_{test_month_str}_{best_model_name}.png",
    )

    plot_residuals(
        y_test,
        best_pred,
        title=f"Residual Plot ({best_model_name}, {test_month_str})",
        filename=PLOTS_DIR / f"residuals_{test_month_str}_{best_model_name}.png",
    )

    plot_model_comparison(
        results_df,
        test_month_str,
        filename=PLOTS_DIR / f"model_comparison_{test_month_str}.png",
    )

    if best_model_name in {"random_forest", "xgboost"}:
        plot_feature_importance(
            fitted_models[best_model_name],
            feature_cols,
            title=f"Feature Importance ({best_model_name}, {test_month_str})",
            filename=PLOTS_DIR / f"feature_importance_{test_month_str}_{best_model_name}.png",
        )

    summary = {
        "test_month": test_month_str,
        "train_months": [str(m) for m in train_months],
        "train_trim": train_trim,
        "test_trim": test_trim,
        "feature_columns": feature_cols,
        "best_model": best_model_name,
        "results_table_csv": str(csv_path),
    }

    with open(METRICS_DIR / f"run_summary_{test_month_str}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f" FINAL RESULTS | TEST MONTH = {test_month_str}")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)

    return results_df, summary


def run_recent_backtest(df, n_recent_tests=2, use_all_history=False):
    months = sorted(df["YearMonth"].dropna().unique())
    if len(months) <= MIN_TRAIN_MONTHS:
        raise ValueError("Not enough months for backtest.")

    candidate_months = months[-n_recent_tests:]
    all_results = []

    for month in candidate_months:
        try:
            results_df, summary = run_single_holdout(
                df,
                test_month=month,
                use_all_history=use_all_history,
            )
            temp = results_df.copy()
            temp["TestMonth"] = str(month)
            all_results.append(temp)
        except Exception as e:
            print(f">> Skipped {month}: {e}")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(TABLES_DIR / "recent_backtest_results.csv", index=False)
        return combined

    return pd.DataFrame()


if __name__ == "__main__":
    df = load_data("CRMLSSold20*.csv")
    df = base_clean(df)

    # 1) Main requirement-compliant run:
    #    test = latest month in the data
    run_single_holdout(df, test_month=None, use_all_history=False)

    # 2) Optional recent backtest:
    #    useful now that we also have 2026-01 and 2026-02
    #    comment this out if we only want the final latest-month result
    print("\n>> Running recent backtest on the last 2 available test months...")
    run_recent_backtest(df, n_recent_tests=2, use_all_history=False)
