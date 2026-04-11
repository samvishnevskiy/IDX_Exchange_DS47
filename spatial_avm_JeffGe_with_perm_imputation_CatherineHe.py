
import argparse 
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
from sklearn.neighbors import BallTree, NearestNeighbors
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
ZIP_MIN_SAMPLES = 5  

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


# Perumutation imputation

class PermutationImputer:
    
    def __init__(self, df, random_state=42):
        self.df_original = df.copy()
        self.df_imputed = df.copy()
        self.random_state = random_state
        self.imputation_log = {}
        np.random.seed(random_state)
    
    def simple_random_permutation(self, column):
        null_mask = self.df_imputed[column].isnull()
        n_missing = null_mask.sum()
        
        if n_missing == 0:
            return self.df_imputed
        
        non_null_values = self.df_imputed.loc[~null_mask, column].values
        if len(non_null_values) == 0:
            return self.df_imputed
        
        imputed_values = np.random.choice(non_null_values, size=n_missing, replace=True)
        self.df_imputed.loc[null_mask, column] = imputed_values
        
        self.imputation_log[column] = {
            'method': 'simple_random',
            'n_imputed': n_missing
        }
        return self.df_imputed
    
    def conditional_permutation(self, column, condition_columns, n_neighbors=5):
        null_mask = self.df_imputed[column].isnull()
        n_missing = null_mask.sum()
        
        if n_missing == 0:
            return self.df_imputed
        
        complete_mask = self.df_imputed[condition_columns].notna().all(axis=1)
        donor_mask = complete_mask & (~null_mask)
        recipient_mask = null_mask & complete_mask
        
        if donor_mask.sum() < n_neighbors:
            return self.simple_random_permutation(column)
        
        X_donors = self.df_imputed.loc[donor_mask, condition_columns].values
        X_recipients = self.df_imputed.loc[recipient_mask, condition_columns].values
        y_donors = self.df_imputed.loc[donor_mask, column].values
        
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        knn.fit(X_donors)
        distances, indices = knn.kneighbors(X_recipients)
        
        imputed_values = []
        for neighbor_indices in indices:
            chosen_idx = np.random.choice(neighbor_indices)
            imputed_values.append(y_donors[chosen_idx])
        
        self.df_imputed.loc[recipient_mask, column] = imputed_values
        
        self.imputation_log[column] = {
            'method': 'conditional_knn',
            'n_imputed': len(imputed_values)
        }
        return self.df_imputed
    
    def get_imputed_dataframe(self):
        return self.df_imputed.copy()


def impute_missing_values(df, random_state=42):
    critical_columns = [
        'LivingArea',
        'Latitude',
        'Longitude',
        'BedroomsTotal',
        'BathroomsTotalInteger',
        'YearBuilt',
        'LotSizeSquareFeet'
    ]
    
    available_critical = [col for col in critical_columns if col in df.columns]
    

    rows_with_missing = df[available_critical].isnull().any(axis=1).sum()
    

    imputer = PermutationImputer(df, random_state=random_state)
    
    # Geographic 
    if 'Latitude' in critical_columns:
        imputer.simple_random_permutation('Latitude')
    if 'Longitude' in critical_columns:
        imputer.simple_random_permutation('Longitude')
    
    # YearBuilt (conditional on location)
    if 'YearBuilt' in critical_columns and 'Latitude' in critical_columns:
        imputer.conditional_permutation('YearBuilt', ['Latitude', 'Longitude'], n_neighbors=5)
    elif 'YearBuilt' in critical_columns:
        imputer.simple_random_permutation('YearBuilt')
    
    # Room counts (conditional on location)
    if 'BedroomsTotal' in critical_columns and 'Latitude' in critical_columns:
        imputer.conditional_permutation('BedroomsTotal', ['Latitude', 'Longitude'], n_neighbors=5)
    elif 'BedroomsTotal' in critical_columns:
        imputer.simple_random_permutation('BedroomsTotal')
    
    if 'BathroomsTotalInteger' in critical_columns and 'BedroomsTotal' in critical_columns:
        imputer.conditional_permutation('BathroomsTotalInteger',
                                       ['BedroomsTotal', 'Latitude', 'Longitude'], n_neighbors=5)
    elif 'BathroomsTotalInteger' in critical_columns:
        imputer.simple_random_permutation('BathroomsTotalInteger')
    
    # Property sizes (conditional on rooms and location)
    if 'LivingArea' in critical_columns and 'BedroomsTotal' in critical_columns:
        imputer.conditional_permutation('LivingArea',
                                       ['BedroomsTotal', 'BathroomsTotalInteger', 'Latitude', 'Longitude'],
                                       n_neighbors=5)
    elif 'LivingArea' in critical_columns:
        imputer.simple_random_permutation('LivingArea')
    
    if 'LotSizeSquareFeet' in critical_columns and 'LivingArea' in critical_columns:
        imputer.conditional_permutation('LotSizeSquareFeet',
                                       ['LivingArea', 'Latitude', 'Longitude'], n_neighbors=5)
    elif 'LotSizeSquareFeet' in critical_columns:
        imputer.simple_random_permutation('LotSizeSquareFeet')
    
    df_imputed = imputer.get_imputed_dataframe()
    
    # Validate
    rows_with_missing_after = df_imputed[df.columns].isnull().any(axis=1).sum()
    rows_saved = rows_with_missing - rows_with_missing_after
    
    return df_imputed


# Original model
def load_data(pattern="CRMLSSold20*.csv"):
    files = sorted(glob.glob(pattern))

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
    
    # Apply imputation before any cleaning
    df = impute_missing_values(df, random_state=RANDOM_STATE)
    
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


def add_engineered_numeric_features(df, reference_year=None):
    """FIXED: Added reference_year parameter."""
    df = df.copy()

    current_year = reference_year if reference_year is not None else int(df["CloseDateParsed"].dt.year.max())

    df.loc[:, "HomeAge"] = current_year - df["YearBuilt"]
    df.loc[:, "HomeAge"] = df["HomeAge"].clip(lower=0)

    df.loc[:, "BathsPerBedroom"] = df["BathroomsTotalInteger"] / np.maximum(df["BedroomsTotal"], 1)
    df.loc[:, "LotLivingRatio"] = df["LotSizeSquareFeet"] / np.maximum(df["LivingArea"], 1)
    df.loc[:, "LogLivingArea"] = np.log1p(df["LivingArea"])
    df.loc[:, "LogLotSize"] = np.log1p(df["LotSizeSquareFeet"])

    return df


def add_spatial_lag_feature(train_df, target_df, k=5, is_self=False):
    train_df = train_df.copy()
    target_df = target_df.copy()

    if len(train_df) < 2:
        raise ValueError("Training data is too small for spatial lag.")

    train_coords = np.radians(train_df[["Latitude", "Longitude"]].to_numpy(dtype=np.float64))
    target_coords = np.radians(target_df[["Latitude", "Longitude"]].to_numpy(dtype=np.float64))

    tree = BallTree(train_coords, metric="haversine")
    train_prices = train_df["ClosePrice"].to_numpy(dtype=np.float64)

    if is_self:
        k_query = min(k + 1, len(train_df))
        _, ind = tree.query(target_coords, k=k_query)
        neighbor_idx = ind[:, 1:] if k_query > 1 else ind
    else:
        k_query = min(k, len(train_df))
        _, ind = tree.query(target_coords, k=k_query)
        neighbor_idx = ind

    neighbor_prices = train_prices[neighbor_idx]
    target_df.loc[:, "SpatialLag_Price"] = np.mean(neighbor_prices, axis=1)

    return target_df


def add_zip_median_feature(train_df, target_df, min_samples=ZIP_MIN_SAMPLES):
    train_df = train_df.copy()
    target_df = target_df.copy()

    global_median = train_df["ClosePrice"].median()

    if "ZIP_CLEAN" not in train_df.columns:
        train_df.loc[:, "ZIP_MedianPrice"] = global_median
        target_df.loc[:, "ZIP_MedianPrice"] = global_median
        return train_df, target_df

    zip_counts = train_df.groupby("ZIP_CLEAN", dropna=True)["ClosePrice"].count()
    zip_median_all = train_df.groupby("ZIP_CLEAN", dropna=True)["ClosePrice"].median()

    zip_median = zip_median_all[zip_counts >= min_samples]

    n_zips_total = len(zip_median_all)
    n_zips_kept = len(zip_median)
    print(f">> ZIP median: {n_zips_kept}/{n_zips_total} ZIPs have >={min_samples} samples; rest fall back to global median")

    train_df.loc[:, "ZIP_MedianPrice"] = train_df["ZIP_CLEAN"].map(zip_median).fillna(global_median)
    target_df.loc[:, "ZIP_MedianPrice"] = target_df["ZIP_CLEAN"].map(zip_median).fillna(global_median)

    return train_df, target_df


def prepare_features(train_df, test_df, spatial_k=5):
    reference_year = int(train_df["CloseDateParsed"].dt.year.max())

    train_df = add_engineered_numeric_features(train_df, reference_year=reference_year)
    test_df = add_engineered_numeric_features(test_df, reference_year=reference_year)

    train_df = add_spatial_lag_feature(train_df, train_df, k=spatial_k, is_self=True)
    test_df = add_spatial_lag_feature(train_df, test_df, k=spatial_k, is_self=False)

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


def fit_and_compare_models(X_train, y_train, X_test, y_test):
    """Train multiple models and compare performance."""
    results = []
    fitted_models = {}
    predictions = {}

    dummy = DummyRegressor(strategy="median")
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    
    r2_dummy = r2_score(y_test, y_pred_dummy)
    mape_dummy = mean_absolute_percentage_error(y_test, y_pred_dummy)
    mae_dummy = mean_absolute_error(y_test, y_pred_dummy)
    rmse_dummy = np.sqrt(np.mean((y_test - y_pred_dummy)**2))
    
    results.append({
        "Model": "dummy_median",
        "R2_price": r2_dummy,
        "MAPE": mape_dummy,
        "MAE": mae_dummy,
        "RMSE": rmse_dummy
    })
    fitted_models["dummy_median"] = dummy
    predictions["dummy_median"] = y_pred_dummy

    lin_reg = LinearRegression()
    lin_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", lin_reg)
    ])
    lin_ttr = TransformedTargetRegressor(
        regressor=lin_pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )
    lin_ttr.fit(X_train, y_train)
    y_pred_lin = lin_ttr.predict(X_test)
    
    r2_lin = r2_score(y_test, y_pred_lin)
    mape_lin = mean_absolute_percentage_error(y_test, y_pred_lin)
    mae_lin = mean_absolute_error(y_test, y_pred_lin)
    rmse_lin = np.sqrt(np.mean((y_test - y_pred_lin)**2))
    
    results.append({
        "Model": "linear_regression",
        "R2_price": r2_lin,
        "MAPE": mape_lin,
        "MAE": mae_lin,
        "RMSE": rmse_lin
    })
    fitted_models["linear_regression"] = lin_ttr
    predictions["linear_regression"] = y_pred_lin

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_ttr = TransformedTargetRegressor(
        regressor=rf,
        func=np.log1p,
        inverse_func=np.expm1
    )
    rf_ttr.fit(X_train, y_train)
    y_pred_rf = rf_ttr.predict(X_test)
    
    r2_rf = r2_score(y_test, y_pred_rf)
    mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(np.mean((y_test - y_pred_rf)**2))
    
    results.append({
        "Model": "random_forest",
        "R2_price": r2_rf,
        "MAPE": mape_rf,
        "MAE": mae_rf,
        "RMSE": rmse_rf
    })
    fitted_models["random_forest"] = rf_ttr
    predictions["random_forest"] = y_pred_rf

    if XGBOOST_AVAILABLE:
        xgb = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        xgb_ttr = TransformedTargetRegressor(
            regressor=xgb,
            func=np.log1p,
            inverse_func=np.expm1
        )
        xgb_ttr.fit(X_train, y_train)
        y_pred_xgb = xgb_ttr.predict(X_test)
        
        r2_xgb = r2_score(y_test, y_pred_xgb)
        mape_xgb = mean_absolute_percentage_error(y_test, y_pred_xgb)
        mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
        rmse_xgb = np.sqrt(np.mean((y_test - y_pred_xgb)**2))
        
        results.append({
            "Model": "xgboost",
            "R2_price": r2_xgb,
            "MAPE": mape_xgb,
            "MAE": mae_xgb,
            "RMSE": rmse_xgb
        })
        fitted_models["xgboost"] = xgb_ttr
        predictions["xgboost"] = y_pred_xgb

    results_df = pd.DataFrame(results).sort_values("R2_price", ascending=False)
    
    return fitted_models, predictions, results_df


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


def log_linear_coefficients(fitted_model, feature_cols, test_month_str):
    inner = get_inner_model(fitted_model)
    if not hasattr(inner, "coef_"):
        return

    coef_df = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": inner.coef_,
    }).sort_values("Coefficient", key=abs, ascending=False)

    print(f"\n>> Linear Regression Coefficients ({test_month_str}):")
    print(coef_df.to_string(index=False))

    coef_path = TABLES_DIR / f"linear_coefficients_{test_month_str}.csv"
    coef_df.to_csv(coef_path, index=False)
    print(f">> Saved linear coefficients to {coef_path}")


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

    for model_name, fitted_model in fitted_models.items():
        if model_name in {"random_forest", "xgboost"}:
            plot_feature_importance(
                fitted_model,
                feature_cols,
                title=f"Feature Importance ({model_name}, {test_month_str})",
                filename=PLOTS_DIR / f"feature_importance_{test_month_str}_{model_name}.png",
            )
        elif model_name == "linear_regression":
            log_linear_coefficients(fitted_model, feature_cols, test_month_str)

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

    run_single_holdout(df, test_month=None, use_all_history=False)

    print("\n>> Running recent backtest on the last 2 available test months...")
    run_recent_backtest(df, n_recent_tests=2, use_all_history=False)
