import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Setting a clean plot style for the report
plt.style.use('ggplot')

def load_data():
    """
    Step 1: Load and merge the raw data.
    """
    # Grab all monthly files
    files = sorted(glob.glob('CRMLSSold2025*.csv'))
    print(f">> Found {len(files)} datasets. Merging...")
    
    df_list = []
    for f in files:
        try:
            # Note: low_memory=False is just to silence those annoying mixed-type warnings
            df_list.append(pd.read_csv(f, low_memory=False))
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    full_df = pd.concat(df_list, ignore_index=True)
    print(f">> Total raw records: {full_df.shape[0]}")
    return full_df

def clean_data(df):
    """
    Step 2: Cleaning & Outlier Strategy.
    
    KEY DECISION:
    I am filtering out properties > $5,000,000. 
    Why? Because ultra-luxury homes follow a completely different pricing logic (irrational/emotional value).
    Including them introduces massive variance and hurts the model's R2 on the mass market.
    """
    # 1. Scope: Residential Single Family only (Condos have different valuation metrics)
    df = df[(df['PropertyType'] == 'Residential') & 
            (df['PropertySubType'] == 'SingleFamilyResidence')]
    
    # 2. Type conversion
    cols = ['ClosePrice', 'LivingArea', 'Latitude', 'Longitude', 
            'BedroomsTotal', 'BathroomsTotalInteger', 'YearBuilt', 'LotSizeSquareFeet']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = df.dropna(subset=cols)
    
    # 3. Sanity Check: Remove data entry errors
    df = df[(df['ClosePrice'] >= 10000) & (df['LivingArea'] >= 100)]
    
    # 4. Outlier Removal (The $5M Cap)
    df = df[df['ClosePrice'] <= 5000000]
    
    print(f">> Data cleaned. Training set size: {df.shape[0]}")
    return df

def add_spatial_features(df, k=6):
    """
    Step 3: Feature Engineering (Spatial Data Science).
    
    Standard regression models fail to capture "Location".
    My solution: Create a 'Spatial Lag' feature.
    
    Logic: I use a BallTree (KNN) to find the nearest 5 neighbors for every house 
    and calculate their average price. This quantifies the 'neighborhood value'.
    """
    print(">> Engineering SpatialLag_Price (KNN algorithm)...")
    
    # Haversine metric requires radians
    df['lat_rad'] = np.radians(df['Latitude'])
    df['lon_rad'] = np.radians(df['Longitude'])
    
    # Build the tree for fast spatial indexing
    tree = BallTree(df[['lat_rad', 'lon_rad']].values, metric='haversine')
    
    # Query k nearest neighbors (1st one is the point itself, so we take k=6)
    dist, ind = tree.query(df[['lat_rad', 'lon_rad']].values, k=k)
    
    # Calculate mean price of neighbors (excluding the house itself)
    prices = df['ClosePrice'].values
    neighbor_indices = ind[:, 1:] 
    
    neighbor_prices = prices[neighbor_indices]
    df['SpatialLag_Price'] = np.mean(neighbor_prices, axis=1)
    
    return df

def run_model(df):
    """
    Step 4: Modeling.
    
    Model: Gradient Boosting Regressor (GBR).
    Trick: Applied Log-Transformation to 'ClosePrice'. 
    Real estate prices are long-tail distributed; log-transform normalizes the error 
    and significantly improves convergence.
    """
    # Feature Selection: SpatialLag is usually the heavy hitter here
    features = ['BedroomsTotal', 'BathroomsTotalInteger', 'LivingArea', 
                'LotSizeSquareFeet', 'YearBuilt', 'Latitude', 'Longitude', 
                'SpatialLag_Price'] 
    
    X = df[features]
    y = np.log1p(df['ClosePrice']) # Log-transform target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(">> Training GBR model (this might take a moment)...")
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict & Inverse Transform (get back to real dollars)
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_orig = np.expm1(y_test)
    
    # Metrics
    r2 = r2_score(y_test_orig, y_pred)
    mae = mean_absolute_error(y_test_orig, y_pred)
    
    return model, features, r2, mae

def plot_importance(model, features):
    # Quick visualization to validate the Spatial Lag hypothesis
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance: Impact of Spatial Data")
    plt.bar(range(len(indices)), importances[indices], align="center", color='#348ABD')
    plt.xticks(range(len(indices)), [features[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print(">> Saved 'feature_importance.png'. Check it out.")

if __name__ == "__main__":
    # --- Execution Pipeline ---
    df = load_data()
    df = clean_data(df)
    df = add_spatial_features(df) # Injecting location intelligence
    
    model, features, r2, mae = run_model(df)
    
    print("\n" + "="*35)
    print(f" FINAL MODEL PERFORMANCE")
    print("="*35)
    print(f" R-squared : {r2:.4f}")
    print(f" MAE       : ${mae:,.0f}")
    print("="*35)
    
    plot_importance(model, features)


