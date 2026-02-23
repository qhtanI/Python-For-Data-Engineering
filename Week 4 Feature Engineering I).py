import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ==========================================
# PART 1: ADVANCED TOOLS (Week 1 & 3)
# ==========================================

def log_pipeline_step(step_name, *args, **kwargs):
    """Log the progress of the AI pipeline."""
    print(f"\n[STEP] {step_name}")
    for arg in args: print(f" - Action: {arg}")
    if kwargs: print(f" - Config: {kwargs}")

# ==========================================
# PART 2: PRODUCTION CLEANING (Week 2 & 3)
# ==========================================



def production_cleaner(df):
    """Handles missing values, types, and outliers."""
    # 1. Standardize Missing Tokens
    df = df.replace(["?", "N/A", "none"], np.nan)
    
    # 2. Type Casting & Missing Values
    # Use Median for numbers (Robust) and Mode for text
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    # 3. Outlier Strategy: Capping (Winsorization)
    # Instead of deleting, we 'cap' extreme values at the 99th percentile
    for col in num_cols:
        upper_limit = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=upper_limit)
        
    return df

# ==========================================
# PART 3: FEATURE ENGINEERING (Week 4)
# ==========================================



[Image of machine learning feature engineering process]


def engineer_features(df):
    """Transforms raw data into model-ready inputs."""
    
    # 1. Log Transformation (Handling Skewness)
    # Helps normalize data with long tails (like income or spend)
    if 'spend' in df.columns:
        df['spend_log'] = np.log1p(df['spend'])
    
    # 2. Binning (Discretization)
    # Turning a continuous age into categories
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], bins=[0, 18, 65, 120], labels=['Minor', 'Adult', 'Senior'])
        
    # 3. Encoding (One-Hot for Nominal, Codes for Ordinal)
    # Convert 'membership' into numerical codes
    if 'membership' in df.columns:
        df['membership_code'] = df['membership'].astype('category').cat.codes
        
    return df

# ==========================================
# PART 4: INTEGRATED EXECUTION
# ==========================================

if __name__ == "__main__":
    # Sample Dirty Data
    raw_data = {
        'age': [25, 30, np.nan, 150],
        'spend': [100, 200, 50, 1000000], # Extreme Outlier
        'membership': ['bronze', 'gold', 'silver', 'bronze'],
        'city': ['NY', 'SF', 'NY', 'LA']
    }
    df = pd.DataFrame(raw_data)

    # 1. Logging (Week 1 Logic)
    log_pipeline_step("Preprocessing", "Cleaning", "Feature Engineering", version="2.0")

    # 2. Cleaning (Week 3 Logic)
    df_cleaned = production_cleaner(df)

    # 3. Engineering (Week 4 Logic)
    df_final = engineer_features(df_cleaned)

    print("\n--- Final Model-Ready Data ---")
    print(df_final.head())

    # 4. Save Metadata (Day 5 & 14 Skill)
    with open("pipeline_metadata.json", "w") as f:
        json.dump({"samples": len(df_final), "status": "Ready for Training"}, f)
