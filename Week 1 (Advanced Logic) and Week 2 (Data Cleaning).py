import pandas as pd
import numpy as np
import json

# ==========================================
# PART 1: ADVANCED PYTHON TOOLS (Week 1)
# ==========================================

def log_process_step(step_name, *details, **metadata):
    """Uses *args and **kwargs to log pipeline progress."""
    print(f"\n>>> Executing: {step_name}")
    for detail in details:
        print(f"  - Action: {detail}")
    if metadata:
        print(f"  - Config: {metadata}")

def save_pipeline_config(config_dict, filename="config.json"):
    """Saves dictionary as JSON (Day 5 - Persistence)."""
    with open(filename, 'w') as f:
        json.dump(config_dict, f, indent=4)

# ==========================================
# PART 2: THE PREPROCESSING ENGINE (Week 2)
# ==========================================



def clean_data_master(df):
    """
    Standardizes data using statistical strategies:
    1. Token Normalization
    2. Imputation with Indicators
    3. IQR Outlier Capping
    """
    
    # A. Normalize Missing Tokens (Day 6)
    df = df.replace(["N/A", "not reported", "?", "none"], np.nan)

    # B. Numerical Processing (Day 7 & 10)
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        # 1. Add Indicator Column (Best Practice)
        if df[col].isna().any():
            df[f'{col}_was_missing'] = df[col].isna().astype(int)
        
        # 2. Impute with Median (Robust against outliers)
        df[col] = df[col].fillna(df[col].median())
        
        # 3. Outlier Handling: IQR Capping (Day 10)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower, upper=upper)

    # C. Categorical Processing (Day 7 & 9)
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        # 1. Impute with Mode (Most Frequent)
        df[col] = df[col].fillna(df[col].mode()[0])
        
        # 2. Functional Cleaning with Lambda (Day 3)
        df[col] = df[col].apply(lambda x: str(x).strip().lower())

    # D. Deduplication (Day 8)
    df = df.drop_duplicates()
    
    return df

# ==========================================
# PART 3: EXECUTION (Main Script)
# ==========================================

if __name__ == "__main__":
    # 1. Setup Dirty Data
    data = {
        'price': [100, 150, np.nan, 999999], # Outlier & Missing
        'category': ['  Tech', 'Home', np.nan, 'Tech'], # Missing & Spaces
        'id': [1, 2, 2, 3] # Duplicate
    }
    df_raw = pd.DataFrame(data)

    # 2. Log & Process (Using Week 1 & 2 Skills)
    log_process_step("AutoClean", "Impute Median", "Clip IQR", version=1.2)
    
    
    df_final = clean_data_master(df_raw)

    # 3. Final Output
    print("\n--- Processed Dataset ---")
    print(df_final)

    # 4. Save Metadata (Day 5)
    save_pipeline_config({"rows": len(df_final), "status": "success"})
