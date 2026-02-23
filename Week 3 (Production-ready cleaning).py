import pandas as pd
import numpy as np
import time

# ==========================================
# PART 1: OUTLIER STRATEGIES (Day 11)
# ==========================================

# THEORETICAL NOTE: 
# Outliers are not always "bad data"; they can be "critical signals" (e.g., fraud). [cite: 28, 106]
# Simply removing them can shift means and decision boundaries. [cite: 35]
# Use "Capping" (Winsorization) to limit their influence without losing the record. [cite: 47, 48]

def winsorize_series(s: pd.Series, lower_q=0.01, upper_q=0.99) -> pd.Series:
    """Cap values at percentiles. Ideal for heavy-tailed data.""" [cite: 53, 54]
    lower, upper = s.quantile(lower_q), s.quantile(upper_q)
    return s.clip(lower=lower, upper=upper) # Limits leverage on loss functions. [cite: 46]

# ==========================================
# PART 2: STRING & DATE NORMALIZATION (Day 12)
# ==========================================

# THEORETICAL NOTE: 
# Inconsistent text (case, whitespace) creates "spurious categories" that confuse models. [cite: 154]
# Naive datetime handling is risky; always standardize to a common timezone (UTC). [cite: 144, 145]

def clean_text_and_dates(df: pd.DataFrame) -> pd.DataFrame:
    if 'city' in df.columns:
        # Step: Normalization Flow -> Raw -> Lower -> Remove Punctuation -> Canonical [cite: 169]
        df['city'] = (df['city']
                      .str.strip().str.lower()
                      .str.replace(r"[^a-z\s]", "", regex=True) # Remove special characters [cite: 168]
                      .str.replace(r"\s+", " ", regex=True)) # Collapse whitespace [cite: 169]

    if 'signup_time' in df.columns:
        # Use errors='coerce' to safely handle invalid calendar dates. [cite: 171]
        df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
        # Localize to UTC to ensure consistency across different regions. [cite: 144]
        df['signup_time'] = df['signup_time'].dt.tz_localize("UTC", ambiguous='coerce')
    
    return df

# ==========================================
# PART 3: LARGE DATASET PERFORMANCE (Day 13)
# ==========================================

# THEORETICAL NOTE: 
# "Vectorization" is the fastest way to clean data. [cite: 177]
# Avoid Python loops (iterrows) because they are memory-intensive and slow. [cite: 178]
# For massive files, use "chunksize" to bound memory usage. [cite: 175]

# 

# ==========================================
# PART 4: PRODUCTION PIPELINE (Day 14)
# ==========================================

# THEORETICAL NOTE: 
# Order of operations matters: 1. Types -> 2. Missing -> 3. Outliers -> 4. Strings. [cite: 185]
# Outlier detection on wrong data types is meaningless. [cite: 185]

def production_pipeline(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Function-based modular design for reproducibility.""" [cite: 183, 184]
    df = df_raw.copy()

    # 1. Type Normalization: Force numeric before math operations. [cite: 185]
    df['income'] = pd.to_numeric(df['income'], errors='coerce')

    # 2. Imputation: Fill gaps with Median (robust to outliers). [cite: 194]
    # Add indicators to let the model know data was guessed. [cite: 194]
    df['income_missing'] = df['income'].isna().astype(int)
    df['income'] = df['income'].fillna(df['income'].median())

    # 3. Handle Outliers: Cap at 99th percentile. [cite: 194]
    df['income'] = winsorize_series(df['income'], upper_q=0.99)

    # 4. Clean Strings/Dates
    df = clean_text_and_dates(df)

    # 5. Validation: Assert key invariants (e.g., no negative income). [cite: 187]
    # assert df['income'].min() >= 0, "Validation failed: Negative income"
    
    return df

# ==========================================
# EXECUTION & LOGGING
# ==========================================

if __name__ == "__main__":
    # Simulate messy production data
    data = {
        'income': [50000, 60000, np.nan, 1e9], # Large outlier + Missing [cite: 31]
        'city': [' New York ', 'nyc', 'NY', 'SF!'], # Unstandardized [cite: 158]
        'signup_time': ['2024-01-01', '01/01/2024', 'invalid_date', '2024-05-01']
    }
    df_raw = pd.DataFrame(data)

    # Logging: Record counts and missingness summaries for audit trails. [cite: 187]
    start_time = time.perf_counter()
    df_clean = production_pipeline(df_raw)
    duration = time.perf_counter() - start_time

    print(f"Pipeline finished in {duration:.4f}s. Rows: {len(df_clean)}")
    print(df_clean.head())
