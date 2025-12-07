import pandas as pd
import numpy as np
from pathlib import Path

def load_data(file_path='../data/MachineLearningRating_v3.txt'):
    """
    Load and preprocess the insurance dataset.
    
    Parameters:
    - file_path: Path to the data file (default: '../data/MachineLearningRating_v3.txt')
    
    Returns:
    - DataFrame with preprocessed data
    """
    # Read the data with proper encoding and separator
    df = pd.read_csv(
        file_path,
        sep='|',  # Pipe-separated values
        encoding='latin1',  # Handles special characters
        dtype={
            'PolicyID': str,
            'PostalCode': str,
            'TransactionMonth': str  # Will be converted to datetime
        },
        parse_dates=['TransactionMonth'],
        dayfirst=True  # For dates in DD/MM/YYYY format
    )
    
    # Convert numeric columns
    numeric_cols = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CapitalOutstanding']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate derived metrics
    df['ClaimFrequency'] = (df['TotalClaims'] > 0).astype(int)
    df['ClaimSeverity'] = df['TotalClaims'].where(df['TotalClaims'] > 0, np.nan)
    df['LossRatio'] = (df['TotalClaims'] / df['TotalPremium']).clip(0, 5)  # Cap at 500% to handle outliers
    
    # Clean categorical columns
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].str.strip().str.upper()
    
    if 'Province' in df.columns:
        df['Province'] = df['Province'].str.strip().str.upper()
    
    print(f"✅ Data loaded successfully! Shape: {df.shape}")
    print(f"📅 Date range: {df['TransactionMonth'].min().date()} to {df['TransactionMonth'].max().date()}")
    print(f"💰 Total Premium: ${df['TotalPremium'].sum()/1e6:.2f}M")
    print(f"💸 Total Claims: ${df['TotalClaims'].sum()/1e6:.2f}M")
    
    return df

# Example usage:
if __name__ == "__main__":
    # Try to load data from different possible locations
    possible_paths = [
        '../data/MachineLearningRating_v3.txt',
        'data/MachineLearningRating_v3.txt',
        'MachineLearningRating_v3.txt'
    ]
    
    for path in possible_paths:
        try:
            print(f"Trying to load data from: {path}")
            df = load_insurance_data(path)
            print("✅ Successfully loaded data!")
            print("\nFirst few rows of the dataset:")
            print(df.head())
            break
        except Exception as e:
            print(f"❌ Failed to load from {path}: {str(e)}")
    else:
        print("❌ Could not find the data file. Please check the file path.")