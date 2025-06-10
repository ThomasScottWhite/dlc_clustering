import pandas as pd
import polars as pl

def read_hdf(csv_path: str) -> pd.DataFrame:
    """
    Reads an HDF5 file and processes the DataFrame.
    
    Returns:
        pd.DataFrame: Processed DataFrame with multi-level columns flattened.
    """
    df = pd.read_hdf(csv_path)
    df.columns = df.columns.droplevel(0)
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df = pl.from_pandas(df)
    return df
