import pandas as pd
import numpy as np
from pathlib import Path

def find_file(filename):
    # First check whether a full or relative path was provided
    path = Path(filename)

    if path.exists():
        return path.resolve()

    # Search the current project directory and its subfolders
    matches = list(Path.cwd().rglob(filename))

    if matches:
        return matches[0]

    # Search the user's home directory as a fallback
    matches = list(Path.home().rglob(filename))

    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not find '{filename}' on the system.")


# function to load and preview the data 
def load_data(filename):
    try:
        # create the full file path 
        path = find_file(filename)
        # skip the first four rows as the files contains headers inforamtion 
        df = pd.read_csv(path, skiprows=4)
        # print file information
        print(f"\n{filename}")
       # number of rows and columns in dataset
        print(f"Shape: {df.shape}")
        # display first few rows to incepct the structure of dataset 
        print(df.head(5))

        return df

    except FileNotFoundError:
        print(f"File not found: {filename}")
        raise

    except Exception as e:
        print(f"Error loading {filename}: {e}")
        raise



# standardise pm2.5 dataset before preprocessing 
def standardise_dataset(df):    
    try:
        # make copy of the orignal dataset so that orignal dataset remains unchanged
        df = df.copy()
        # remove the extra spaces from the column names  
        df.columns = df.columns.str.strip()
        #convert date column to datetime format 
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        hourly_cols = [col for col in df.columns if col != "Date"]
        # convert hourly PM2.5 values to numeric 
        df[hourly_cols] = df[hourly_cols].apply(pd.to_numeric, errors="coerce")
        return df
    except Exception as e:
        print(f"Error standardising dataset: {e}")
        raise


# Calculate daily pm2.5 mean using 75% data capture threshould 
def calculate_daily_mean(df):   
    try:
        df = df.copy()
         # Count valid hours for each day 
        df["valid_hours"] = df[hourly_cols].notna().sum(axis=1)
        # Count missing hours for each day 
        df["missing_hours"] = 24 - df["valid_hours"]
         # Compute the average pm2.5 value across all available hours, skipping the nans values
        df["pm25_daily_mean"] = df[hourly_cols].mean(axis=1, skipna=True)
       # Mark daily mean as missing if fewer than 18 valid hours data are available 
        df.loc[df["valid_hours"] < 18, "pm25_daily_mean"] = np.nan
        return df[["Date", "valid_hours", "missing_hours", "pm25_daily_mean"]]
    except Exception as e:
        print(f"Error calculating daily mean: {e}")
        raise
