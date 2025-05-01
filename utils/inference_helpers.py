import dask.dataframe as dd
from multiprocessing import Pool
import time
import gc # Import garbage collector
import tsfel
import multiprocessing
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import pickle
import tsfel

from utils.TimeSeriesFeatureCreation import TimeSeriesFeatureCreation
from utils.backfill import backfill_null_values
from utils.tsfel_feat_creation import extract_tsfel_features_joblib_optimized

def load_data(data_path, file_type = 'parquet'):
    """
    Load the data from the specified file path.

    Args:
        data_path (str): Path to the data file.
        file_type (str): Type of the file to load ('parquet' or 'csv'). Defaults to 'parquet'.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame with additional 'end_of_month' column.
    """
    if file_type == 'parquet':
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    df['S_2'] = pd.to_datetime(df['S_2'])
    df['end_of_month'] = df['S_2'] + pd.offsets.MonthEnd(0)

    return df

def filter_and_impute_df(df, non_null_cols_path):
    """
    Filter the DataFrame to include only non-null columns and impute missing values.

    Args:
        df (pd.DataFrame): Input DataFrame.
        non_null_cols_path (str): Path to the CSV file containing the list of non-null columns.

    Returns:
        pd.DataFrame: Filtered and imputed DataFrame.
    """

    non_null_cols = pd.read_csv(non_null_cols_path)['Feature'].values
    df = df[non_null_cols]

    cat_cols = ['D_126', 'D_63']
    df[cat_cols] = df[cat_cols].astype('category')

    df = backfill_null_values(df, columns=None, group_by='customer_ID')

    return df

def encode_df(df, cat_encoder_path):
    """
    Encode categorical columns in the DataFrame using a pre-trained encoder.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cat_encoder_path (str): Path to the pickled categorical encoder.

    Returns:
        pd.DataFrame: DataFrame with encoded categorical columns.
    """
    cat_encoder = pickle.load(open(cat_encoder_path, 'rb'))

    df = pd.concat([df.drop(columns=cat_encoder.feature_names_in_),
                       pd.DataFrame(cat_encoder.transform(df[cat_encoder.feature_names_in_]),
                                    columns=cat_encoder.get_feature_names_out(),index=df.index)], axis=1)
    
    return df

def get_imp_cols_append(file_path):
    """
    Retrieve important columns to append from a file.

    Args:
        file_path (str): Path to the CSV file containing column importance information.

    Returns:
        np.ndarray: Array of important columns to append, including 'customer_ID' and 'end_of_month'.
    """
    cols_to_append = pd.read_csv(file_path)
    cols_to_append = cols_to_append[cols_to_append['Importance'] == 1]['Feature'].values
    cols_to_append = np.append(['customer_ID', 'end_of_month'], cols_to_append)

    return cols_to_append

def create_ts_features(df, imp_cols_path, ts_feat_imp_path, select_date = '2018-03-31'):
    """
    Create time series features and filter data for a specific date.

    Args:
        df (pd.DataFrame): Input DataFrame.
        imp_cols_path (str): Path to the CSV file containing important columns.
        ts_feat_imp_path (str): Path to the CSV file containing time series feature importance.
        select_date (str): Date to filter the data. Defaults to '2018-03-31'.

    Returns:
        pd.DataFrame: DataFrame with time series features for the selected date.
    """

    ts_creator = TimeSeriesFeatureCreation(create_2_diff=False, verbose = False, id_col='customer_ID', date_col = 'end_of_month',
                                       rolling_window_size=6, span = 6, num_lags=3)
    

    cols_to_append = get_imp_cols_append(imp_cols_path)

    df = ts_creator.transform(df[cols_to_append])
    df = df[df['end_of_month'] == select_date]

    ts_feat = pd.read_csv(ts_feat_imp_path)
    ts_feat_cols = ts_feat[ts_feat['Importance'] == 1]['Feature'].values
    ts_feat_cols = np.append(['customer_ID'], ts_feat_cols)

    df_ts = df[ts_feat_cols]
    df_ts = df_ts.fillna(0)

    return df_ts


def create_tsfel_features(df, imp_cols_path, stat_select_path, temporal_select_path):
    """
    Create TSFEL (Time Series Feature Extraction Library) features.

    Args:
        df (pd.DataFrame): Input DataFrame.
        imp_cols_path (str): Path to the CSV file containing important columns.
        stat_select_path (str): Path to the CSV file containing statistical feature selection.
        temporal_select_path (str): Path to the CSV file containing temporal feature selection.

    Returns:
        pd.DataFrame: DataFrame with extracted TSFEL features.
    """

    cfg = tsfel.get_features_by_domain(domain=['temporal', 'statistical'])

    cols_to_append = get_imp_cols_append(imp_cols_path)

    df_tsfel = extract_tsfel_features_joblib_optimized(
        df[cols_to_append],
        customer_id_col='customer_ID',
        timestamp_col='end_of_month',
        features_to_extract=cfg,
        n_jobs=-1, # Use all cores
        verbose=0 # Set verbosity for progress updates and ETA
    )

    stat_select = pd.read_csv(stat_select_path)
    temporal_select = pd.read_csv(temporal_select_path)

    stat_select_cols = stat_select[stat_select['Importance'] == 1]['Feature'].values
    temporal_select_cols = temporal_select[temporal_select['Importance'] == 1]['Feature'].values

    tsfel_cols = np.append(stat_select_cols, temporal_select_cols)

    # Check if any features were created
    if df_tsfel.empty:
        # Create an empty DataFrame with the required columns and fill with zeros
        df_tsfel = pd.DataFrame(0, index=[0], columns=tsfel_cols)
        df_tsfel['customer_ID'] = df['customer_ID'].unique()[0]

    else:
        df_tsfel = df_tsfel[tsfel_cols]
        df_tsfel.reset_index(drop=False, inplace=True)
        df_tsfel = df_tsfel.fillna(0)

    return df_tsfel
    
def combine_dfs(df, ts_df, tsfel_df, ref_cols_path, selected_date = '2018-03-31'):
    """
    Combine the original DataFrame with time series and TSFEL features.

    Args:
        df (pd.DataFrame): Original DataFrame.
        ts_df (pd.DataFrame): DataFrame with time series features.
        tsfel_df (pd.DataFrame): DataFrame with TSFEL features.
        ref_cols_path (str): Path to the parquet file containing reference columns.
        selected_date (str): Date to filter the original DataFrame. Defaults to '2018-03-31'.

    Returns:
        pd.DataFrame: Combined DataFrame with all features.
    """

    one_month_df = df[df['end_of_month'] == selected_date]

    ts_df = ts_df.drop(columns=[col for col in one_month_df.columns if col != 'customer_ID' and col in ts_df.columns])

    tsfel_df = tsfel_df.drop(columns=[
        col for col in one_month_df.columns if col != 'customer_ID' and col in tsfel_df.columns
    ])

    # Merge all features into a single DataFrame using the 'customer_ID' column
    merged_df = one_month_df.merge(
        ts_df, on='customer_ID', how='left'
    ).merge(
        tsfel_df, on='customer_ID', how='left'
    )

    cols = pd.read_parquet(ref_cols_path).columns
    merged_df = merged_df[cols]
    merged_df = merged_df.fillna(0)

    return merged_df

    