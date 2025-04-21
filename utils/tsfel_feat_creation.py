import time
import gc # Import garbage collector
import tsfel
import multiprocessing
from joblib import Parallel, delayed
import pandas as pd
import numpy as np

def process_customer_group_for_joblib(group_info, cfg, id_col, ts_col):
    """
    Applies TSFEL feature extraction to a single customer's data.
    Designed to be called by joblib.Parallel.
    Includes verbose prints for skipped customers.

    Args:
        group_info (tuple): A tuple containing (customer_id, customer_data_df).
        cfg (dict): The TSFEL configuration dictionary.
        id_col (str): Name of the customer ID column.
        ts_col (str): Name of the timestamp column.

    Returns:
        pd.DataFrame or None: A DataFrame containing the extracted features for the customer
                              (indexed by customer ID), or None if processing fails/skipped.
    """
    customer_id, customer_data = group_info # Unpack the tuple from groupby

    try:
        # --- Data Validation / Column Selection ---
        if len(customer_data) < 6:
            # print(f"Skipping {customer_id}: Insufficient data points ({len(customer_data)}).") # Verbose skipping
            return None

        numeric_cols = customer_data.select_dtypes(include=['number']).columns
        cols_to_process = [col for col in numeric_cols if col not in [id_col, ts_col]]

        if not cols_to_process:
             # print(f"Skipping {customer_id}: No valid numeric columns found.") # Verbose skipping
            return None

        # --- TSFEL Extraction ---
        features_df = tsfel.time_series_features_extractor(cfg, customer_data[cols_to_process], verbose=0)

        # --- Result Handling ---
        if features_df.empty:
            # print(f"Skipping {customer_id}: TSFEL returned empty features.") # Verbose skipping
            return None

        # Set index *before* returning
        features_df.index = [customer_id] * len(features_df)
        features_df.index.name = id_col

        return features_df

    except Exception as e:
        print(f"Error processing customer {customer_id}: {type(e).__name__} - {e}")
        return None
    
def extract_tsfel_features_joblib_optimized(df, customer_id_col, timestamp_col, features_to_extract=None, n_jobs=-1, verbose=10):
    """
    Extracts time-series features using TSFEL in parallel using joblib
    (provides progress updates & ETA), with robust concatenation using reindexing.

    Args:
        df (pd.DataFrame): Input DataFrame.
        customer_id_col (str): Customer ID column name.
        timestamp_col (str): Timestamp column name.
        features_to_extract (dict, optional): TSFEL config dict. Defaults to all features.
        n_jobs (int, optional): Number of parallel processes (-1 for all cores). Defaults to -1.
        verbose (int, optional): Joblib verbosity level. Higher values (e.g., 5, 10)
                                 show progress bars and ETA. Defaults to 10.

    Returns:
        pd.DataFrame: DataFrame with customer_ID as index and TSFEL features as columns.
                      Returns an empty DataFrame on major errors or if no features are extracted.
    """
    if df.empty:
        print("Input DataFrame is empty.")
        return pd.DataFrame()

    print(f"Input shape: {df.shape}, Grouping by: {customer_id_col}, Timestamp: {timestamp_col}")

    # --- Data Preparation ---
    df_sorted = df.sort_values(by=[customer_id_col, timestamp_col])

    # --- TSFEL Configuration ---
    # (Configuration logic remains the same)
    if features_to_extract is None:
        cfg = tsfel.get_features_by_domain()
        print("Using default (all) TSFEL feature configuration.")
    elif isinstance(features_to_extract, dict):
        cfg = features_to_extract
        print("Using provided TSFEL configuration dictionary.")
    else:
         print("Warning: Invalid 'features_to_extract' format. Using default config.")
         cfg = tsfel.get_features_by_domain()

    # --- Grouping ---
    grouped_data = df_sorted.groupby(customer_id_col)
    num_groups = len(grouped_data)
    if num_groups == 0:
        print("No groups found for the specified customer ID.")
        del df_sorted
        gc.collect()
        return pd.DataFrame()
    print(f"Processing {num_groups} customer groups...")

    del df_sorted
    gc.collect()

    # --- Determine Number of Processes ---
    if n_jobs == -1:
        num_cores = multiprocessing.cpu_count()
    else:
        num_cores = min(n_jobs, multiprocessing.cpu_count())
    print(f"Setting up Joblib with {num_cores} processes (verbose={verbose})...")


    # --- Parallel Execution using joblib.Parallel ---
    start_time = time.time()
    results = Parallel(n_jobs=num_cores, verbose=verbose)(
        delayed(process_customer_group_for_joblib)(
            group_info,
            cfg,
            customer_id_col,
            timestamp_col
        )
        for group_info in grouped_data
    )
    end_time = time.time()
    print(f"\nJoblib parallel processing finished in {end_time - start_time:.2f} seconds.")


    # --- Combine Results (Robust Method using Reindexing) ---
    start_concat_time = time.time()
    successful_results = [res for res in results if res is not None and not res.empty]
    del results
    gc.collect()

    if not successful_results:
        print("No features were successfully extracted for any customer.")
        return pd.DataFrame()

    print(f"Successfully processed {len(successful_results)} out of {num_groups} groups.")
    print("Preparing results for concatenation...")

    try:
        # 1. Find the union of all columns from all successful results
        all_columns = set()
        for df_res in successful_results:
            all_columns.update(df_res.columns)
        all_columns = sorted(list(all_columns)) # Sort for consistent order

        # 2. Reindex each result DataFrame to include all columns, filling missing with NaN
        #    This ensures all DataFrames have the same structure before concatenation.
        reindexed_results = [
            df_res.reindex(columns=all_columns, fill_value=np.nan)
            for df_res in successful_results
        ]
        del successful_results # Free memory from original list
        gc.collect()

        # 3. Concatenate the reindexed DataFrames. pd.concat is efficient here.
        print("Concatenating reindexed results...")
        final_features_df = pd.concat(reindexed_results)

        # Index name should be preserved from worker function, but set again just in case
        final_features_df.index.name = customer_id_col

        del reindexed_results # Free memory from reindexed list
        gc.collect()

        print("Concatenation complete.")
        end_concat_time = time.time()
        print(f"Robust concatenation finished in {end_concat_time - start_concat_time:.2f} seconds.")

    except Exception as e:
        print(f"Error during robust concatenation: {e}")
        # Clean up potential large lists
        try: del successful_results
        except NameError: pass
        try: del reindexed_results
        except NameError: pass
        gc.collect()
        return pd.DataFrame()

    print(f"Final features DataFrame shape: {final_features_df.shape}")
    return final_features_df