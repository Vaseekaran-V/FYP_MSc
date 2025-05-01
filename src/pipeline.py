import pandas as pd
import pickle
import shap
import os
import sys
import logging  # Import logging module
import matplotlib.pyplot as plt  # Import for saving SHAP plots

# Setting path to load util functions
from pathlib import Path
parent_dir = Path(__file__).resolve().parents[1]  # Adjusted to point to the parent of 'src'
sys.path.append(str(parent_dir))  # Add the parent directory to sys.path

from utils.inference_helpers import (
    load_data,
    filter_and_impute_df,
    encode_df,
    create_ts_features,
    create_tsfel_features,
    combine_dfs,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),  # Log to a file
        logging.StreamHandler()  # Log to the console
    ]
)

def run_pipeline(data_path, non_null_cols_path, cat_encoder_path, ts_feat_imp_path, 
                 imp_cols_path, stat_select_path, temporal_select_path, ref_cols_path, 
                 model_path, train_data_path, output_dir, threshold=0.5):
    """
    Executes the pipeline to load data, preprocess it, generate predictions, probabilities, 
    and SHAP waterfall plots.

    Parameters:
        data_path (str): Path to the input data file.
        non_null_cols_path (str): Path to the non-null columns filter CSV.
        cat_encoder_path (str): Path to the categorical encoder pickle file.
        ts_feat_imp_path (str): Path to the time-series feature importance CSV.
        imp_cols_path (str): Path to the important columns CSV.
        stat_select_path (str): Path to the TSFEL statistical features CSV.
        temporal_select_path (str): Path to the TSFEL temporal features CSV.
        ref_cols_path (str): Path to the reference columns parquet file.
        model_path (str): Path to the trained model pickle file.
        train_data_path (str): Path to the training data parquet file.
        output_dir (str): Directory to save SHAP plots.

    Returns:
        predictions (array): Model predictions.
        probabilities (array): Model prediction probabilities.
        shap_plots (dict): Paths to the saved SHAP plots.
    """
    logging.info("Pipeline execution started.")

    # Load data
    logging.info(f"Loading data from {data_path}.")
    df = load_data(data_path=data_path)
    logging.info(f"Data loaded successfully. Unique 'end_of_month' values: {df['end_of_month'].unique()}")

    # Filter and impute data
    logging.info("Filtering and imputing data.")
    df = filter_and_impute_df(df, non_null_cols_path=non_null_cols_path)
    logging.info("Data filtering and imputation completed.")

    # Encode categorical features
    logging.info("Encoding categorical features.")
    df = encode_df(df, cat_encoder_path=cat_encoder_path)
    logging.info(f"Categorical encoding completed. Columns: {df.columns}")

    # Create time-series features
    logging.info("Creating time-series features.")
    ts_df = create_ts_features(df, imp_cols_path=imp_cols_path, ts_feat_imp_path=ts_feat_imp_path)
    logging.info("Time-series features created successfully.")

    # Create TSFEL features
    logging.info("Creating TSFEL features.")
    tsfel_df = create_tsfel_features(
        df,
        imp_cols_path=imp_cols_path,
        stat_select_path=stat_select_path,
        temporal_select_path=temporal_select_path
    )
    logging.info("TSFEL features created successfully.")

    # Combine all dataframes
    logging.info("Combining all dataframes.")
    final_df = combine_dfs(df=df, ts_df=ts_df, tsfel_df=tsfel_df, ref_cols_path=ref_cols_path)
    logging.info("Dataframes combined successfully.")

    # Load the model
    logging.info(f"Loading model from {model_path}.")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logging.info("Model loaded successfully.")

    # Generate predictions and probabilities
    logging.info("Generating predictions and probabilities.")
    probabilities = model.predict_proba(final_df.drop(columns=['customer_ID']))
    predictions = ['default' if prob[1] >= threshold else 'non-default' for prob in probabilities]
    logging.info("Predictions and probabilities generated successfully.")

    # Load training data for SHAP
    logging.info(f"Loading training data from {train_data_path} for SHAP analysis.")
    train_df = pd.read_parquet(train_data_path)

    # Initialize SHAP explainer
    logging.info("Initializing SHAP explainer.")
    explainer = shap.Explainer(model.predict_proba, train_df.drop(columns=['customer_ID']), model_output='probability', seed=0)

    # Compute SHAP values
    logging.info("Computing SHAP values.")
    shap_values = explainer(final_df.drop(columns=['customer_ID']), max_evals=844)
    logging.info("SHAP values computed successfully.")

    # Save SHAP waterfall plot for the first prediction
    waterfall_plot_path = os.path.join(output_dir, 'shap_waterfall_plot.png')
    logging.info(f"Saving SHAP waterfall plot to {waterfall_plot_path}.")
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[..., 1][0], show=False)
    plt.savefig(waterfall_plot_path, bbox_inches='tight')
    plt.close()
    logging.info("SHAP waterfall plot saved successfully.")

    # Save SHAP force plot as HTML
    force_plot_path = os.path.join(output_dir, 'shap_force_plot.html')
    logging.info(f"Saving SHAP force plot to {force_plot_path}.")
    shap.save_html(force_plot_path, shap.plots.force(shap_values[..., 1][0], matplotlib=False))
    logging.info("SHAP force plot saved successfully.")

    # Return predictions, probabilities, and SHAP plot paths
    shap_plots = {
        'waterfall_plot': waterfall_plot_path
    }

    logging.info("Pipeline execution completed successfully.")
    return predictions, probabilities, shap_plots