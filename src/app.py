import streamlit as st
import pandas as pd
from pipeline import run_pipeline

def main():
    st.title("Data Processing and Prediction App")
    
    st.sidebar.header("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["parquet", "csv"])
    
    if uploaded_file is not None:
        # Load the dataset
        if uploaded_file.name.endswith('.parquet'):
            df = pd.read_parquet(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        st.write("Data Preview:")
        st.dataframe(df.head())
        
        threshold = st.sidebar.slider("Prediction Threshold", min_value=0.0, max_value=1.0, value=0.9094, step=0.01)
        
        if st.sidebar.button("Run Pipeline"):
            with st.spinner("Processing..."):
                predictions, probabilities, shap_plots = run_pipeline(
                    data_path=uploaded_file,
                    non_null_cols_path='./utils/selected_features/non_null_cols_filter.csv',
                    cat_encoder_path='./utils/encoders/categorical_encoder.pkl',
                    ts_feat_imp_path='./utils/selected_features/ts_features_boruta.csv',
                    imp_cols_path='./utils/selected_features/boruta_features_baseline.csv',
                    stat_select_path='./utils/selected_features/tsfel_stat_features_boruta.csv',
                    temporal_select_path='./utils/selected_features/tsfel_temporal_features_boruta.csv',
                    ref_cols_path='./data/final_features_ml/val_all_features_last_month.parquet',
                    model_path='./models/machine_learning/eec_xgb_tuned.pkl',
                    train_data_path='./data/final_features_ml/train_all_features_last_month.parquet',
                    output_dir='./plots/',
                    threshold=threshold
                )
                
                st.success("Processing complete!")
                st.write("Predictions:")
                st.dataframe(predictions)
                
                st.write("Probabilities:")
                probabilities_df = pd.DataFrame(probabilities, columns=['Non-Default Probability', 'Default Probability'])
                st.dataframe(probabilities_df)
                
                # Display SHAP waterfall plot
                st.write("SHAP Waterfall Plot:")
                st.image(shap_plots['waterfall_plot'], caption="SHAP Waterfall Plot", use_column_width=True)

if __name__ == "__main__":
    main()