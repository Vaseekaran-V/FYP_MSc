from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

def one_hot_encode_categories(df, categorical_columns=None, drop_original=True, handle_unknown='error'):
    """
    One-hot encode categorical columns in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing categorical columns
    categorical_columns : list or None
        List of categorical column names to encode. If None, automatically detects categorical columns.
    drop_original : bool
        Whether to drop the original categorical columns
    handle_unknown : str
        Strategy for handling unknown categories in new data: 'error', 'ignore' or 'infrequent_if_exist'
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with one-hot encoded columns
    OneHotEncoder
        Fitted encoder for future transformations
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Automatically detect categorical columns if not specified
    if categorical_columns is None:
        categorical_columns = result_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_columns:
        print("No categorical columns found to encode.")
        return result_df, None
        
    # Initialize the encoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown=handle_unknown, drop='if_binary')
    
    # Fit and transform the categorical columns
    encoded_array = encoder.fit_transform(result_df[categorical_columns])
    
    # Get feature names
    feature_names = encoder.get_feature_names_out(categorical_columns)
    
    # Create a DataFrame with the encoded features
    encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=result_df.index)
    
    # Combine with the original DataFrame
    if drop_original:
        # Drop the original categorical columns
        result_df = result_df.drop(columns=categorical_columns)
    
    # Concatenate the encoded columns with the original DataFrame
    result_df = pd.concat([result_df, encoded_df], axis=1)
    
    print(f"One-hot encoded {len(categorical_columns)} categorical columns into {len(feature_names)} binary features.")
    
    return result_df, encoder