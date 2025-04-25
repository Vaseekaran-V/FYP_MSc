import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
from pandarallel import pandarallel
import tqdm

class TimeSeriesFeatureCreation:
    '''
    A class to create time series features for a given dataframe.
    
    This class is designed to create various time series features such as lag features, rolling window features,
    cumulative features, expanding window features, time differencing features, and exponentially weighted moving average features.

    It also provides options to customize the feature creation process based on user preferences.

    Attributes:

        id_col (str): The column name representing the unique identifier for each time series.
        date_col (str): The column name representing the date or time information.
        num_lags (int): The number of lag features to create.
        rolling_window_size (int): The size of the rolling window for rolling features.
        std_dev (bool): Whether to calculate the standard deviation for rolling features.
        use_non_lag (bool): Whether to use non-lag features in addition to lag features.
        cum_mean (bool): Whether to calculate cumulative mean features.
        cum_sum (bool): Whether to calculate cumulative sum features.
        return_rolling (bool): Whether to calculate rolling features.
        return_min (bool): Whether to calculate minimum features for expanding windows.
        return_max (bool): Whether to calculate maximum features for expanding windows.
        return_diff (bool): Whether to calculate differencing features.
        return_ewm (bool): Whether to calculate exponentially weighted moving average features.
        time_window_size (int): The size of the time window for differencing features.
        drop_null_rows (bool): Whether to drop rows with null values after feature creation.
        features (list): The list of feature names to create features for.
        create_2_diff (bool): Whether to create two differencing features.
        verbose (bool): Whether to print verbose output during feature creation.
        alpha (float): The smoothing factor for exponentially weighted moving average features.
        span (int): The span for exponentially weighted moving average features.
    
    '''

    def __init__(self, id_col = 'customer_ID',date_col='end_of_month', num_lags=1, rolling_window_size=2,
                 std_dev=True, use_non_lag=True, cum_mean=True, cum_sum=True, return_rolling = True,
                 return_min=True, return_max=True, return_diff = True, return_ewm = True,
                 time_window_size=10, drop_null_rows = False, features = None, create_2_diff = True, verbose = False,
                 alpha = 0.1, span = 10):
        
        self.id_col = id_col
        self.date_col = date_col
        self.num_lags = num_lags
        self.rolling_window_size = rolling_window_size
        self.std_dev = std_dev
        self.use_non_lag = use_non_lag
        self.cum_mean = cum_mean
        self.cum_sum = cum_sum
        self.return_rolling = return_rolling
        # self.years = years
        self.return_min = return_min
        self.return_max = return_max
        self.return_diff = return_diff
        self.return_ewm = return_ewm
        self.time_window_size = time_window_size
        self.drop_null_rows = drop_null_rows
        self.features = features
        self.create_2_diff = create_2_diff
        self.verbose = verbose
        self.alpha = alpha
        self.span = span

    def _create_lag_features(self, df, feature_name):
        '''Creates lag features for a given feature and returns them in a new DataFrame.'''
        new_features_df = pd.DataFrame(index=df.index) # Ensure index alignment
        if self.verbose:
            print(f'\tCreating lag features for {feature_name}')
        grouped_data = df.groupby(self.id_col)[feature_name]
        for i in range(1, self.num_lags + 1):
            new_features_df[f'lag_{feature_name}_{i}'] = grouped_data.shift(i)
        return new_features_df
        
    def _create_rolling_window_features_optimized(self, df, feature_name):
        """
        Creates rolling window features (optimized) and returns them in a new DataFrame.
        """
        new_features_df = pd.DataFrame(index=df.index) # Ensure index alignment
        if not self.return_rolling:
            return new_features_df # Return empty DataFrame if not requested

        if self.verbose:
            print(f'\tCreating rolling window features for {feature_name}')

        grouped_data = df.groupby(self.id_col)[feature_name]

        if self.use_non_lag:
            mean_col_name = f'rolling_mean_{feature_name}_{self.rolling_window_size}'
            new_features_df[mean_col_name] = grouped_data.rolling(
                window=self.rolling_window_size, min_periods=1
            ).mean().reset_index(level=0, drop=True)

            if self.std_dev:
                std_col_name = f'rolling_std_{feature_name}_{self.rolling_window_size}'
                new_features_df[std_col_name] = grouped_data.rolling(
                    window=self.rolling_window_size, min_periods=1
                ).std().reset_index(level=0, drop=True)

        return new_features_df

    def _create_cumulative_features(self, df, feature_name):
        '''Creates cumulative features and returns them in a new DataFrame.'''
        new_features_df = pd.DataFrame(index=df.index) # Ensure index alignment
        if self.use_non_lag:
            if self.verbose:
                print(f'\tCreating cumulative features for {feature_name}')
            grouped_data = df.groupby(self.id_col)[feature_name]

            if self.cum_sum:
                new_features_df[f'cumsum_{feature_name}'] = grouped_data.cumsum()

            if self.cum_mean:
                cumulative_sum = grouped_data.cumsum()
                cumulative_count = grouped_data.cumcount() + 1
                new_features_df[f'cummean_{feature_name}'] = cumulative_sum / cumulative_count
                # Optional: Handle potential division by zero
                # new_features_df[f'cummean_{feature_name}'] = new_features_df[f'cummean_{feature_name}'].fillna(0)

        return new_features_df

    def _create_expanding_window_features(self, df, feature_name):
        '''Creates expanding window features and returns them in a new DataFrame.'''
        new_features_df = pd.DataFrame(index=df.index) # Ensure index alignment
        if self.verbose:
            print(f'\tCreating expanding window features for {feature_name}')
        if self.use_non_lag:
            grouped_data = df.groupby(self.id_col)[feature_name]

            if self.return_min:
                new_features_df[f'expanding_min_{feature_name}'] = grouped_data.cummin()

            if self.return_max:
                new_features_df[f'expanding_max_{feature_name}'] = grouped_data.cummax()
        return new_features_df
    
    def _create_time_differencing_optimized(self, df, feature_name):
        """
        Creates time difference features (optimized) and returns them in a new DataFrame.
        """
        new_features_df = pd.DataFrame(index=df.index) # Ensure index alignment
        if not self.return_diff:
             return new_features_df # Return empty DataFrame if not requested

        if self.verbose:
            print(f'\tCreating time differencing features for {feature_name}')

        grouped_data = df.groupby(self.id_col)[feature_name]

        if self.create_2_diff:
            new_features_df[f'diff_{feature_name}_1'] = grouped_data.diff(1)
            if self.time_window_size > 1:
                 new_features_df[f'diff_{feature_name}_{self.time_window_size}'] = grouped_data.diff(self.time_window_size)
        else:
            for i in range(1, self.time_window_size + 1):
                new_features_df[f'diff_{feature_name}_{i}'] = grouped_data.diff(i)

        return new_features_df
    
    def _create_ewm_features_optimized(self, df, feature_name):
        """
        Create EWM features (optimized) and returns them in a new DataFrame.
        """
        new_features_df = pd.DataFrame(index=df.index) # Ensure index alignment
        if not self.return_ewm:
            return new_features_df # Return empty DataFrame if not requested

        if self.verbose:
            print(f'\tCreating ewm features for {feature_name} with span {self.span}')

        # Ensure float type for EWM calculation
        feature_series = df[feature_name].astype('float32')

        ewm_feature_name = f'ewm_{feature_name}_{self.span}'
        reverse_ewm_feature_name = f'reverse_ewm_{feature_name}_{self.span}'

        # Group by the identifier column on the original feature series
        grouped_data = feature_series.groupby(df[self.id_col]) # Group the Series using original DF's ID

        # Calculate forward EWM
        new_features_df[ewm_feature_name] = grouped_data.ewm(
            span=self.span, min_periods=1, adjust=True
        ).mean().reset_index(level=0, drop=True)

        # Calculate reverse EWM using transform
        new_features_df[reverse_ewm_feature_name] = grouped_data.transform(
            lambda x: x[::-1].ewm(span=self.span, min_periods=1).mean()[::-1]
        )

        return new_features_df

    
    def _preprocess_dataframe(self, df):
        '''Applies all preprocessing steps to a dataframe using optimized concatenation.'''

        df_original = df.copy() # Keep original untouched until the end
        df_original[self.date_col] = pd.to_datetime(df_original[self.date_col])

        if self.features is None:
            cols = df_original.drop(columns = [self.id_col, self.date_col]).columns
            self.features = cols.tolist()

        # Aggregate the initial DataFrame
        df_gb = df_original.groupby([self.date_col, self.id_col])[self.features].sum().reset_index()
        print(f"Aggregated DataFrame type: {type(df_gb)}")

        # List to hold DataFrames of new features
        all_new_features_dfs = []

        # Sort df_gb ensures consistent order for group operations (important for rolling, diff, etc.)
        # Use sort_values instead of relying on groupby order if necessary, although groupby usually sorts.
        df_gb = df_gb.sort_values(by=[self.id_col, self.date_col]).reset_index(drop=True)


        for feature_name in tqdm.tqdm(self.features, desc = 'Creating time series features for each feature', total = len(self.features)): # Use self.features here
            if self.verbose:
                print(f'Creating features for {feature_name}')

            # Collect new features from each method
            all_new_features_dfs.append(self._create_lag_features(df_gb, feature_name))
            all_new_features_dfs.append(self._create_cumulative_features(df_gb, feature_name))
            all_new_features_dfs.append(self._create_expanding_window_features(df_gb, feature_name))
            all_new_features_dfs.append(self._create_time_differencing_optimized(df_gb, feature_name))
            # all_new_features_dfs.append(self._create_trend_features_rolling(df_gb.copy(), feature_name)) # Pass a copy to avoid modifying df_gb in place if method isn't fully refactored
            all_new_features_dfs.append(self._create_rolling_window_features_optimized(df_gb, feature_name))
            all_new_features_dfs.append(self._create_ewm_features_optimized(df_gb, feature_name))


        # Concatenate all new features at once
        print(f"Concatenating {len(all_new_features_dfs)} feature DataFrames.")
        df_gb = pd.concat([df_gb] + all_new_features_dfs, axis=1)

        # Drop rows with nulls if requested
        if self.drop_null_rows:
            print(f"Shape before dropping NaNs: {df_gb.shape}")
            df_gb = df_gb.dropna(axis=0)
            print(f"Shape after dropping NaNs: {df_gb.shape}")

        return df_gb

    def fit(self, X, y=None):
        '''
        Fits the preprocessor to the training data.

        In this case, fit does nothing as we are calculating rolling, expanding, 
        and lag features based on the data itself. We will keep this method for 
        consistency with the sklearn API.

        Args:
            X (pd.DataFrame): The training input samples.
            y (pd.Series, optional): The target values. Defaults to None.

        Returns:
            self: Returns the instance itself.
        '''
        return self

    def transform(self, X):
        '''
        Applies the preprocessing steps to a dataframe.

        Args:
            X (pd.DataFrame): The input dataframe to transform.

        Returns:
            pd.DataFrame: The transformed dataframe.
        '''
        return self._preprocess_dataframe(X)

    def fit_transform(self, X, y=None):
        '''
        Fits to the data and then transforms it.

        Args:
            X (pd.DataFrame): The training input samples.
            y (pd.Series, optional): The target values. Defaults to None.

        Returns:
            pd.DataFrame: The transformed dataframe.
        '''
        return self.fit(X, y).transform(X)
