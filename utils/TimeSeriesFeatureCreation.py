import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
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
        '''Creates lag features for a given feature.'''
        if self.verbose:
            print(f'\tCreating lag features for {feature_name}')
        # Group once outside the loop
        grouped_data = df.groupby(self.id_col)[feature_name]

        # Calculate lag 1 through num_lags
        # Calling .shift() repeatedly is generally efficient in pandas
        for i in range(1, self.num_lags + 1):
            df[f'lag_{feature_name}_{i}'] = grouped_data.shift(i)
        return df
    
    def _groupby_df(self, df):
        '''Groups the dataframe by the given feature list.'''
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df_gb = df.groupby([self.date_col, self.id_col])[[self.features]].sum().reset_index()
        return df_gb
        
    def _create_rolling_window_features_optimized(self, df, feature_name):
        """
        Creates rolling window features for a given feature (optimized version).

        Args:
            df (pd.DataFrame): The input DataFrame.
            feature_name (str): The name of the column to create rolling features for.

        Returns:
            pd.DataFrame: The DataFrame with added rolling window features.
        """
        if not self.return_rolling:
            return df # Return early if rolling features are not requested

        if self.verbose:
            print(f'\tCreating rolling window features for {feature_name}')

        # Group by the identifier column
        grouped_data = df.groupby(self.id_col)[feature_name]

        # Calculate rolling mean if requested
        if self.use_non_lag:
            mean_col_name = f'rolling_mean_{feature_name}_{self.rolling_window_size}'
            # Apply rolling mean directly on the grouped series
            # Using min_periods=1 ensures that the calculation happens even if the window is not full
            # (e.g., at the beginning of each group)
            df[mean_col_name] = grouped_data.rolling(
                window=self.rolling_window_size, min_periods=1
            ).mean().reset_index(level=0, drop=True) # reset_index aligns result back to original df index

            # Calculate rolling standard deviation if requested
            if self.std_dev:
                std_col_name = f'rolling_std_{feature_name}_{self.rolling_window_size}'
                # Apply rolling std directly on the grouped series
                df[std_col_name] = grouped_data.rolling(
                    window=self.rolling_window_size, min_periods=1 # min_periods=1 similar to mean
                ).std().reset_index(level=0, drop=True) # reset_index aligns result back to original df index

        return df

    def _create_cumulative_features(self, df, feature_name):
        '''Creates cumulative features for a given feature.'''
        if self.use_non_lag:
            if self.verbose:
                print(f'\tCreating cumulative features for {feature_name}')
            # Group once outside the conditions
            grouped_data = df.groupby(self.id_col)[feature_name]

            if self.cum_sum:
                df[f'cumsum_{feature_name}'] = grouped_data.cumsum()

            if self.cum_mean:
                # Calculate cumsum and count
                cumulative_sum = grouped_data.cumsum()
                # cumcount starts at 0, add 1 for mean calculation (count of elements up to current row)
                cumulative_count = grouped_data.cumcount() + 1
                df[f'cummean_{feature_name}'] = cumulative_sum / cumulative_count
                # Optional: Handle potential division by zero if the first element could be NaN/missing
                # df[f'cummean_{feature_name}'] = df[f'cummean_{feature_name}'].fillna(0) # Or some other appropriate value

        return df

    def _create_expanding_window_features(self, df, feature_name):
        '''Creates expanding window features for a given feature.'''
        if self.verbose:
            print(f'\tCreating expanding window features for {feature_name}')
        if self.use_non_lag:
            grouped_data = df.groupby(self.id_col)[feature_name]

            if self.return_min:
                df[f'expanding_min_{feature_name}'] = grouped_data.cummin()

            if self.return_max:
                df[f'expanding_max_{feature_name}'] = grouped_data.cummax()
        return df

    def _create_time_differencing(self, df, feature_name):
        if self.verbose:
            print(f'\tCreating time differencing features for {feature_name}')
        '''Creates difference features for a given feature.'''
        if self.use_non_lag:
            if self.create_2_diff:
                df[f'diff_{feature_name}'] = df.groupby([self.id_col])[f'{feature_name}'].diff()
                df[f'diff_{feature_name}_{self.time_window_size}'] = (
                    df.groupby([self.id_col])[f'{feature_name}'].diff(self.time_window_size)
                )
            else:
                for i in range(self.time_window_size):
                    df[f'diff_{feature_name}_{i+1}'] = df.groupby([self.id_col])[f'{feature_name}'].diff(i+1)
        return df
    
    def _create_time_differencing_optimized(self, df, feature_name):
        """
        Creates time difference features for a given feature (optimized version).

        Args:
            df (pd.DataFrame): The input DataFrame. Assumed to be sorted by id_col and time/sequence.
            feature_name (str): The name of the column to create difference features for.

        Returns:
            pd.DataFrame: The DataFrame with added difference features.
        """
        if not self.return_diff:
             # Check flags before proceeding
            return df

        if self.verbose:
            print(f'\tCreating time differencing features for {feature_name}')

        # Group once outside the conditions/loop
        grouped_data = df.groupby(self.id_col)[feature_name]

        if self.create_2_diff:
            # Calculate only diff 1 and diff N
            df[f'diff_{feature_name}_1'] = grouped_data.diff(1)
            if self.time_window_size > 1: # Avoid creating diff_1 twice if time_window_size is 1
                 df[f'diff_{feature_name}_{self.time_window_size}'] = grouped_data.diff(self.time_window_size)
        else:
            # Calculate diff 1 through N
            # Calling .diff() repeatedly is generally efficient in pandas
            for i in range(1, self.time_window_size + 1):
                df[f'diff_{feature_name}_{i}'] = grouped_data.diff(i)

        return df

    @staticmethod
    def _calculate_trend(x):
        """
        Calculates the trend of a pandas Series.
        """
        x = x.dropna()  # Drop null values
        if len(x) < 2:
            return np.nan  # Not enough data to calculate trend
        X = np.arange(len(x))
        y = x.values
        slope, _ = np.polyfit(X, y, 1)
        return slope

    def _create_trend_features_rolling(self, df, feature_name):
        """
        Create trend features for a given feature.
        """
        trend_feature_name = f'trend_{feature_name}_rolling'
        df[trend_feature_name] = np.nan

        def process_group(group):
            group = group.sort_values(by=self.date_col)
            trend_values = group[f'{feature_name}'].rolling(window=len(group), min_periods=2).apply(
                lambda x: self._calculate_trend(x), raw=False)
            return group.index, trend_values

        results = Parallel(n_jobs=-1)(delayed(process_group)(group) for key, group in df.groupby([self.id_col]))

        for index, trend_values in results:
            df.loc[index, trend_feature_name] = trend_values

        return df
    
    def _create_ewm_features_optimized(self, df, feature_name):
        """
        Create exponentially weighted moving average features (optimized version).

        Args:
            df (pd.DataFrame): The input DataFrame. Assumed to be sorted by id_col and time/sequence.
            feature_name (str): The name of the column to create EWM features for.

        Returns:
            pd.DataFrame: The DataFrame with added EWM features.
        """
        if not self.return_ewm:
            return df # Return early if EWM features are not requested

        if self.verbose:
            print(f'\tCreating ewm features for {feature_name} with span {self.span}')

        ewm_feature_name = f'ewm_{feature_name}_{self.span}'
        reverse_ewm_feature_name = f'reverse_ewm_{feature_name}_{self.span}'

        # Group by the identifier column
        grouped_data = df.groupby(self.id_col)[feature_name]

        # Calculate forward EWM directly on the grouped series
        df[ewm_feature_name] = grouped_data.ewm(
            span=self.span, min_periods=3, adjust=True # adjust=True is default
        ).mean().reset_index(level=0, drop=True) # reset_index aligns result back

        # Calculate reverse EWM using transform to handle the reversal within each group
        # This is often clearer than trying to reverse the entire grouped object externally
        df[reverse_ewm_feature_name] = grouped_data.transform(
            lambda x: x[::-1].ewm(span=self.span, min_periods=3).mean()[::-1]
        )

        return df

    
    def _preprocess_dataframe(self, df):
        '''Applies all preprocessing steps to a dataframe.'''
        
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        if self.features is None:
            cols = df.drop(columns = [self.id_col, self.date_col]).columns
            self.features = cols.tolist()
        df_gb = df.groupby([self.date_col, self.id_col])[self.features].sum().reset_index()
        print(type(df_gb))

        for feature_name in tqdm.tqdm(self.features, desc = 'Creating time series features for each feature', total = len(df_gb.columns)):
            if self.verbose:
                print(f'Creating features for {feature_name}')
            df_gb = self._create_lag_features(df_gb, feature_name)
            df_gb = self._create_cumulative_features(df_gb, feature_name)
            df_gb = self._create_expanding_window_features(df_gb, feature_name)
            df_gb = self._create_time_differencing_optimized(df_gb, feature_name)
            # # df_gb = self._create_trend_features(df_gb, feature_name)
            # df_gb = self._create_trend_features_rolling(df_gb, feature_name)
            df_gb = self._create_rolling_window_features_optimized(df_gb, feature_name)
            df_gb = self._create_ewm_features_optimized(df_gb, feature_name)

        # df_gb = self._create_time_based_features(df_gb)
        # df_gb = self._encode_categorical_columns(df_gb, categorical_columns = self.categorical_columns)
        if self.drop_null_rows:
            df_gb = df_gb.dropna(axis=0)
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
