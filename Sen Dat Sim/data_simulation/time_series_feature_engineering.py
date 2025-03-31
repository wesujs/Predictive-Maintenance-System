import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import pywt

class TimeSeriesFeatureEngineering:
    def __init__(self, df=None, timestamp_col='timestamp'):
        """
        Initialize the Time Series Feature Engineering pipeline.
        
        Parameters:
        -----------
        df : pandas.DataFrame, optional
            DataFrame containing sensor data with timestamps
        timestamp_col : str, default='timestamp'
            Name of the timestamp column in the DataFrame
        """
        self.df = df
        self.timestamp_col = timestamp_col
        self.features_df = None
        self.scalers = {}
        
    def load_data(self, df):
        """
        Load data into the pipeline.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing sensor data with timestamps
        """
        self.df = df.copy()
        # Ensure timestamp column is datetime type
        if self.timestamp_col in self.df.columns:
            self.df[self.timestamp_col] = pd.to_datetime(self.df[self.timestamp_col])
            # Set timestamp as index for time series operations
            self.df = self.df.set_index(self.timestamp_col)
        
        return self
        
    def load_data_from_simulator(self, simulator):
        """
        Load data directly from a SensorDataSimulator instance.
        
        Parameters:
        -----------
        simulator : SensorDataSimulator
            Initialized simulator with generated data
        """
        df = simulator.to_dataframe()
        return self.load_data(df)
    
    def generate_rolling_statistics(self, columns=None, windows=[5, 10, 30, 60], include_original=True):
        """
        Calculate rolling statistics for specified columns.
        
        Parameters:
        -----------
        columns : list or None
            List of column names to process. If None, all numeric columns will be processed.
        windows : list
            List of window sizes for rolling calculations (in number of observations)
        include_original : bool
            Whether to include original columns in the output
        
        Returns:
        --------
        self : for method chaining
        """
        if self.df is None:
            raise ValueError("No data loaded. Use load_data method first.")
            
        # If columns not specified, use all numeric columns
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        # Initialize features DataFrame
        if self.features_df is None:
            if include_original:
                self.features_df = self.df[columns].copy()
            else:
                self.features_df = pd.DataFrame(index=self.df.index)
        
        # Generate rolling statistics for each column and window size
        for col in columns:
            for window in windows:
                # Rolling mean
                self.features_df[f"{col}_roll_mean_{window}"] = self.df[col].rolling(window=window).mean()
                
                # Rolling standard deviation
                self.features_df[f"{col}_roll_std_{window}"] = self.df[col].rolling(window=window).std()
                
                # Rolling min/max
                self.features_df[f"{col}_roll_min_{window}"] = self.df[col].rolling(window=window).min()
                self.features_df[f"{col}_roll_max_{window}"] = self.df[col].rolling(window=window).max()
                
                # Rolling median (more robust to outliers)
                self.features_df[f"{col}_roll_median_{window}"] = self.df[col].rolling(window=window).median()
                
                # Rolling skewness and kurtosis (distribution shape)
                self.features_df[f"{col}_roll_skew_{window}"] = self.df[col].rolling(window=window).skew()
                self.features_df[f"{col}_roll_kurt_{window}"] = self.df[col].rolling(window=window).kurt()
                
                # Rolling quantiles
                self.features_df[f"{col}_roll_q25_{window}"] = self.df[col].rolling(window=window).quantile(0.25)
                self.features_df[f"{col}_roll_q75_{window}"] = self.df[col].rolling(window=window).quantile(0.75)
                
                # IQR (Interquartile range) - good for detecting outliers
                q75 = self.df[col].rolling(window=window).quantile(0.75)
                q25 = self.df[col].rolling(window=window).quantile(0.25)
                self.features_df[f"{col}_roll_iqr_{window}"] = q75 - q25
        
        return self
    
    def generate_lag_features(self, columns=None, lags=[1, 5, 10, 30], include_original=True):
        """
        Create lagged versions of the columns.
        
        Parameters:
        -----------
        columns : list or None
            List of column names to process. If None, all numeric columns will be processed.
        lags : list
            List of lag values (in number of observations)
        include_original : bool
            Whether to include original columns in the output
        
        Returns:
        --------
        self : for method chaining
        """
        if self.df is None:
            raise ValueError("No data loaded. Use load_data method first.")
            
        # If columns not specified, use all numeric columns
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        # Initialize features DataFrame if needed
        if self.features_df is None:
            if include_original:
                self.features_df = self.df[columns].copy()
            else:
                self.features_df = pd.DataFrame(index=self.df.index)
        elif include_original and not all(col in self.features_df.columns for col in columns):
            for col in columns:
                if col not in self.features_df.columns:
                    self.features_df[col] = self.df[col]
        
        # Generate lag features
        for col in columns:
            for lag in lags:
                self.features_df[f"{col}_lag_{lag}"] = self.df[col].shift(lag)
                
                # Also add rate of change (percent change)
                self.features_df[f"{col}_pct_change_{lag}"] = self.df[col].pct_change(periods=lag)
                
                # Difference (absolute change)
                self.features_df[f"{col}_diff_{lag}"] = self.df[col].diff(periods=lag)
        
        return self
    
    def extract_frequency_domain_features(self, columns=None, include_original=True):
        """
        Extract frequency domain features using Fast Fourier Transform.
        
        Parameters:
        -----------
        columns : list or None
            List of column names to process. If None, all numeric columns will be processed.
        include_original : bool
            Whether to include original columns in the output
            
        Returns:
        --------
        self : for method chaining
        """
        if self.df is None:
            raise ValueError("No data loaded. Use load_data method first.")
            
        # If columns not specified, use all numeric columns
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        # Initialize features DataFrame if needed
        if self.features_df is None:
            if include_original:
                self.features_df = self.df[columns].copy()
            else:
                self.features_df = pd.DataFrame(index=self.df.index)
        
        # Process each column
        for col in columns:
            # Get the data
            data = self.df[col].values
            
            # Handle NaN values
            data = np.nan_to_num(data)
            
            # Apply FFT
            fft_result = np.fft.rfft(data)
            fft_freq = np.fft.rfftfreq(len(data))
            
            # Calculate magnitude spectrum
            magnitude = np.abs(fft_result)
            power = magnitude ** 2
            
            # Extract features from frequency domain
            # Top 5 frequencies by magnitude
            top_indices = np.argsort(magnitude)[-5:]
            top_freqs = fft_freq[top_indices]
            
            # Add features to DataFrame
            # Dominant frequency
            self.features_df[f"{col}_dominant_freq"] = np.repeat(top_freqs[-1], len(self.df))
            
            # Power at dominant frequency
            self.features_df[f"{col}_dominant_power"] = np.repeat(power[top_indices[-1]], len(self.df))
            
            # Spectral entropy (measure of complexity)
            normalized_power = power / np.sum(power)
            spectral_entropy = -np.sum(normalized_power * np.log2(normalized_power + 1e-10))
            self.features_df[f"{col}_spectral_entropy"] = np.repeat(spectral_entropy, len(self.df))
            
            # Spectral centroid (weighted mean of frequencies)
            spectral_centroid = np.sum(fft_freq * normalized_power) / np.sum(normalized_power)
            self.features_df[f"{col}_spectral_centroid"] = np.repeat(spectral_centroid, len(self.df))
            
            # Energy in different frequency bands
            # Low frequency band (0-25% of spectrum)
            low_mask = fft_freq <= np.max(fft_freq) * 0.25
            self.features_df[f"{col}_low_freq_energy"] = np.repeat(np.sum(power[low_mask]), len(self.df))
            
            # Mid frequency band (25-75% of spectrum)
            mid_mask = (fft_freq > np.max(fft_freq) * 0.25) & (fft_freq <= np.max(fft_freq) * 0.75)
            self.features_df[f"{col}_mid_freq_energy"] = np.repeat(np.sum(power[mid_mask]), len(self.df))
            
            # High frequency band (75-100% of spectrum)
            high_mask = fft_freq > np.max(fft_freq) * 0.75
            self.features_df[f"{col}_high_freq_energy"] = np.repeat(np.sum(power[high_mask]), len(self.df))
            
        return self
    
    def decompose_trend_seasonality(self, columns=None, period=1440, model='additive', include_original=True):
        """
        Decompose time series into trend, seasonality, and residual components.
        Default period is 1440 (minutes in a day) assuming minute-level data.
        
        Parameters:
        -----------
        columns : list or None
            List of column names to process. If None, all numeric columns will be processed.
        period : int
            Number of time points in a seasonal cycle
        model : str
            Type of decomposition ('additive' or 'multiplicative')
        include_original : bool
            Whether to include original columns in the output
            
        Returns:
        --------
        self : for method chaining
        """
        if self.df is None:
            raise ValueError("No data loaded. Use load_data method first.")
            
        # If columns not specified, use all numeric columns
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        # Initialize features DataFrame if needed
        if self.features_df is None:
            if include_original:
                self.features_df = self.df[columns].copy()
            else:
                self.features_df = pd.DataFrame(index=self.df.index)
        
        # Process each column
        for col in columns:
            try:
                # Handle missing values with interpolation
                series = self.df[col].interpolate()
                
                # Decompose the time series
                result = seasonal_decompose(series, model=model, period=period)
                
                # Add components to features
                self.features_df[f"{col}_trend"] = result.trend
                self.features_df[f"{col}_seasonal"] = result.seasonal
                self.features_df[f"{col}_residual"] = result.resid
                
            except Exception as e:
                print(f"Warning: Could not decompose {col}: {e}")
                # If decomposition fails, add NaN columns to maintain structure
                self.features_df[f"{col}_trend"] = np.nan
                self.features_df[f"{col}_seasonal"] = np.nan
                self.features_df[f"{col}_residual"] = np.nan
        
        return self
    
    def extract_wavelet_features(self, columns=None, wavelet='db4', level=3, include_original=True):
        """
        Extract wavelet transform features.
        
        Parameters:
        -----------
        columns : list or None
            List of column names to process. If None, all numeric columns will be processed.
        wavelet : str
            Wavelet type ('db4' is Daubechies 4, a common choice)
        level : int
            Decomposition level
        include_original : bool
            Whether to include original columns in the output
            
        Returns:
        --------
        self : for method chaining
        """
        if self.df is None:
            raise ValueError("No data loaded. Use load_data method first.")
            
        # If columns not specified, use all numeric columns
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        # Initialize features DataFrame if needed
        if self.features_df is None:
            if include_original:
                self.features_df = self.df[columns].copy()
            else:
                self.features_df = pd.DataFrame(index=self.df.index)
        
        try:
            # Create a dictionary to collect all wavelet features
            all_features = {}
            
            # Process each column
            for col in columns:
                # Get the data
                data = self.df[col].interpolate().values
                
                # Skip columns with insufficient data
                if len(data) < 2**level:
                    print(f"Warning: Not enough data points for wavelet decomposition on {col}")
                    continue
                    
                # Pad data to power of 2 if needed
                pad_len = int(2**np.ceil(np.log2(len(data))))
                padded_data = np.pad(data, (0, pad_len - len(data)), 'constant', constant_values=(0))
                
                # Perform wavelet decomposition
                coeffs = pywt.wavedec(padded_data, wavelet, level=level)
                
                # Extract features from each level
                for i, coeff in enumerate(coeffs):
                    if i == 0:
                        name = "approx"  # Approximation coefficients
                    else:
                        name = f"detail_{i}"  # Detail coefficients
                    
                    # Calculate statistics of wavelet coefficients
                    coeff_energy = np.sum(coeff**2)
                    coeff_mean = np.mean(coeff)
                    coeff_std = np.std(coeff)
                    coeff_max = np.max(np.abs(coeff))
                    
                    # Instead of repeating, create a single value for these global statistics
                    all_features[f"{col}_wavelet_{name}_energy"] = [coeff_energy] * len(self.df.index)
                    all_features[f"{col}_wavelet_{name}_mean"] = [coeff_mean] * len(self.df.index)
                    all_features[f"{col}_wavelet_{name}_std"] = [coeff_std] * len(self.df.index)
                    all_features[f"{col}_wavelet_{name}_max"] = [coeff_max] * len(self.df.index)
            
            # Create a DataFrame with the correct index
            wavelet_df = pd.DataFrame(all_features, index=self.df.index)
            
            # Merge with features_df
            for col in wavelet_df.columns:
                self.features_df[col] = wavelet_df[col]
            
        except Exception as e:
            print(f"Could not extract wavelet features: {e}")
            import traceback
            traceback.print_exc()
        
        return self
    
    def add_time_features(self, include_original=True):
        """
        Add time-based features derived from timestamp index.
        
        Parameters:
        -----------
        include_original : bool
            Whether to include original columns in the output
            
        Returns:
        --------
        self : for method chaining
        """
        if self.df is None:
            raise ValueError("No data loaded. Use load_data method first.")
        
        # Initialize features DataFrame if needed
        if self.features_df is None:
            if include_original:
                self.features_df = self.df.copy()
            else:
                self.features_df = pd.DataFrame(index=self.df.index)
        
        # Get timestamp as Series for feature extraction
        timestamp = pd.Series(self.features_df.index)
        
        # Extract various time components
        self.features_df['hour_of_day'] = timestamp.dt.hour
        self.features_df['day_of_week'] = timestamp.dt.dayofweek
        self.features_df['day_of_month'] = timestamp.dt.day
        self.features_df['month_of_year'] = timestamp.dt.month
        self.features_df['quarter'] = timestamp.dt.quarter
        self.features_df['is_weekend'] = timestamp.dt.dayofweek.isin([5, 6]).astype(int)
        
        # Cyclical encoding of time features (maintains circular relationship)
        self.features_df['hour_sin'] = np.sin(2 * np.pi * self.features_df['hour_of_day'] / 24)
        self.features_df['hour_cos'] = np.cos(2 * np.pi * self.features_df['hour_of_day'] / 24)
        self.features_df['day_sin'] = np.sin(2 * np.pi * self.features_df['day_of_week'] / 7)
        self.features_df['day_cos'] = np.cos(2 * np.pi * self.features_df['day_of_week'] / 7)
        self.features_df['month_sin'] = np.sin(2 * np.pi * self.features_df['month_of_year'] / 12)
        self.features_df['month_cos'] = np.cos(2 * np.pi * self.features_df['month_of_year'] / 12)
        
        return self
    
    def add_correlation_features(self, columns=None, window=60, include_original=True):
        """
        Add correlation features between sensors.
        
        Parameters:
        -----------
        columns : list or None
            List of column names to process. If None, all numeric columns will be processed.
        window : int
            Rolling window size for correlation calculation
        include_original : bool
            Whether to include original columns in the output
            
        Returns:
        --------
        self : for method chaining
        """
        if self.df is None:
            raise ValueError("No data loaded. Use load_data method first.")
            
        # If columns not specified, use all numeric columns
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        # Need at least 2 columns to calculate correlation
        if len(columns) < 2:
            return self
        
        # Initialize features DataFrame if needed
        if self.features_df is None:
            if include_original:
                self.features_df = self.df[columns].copy()
            else:
                self.features_df = pd.DataFrame(index=self.df.index)
        
        # Prepare to collect all correlation features
        corr_features = {}
        
        # Calculate pairwise correlations
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i < j:
                    # Calculate rolling correlation
                    corr_name = f"{col1}_{col2}_corr_{window}"
                    corr_features[corr_name] = self.df[col1].rolling(window).corr(self.df[col2])
                    
                    # Calculate rolling covariance
                    cov_name = f"{col1}_{col2}_cov_{window}"
                    corr_features[cov_name] = self.df[col1].rolling(window).cov(self.df[col2])
        
        # Create a DataFrame with all correlation features at once
        corr_df = pd.DataFrame(corr_features, index=self.df.index)
        
        # Join with existing features
        self.features_df = pd.concat([self.features_df, corr_df], axis=1)
        
        return self
    
    def scale_features(self, method='standard', columns=None):
        """
        Scale the features using StandardScaler or MinMaxScaler.
        
        Parameters:
        -----------
        method : str
            Scaling method ('standard' or 'minmax')
        columns : list or None
            List of column names to scale. If None, all numeric columns will be scaled.
            
        Returns:
        --------
        self : for method chaining
        """
        if self.features_df is None:
            raise ValueError("No features generated. Generate features first.")
        
        # If columns not specified, use all numeric columns
        if columns is None:
            columns = self.features_df.select_dtypes(include=[np.number]).columns
        
        # Create a copy to avoid modifying the original
        scaled_df = self.features_df.copy()
        
        # Choose scaler based on method
        for col in columns:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError("Method must be 'standard' or 'minmax'")
            
            # Fit and transform
            values = scaled_df[col].values.reshape(-1, 1)
            # Handle NaN values
            mask = ~np.isnan(values.ravel())
            if np.any(mask):
                values_to_scale = values[mask].reshape(-1, 1)
                scaler.fit(values_to_scale)
                values[mask] = scaler.transform(values_to_scale)
                scaled_df[col] = values
                # Store the scaler for future use
                self.scalers[col] = scaler
        
        # Update features_df with scaled values
        self.features_df = scaled_df
        
        return self
    
    def handle_missing_values(self, method='interpolate', limit=5):
        """
        Handle missing values in the features DataFrame.
        
        Parameters:
        -----------
        method : str
            Method for handling missing values ('interpolate', 'ffill', 'bfill', 'drop', 'zero')
        limit : int
            Maximum number of consecutive NaN values to fill when using interpolate/ffill/bfill
            
        Returns:
        --------
        self : for method chaining
        """
        if self.features_df is None:
            raise ValueError("No features generated. Generate features first.")
        
        if method == 'interpolate':
            self.features_df = self.features_df.interpolate(method='linear', limit=limit)
            # Handle remaining NaNs at the beginning/end
            self.features_df = self.features_df.ffill().bfill()
        elif method == 'ffill':
            self.features_df = self.features_df.fillna(method='ffill', limit=limit)
            # Handle remaining NaNs at the beginning
            self.features_df = self.features_df.bfill()
        elif method == 'bfill':
            self.features_df = self.features_df.fillna(method='bfill', limit=limit)
            # Handle remaining NaNs at the end
            self.features_df = self.features_df.ffill()
        elif method == 'zero':
            self.features_df = self.features_df.fillna(0)
        elif method == 'drop':
            self.features_df = self.features_df.dropna()
        else:
            raise ValueError("Method must be 'interpolate', 'ffill', 'bfill', 'zero', or 'drop'")
        
        return self
    
    def select_features(self, columns):
        """
        Select specific columns from the features DataFrame.
        
        Parameters:
        -----------
        columns : list
            List of column names to keep
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with selected features
        """
        if self.features_df is None:
            raise ValueError("No features generated. Generate features first.")
        
        # Ensure all requested columns exist
        missing_cols = [col for col in columns if col not in self.features_df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")
        
        return self.features_df[columns]
    
    def get_features(self):
        """
        Get the complete features DataFrame.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with all generated features
        """
        if self.features_df is None:
            raise ValueError("No features generated. Generate features first.")
        
        return self.features_df
    
    def plot_features(self, columns=None, n_cols=3, figsize=(15, 4)):
        """
        Plot selected features.
        
        Parameters:
        -----------
        columns : list or None
            List of column names to plot. If None, up to 9 random columns will be plotted.
        n_cols : int
            Number of columns in the plot grid
        figsize : tuple
            Figure size per subplot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if self.features_df is None:
            raise ValueError("No features generated. Generate features first.")
        
        # If columns not specified, select a random sample
        if columns is None:
            all_cols = list(self.features_df.columns)
            columns = np.random.choice(all_cols, min(9, len(all_cols)), replace=False)
        
        n_plots = len(columns)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows))
        axes = np.array(axes).reshape(-1)
        
        for i, col in enumerate(columns):
            ax = axes[i]
            self.features_df[col].plot(ax=ax)
            ax.set_title(col)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True)
        
        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def save_features(self, filename='engineered_features.csv'):
        """
        Save the features DataFrame to a CSV file.
        
        Parameters:
        -----------
        filename : str
            Output filename
        """
        if self.features_df is None:
            raise ValueError("No features generated. Generate features first.")
        
        # Reset index to include timestamp column
        output_df = self.features_df.reset_index()
        output_df.to_csv(filename, index=False)
        print(f"Features saved to {filename}")
    
    def run_full_pipeline(self, df=None, sensor_columns=None, scale=True, handle_missing=True):
        """
        Run the complete feature engineering pipeline with default settings.
        
        Parameters:
        -----------
        df : pandas.DataFrame or None
            Input DataFrame. If None, uses the previously loaded DataFrame.
        sensor_columns : list or None
            List of sensor columns to process. If None, uses all numeric columns.
        scale : bool
            Whether to scale the features
        handle_missing : bool
            Whether to handle missing values
            
        Returns:
        --------
        pandas.DataFrame
            The complete engineered features DataFrame
        """
        if df is not None:
            self.load_data(df)
        
        if self.df is None:
            raise ValueError("No data loaded. Provide a DataFrame or use load_data method first.")
        
        # If sensor columns not specified, use all numeric columns
        if sensor_columns is None:
            sensor_columns = self.df.select_dtypes(include=[np.number]).columns
        
        # Run the feature engineering steps
        (self
         .generate_rolling_statistics(columns=sensor_columns)
         .generate_lag_features(columns=sensor_columns)
         .extract_frequency_domain_features(columns=sensor_columns)
         .add_time_features()
         .add_correlation_features(columns=sensor_columns)
        )
        

        if len(self.df) >= 1440: 
            self.decompose_trend_seasonality(columns=sensor_columns)
        
        try:
            self.extract_wavelet_features(columns=sensor_columns)
        except Exception as e:
            print(f"Warning: Could not extract wavelet features: {e}")
        
        # Handle missing values
        if handle_missing:
            self.handle_missing_values()
        
        # Scale features
        if scale:
            self.scale_features()
        
        return self.features_df