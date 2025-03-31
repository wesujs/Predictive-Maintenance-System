import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from data_simulation.sensor_data_simulator import SensorDataSimulator
from data_simulation.time_series_feature_engineering import TimeSeriesFeatureEngineering
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def run_feature_engineering_demo():
    """Demonstrate the time series feature engineering pipeline with anomaly detection"""
    # Create a simulator and generate data
    print("Generating sensor data...")
    start_date = datetime(2025, 3, 1)
    simulator = SensorDataSimulator(start_date=start_date, days=7, sample_rate=60)
    simulator.generate_all_sensor_data()
    
    # Add anomalies to the data
    simulator.add_spike_anomaly('temperature', 0.3, duration_ratio=0.005, amplitude_factor=4.0)
    simulator.add_drift_anomaly('pressure', 0.5, 0.7, drift_factor=1.5)
    simulator.add_intermittent_anomaly('vibration', 0.4, 0.5, frequency=0.3, amplitude_factor=3.0)
    simulator.add_spike_anomaly('temperature', 0.7, duration_ratio=0.003, amplitude_factor=3.5)
    simulator.add_drift_anomaly('power_consumption', 0.6, 0.9, drift_factor=2.0)
    
    # Track anomalies for evaluation
    # From test_anomaly_detection.py
    anomaly_tracker = pd.DataFrame({'timestamp': simulator.timestamps})
    for sensor_name in simulator.sensor_data.keys():
        anomaly_tracker[f"{sensor_name}_anomaly"] = False
    
    # Mark where anomalies were added (simplified version)
    anomaly_indices = {
        'temperature': [
            list(range(int(0.3 * simulator.total_points - 5), int(0.3 * simulator.total_points + 5))),
            list(range(int(0.7 * simulator.total_points - 3), int(0.7 * simulator.total_points + 3)))
        ],
        'pressure': [
            list(range(int(0.5 * simulator.total_points), int(0.7 * simulator.total_points)))
        ],
        'vibration': [
            list(range(int(0.4 * simulator.total_points), int(0.5 * simulator.total_points)))
        ],
        'power_consumption': [
            list(range(int(0.6 * simulator.total_points), int(0.9 * simulator.total_points)))
        ]
    }
    
    # Apply the anomaly indices to the tracker
    for sensor_name, indices_list in anomaly_indices.items():
        for indices in indices_list:
            for idx in indices:
                if 0 <= idx < len(anomaly_tracker):
                    anomaly_tracker.loc[idx, f"{sensor_name}_anomaly"] = True
    
    # Get the data as DataFrame
    df = simulator.to_dataframe()
    
    # Initialize the feature engineering pipeline
    print("Running feature engineering pipeline...")
    pipeline = TimeSeriesFeatureEngineering()
    pipeline.load_data(df)
    
    # Run different feature engineering steps
    # Generate rolling statistics
    pipeline.generate_rolling_statistics(windows=[5, 10, 30])
    
    # Generate lag features
    pipeline.generate_lag_features(lags=[1, 5, 10])
    
    # Extract frequency domain features
    pipeline.extract_frequency_domain_features()
    
    # Add time-based features
    pipeline.add_time_features()
    
    # Add correlation features
    pipeline.add_correlation_features(window=30)
    
    # Handle missing values
    pipeline.handle_missing_values()
    
    # Get the complete feature set
    features_df = pipeline.get_features()
    
    print(f"Generated {features_df.shape[1]} features from {df.shape[1]} original sensor readings")
    
    # Create labels from the anomaly tracker
    labels_df = pd.DataFrame()
    
    # Create a combined anomaly label (True if any sensor has an anomaly)
    combined_anomaly = anomaly_tracker[[col for col in anomaly_tracker.columns if col.endswith('_anomaly')]].any(axis=1)
    labels_df['anomaly'] = combined_anomaly
    
    # Join features with labels
    features_df = features_df.reset_index()
    anomaly_tracker = anomaly_tracker.reset_index(drop=True)
    combined_df = pd.concat([features_df, labels_df], axis=1)
    
    # Train a simple anomaly classifier
    print("Training anomaly detection model with engineered features...")
    
    # Prepare features and target
    X = combined_df.drop(columns=['timestamp', 'anomaly'])
    y = combined_df['anomaly']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Visualize some of the engineered features
    print("\nVisualizing sample engineered features...")
    
    # Select a few interesting features to plot
    selected_features = [
        'temperature', 
        'temperature_roll_std_30',
        'temperature_lag_5',
        'pressure_dominant_freq',
        'vibration_residual',
        'temperature_power_consumption_corr_30'
    ]
    
    # Filter features that actually exist in the DataFrame
    existing_features = [f for f in selected_features if f in combined_df.columns]
    
    # Plot
    fig = plt.figure(figsize=(15, 12))
    for i, feature in enumerate(existing_features):
        ax = fig.add_subplot(len(existing_features), 1, i+1)
        ax.plot(combined_df['timestamp'], combined_df[feature], label=feature)
        
        # Highlight anomaly regions
        anomaly_regions = combined_df[combined_df['anomaly']]
        ax.scatter(anomaly_regions['timestamp'], anomaly_regions[feature], 
                   color='red', label='Anomaly', alpha=0.5)
        
        ax.set_title(feature)
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('engineered_features_visualization.png')
    plt.close()
    
    print("Feature visualization saved to 'engineered_features_visualization.png'")
    
    # Save the engineered features
    pipeline.save_features('engineered_features.csv')
    
    return pipeline, model, feature_importance

if __name__ == "__main__":
    run_feature_engineering_demo()