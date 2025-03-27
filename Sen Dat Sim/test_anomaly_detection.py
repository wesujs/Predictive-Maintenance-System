import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sensor_data_simulator import SensorDataSimulator

# Definition of anomaly detection methods
def threshold_detector(df, sensor_name, stddev_factor=3):
    """ Detect anomalies using a simple threshold based on standard deviation """
    mean = df[sensor_name].mean()
    std = df[sensor_name].std()
    
    upper_threshold = mean + stddev_factor * std
    lower_threshold = mean - stddev_factor * std
    
    anomalies = (df[sensor_name] > upper_threshold) | (df[sensor_name] < lower_threshold)
    
    return anomalies.values

def moving_average_detector(df, sensor_name, window=20, threshold_factor=2.5):
    """ Detect anomalies by comparing values to a moving average """
    rolling = df[sensor_name].rolling(window=window)
    moving_avg = rolling.mean()
    moving_std = rolling.std()
    
    deviations = abs(df[sensor_name] - moving_avg)
    threshold = threshold_factor * moving_std
    
    anomalies = deviations > threshold
    anomalies = anomalies.fillna(False)
    
    return anomalies.values

def z_score_detector(df, sensor_name, window=20, threshold=3.0):
    """ Detect anomalies using z-scores with median """
    data = df[sensor_name].copy()
    
    rolling_median = data.rolling(window=window).median()
    # Use a small epsilon to avoid division by zero
    rolling_mad = (data - rolling_median).abs().rolling(window=window).median() + 1e-9
    
    z_scores = 0.6745 * (data - rolling_median) / rolling_mad
    z_scores = z_scores.fillna(0)
    
    anomalies = abs(z_scores) > threshold
    anomalies = anomalies.fillna(False)
    
    return anomalies.values

# Add the method to track anomalies to your SensorDataSimulator class
def add_anomaly_tracking(simulator):
    """ Create a tracking DataFrame that records where anomalies were inserted """
    simulator.anomaly_tracker = pd.DataFrame({'timestamp': simulator.timestamps})
    
    for sensor_name in simulator.sensor_data.keys():
        simulator.anomaly_tracker[f"{sensor_name}_anomaly"] = False
    
    # Monkey patch the anomaly methods to record where they add anomalies
    original_spike = simulator.add_spike_anomaly
    original_drift = simulator.add_drift_anomaly
    original_intermittent = simulator.add_intermittent_anomaly
    
    def tracked_spike_anomaly(sensor_name, position_ratio, duration_ratio=0.002, amplitude_factor=3.0):
        position = int(position_ratio * simulator.total_points)
        duration = max(1, int(duration_ratio * simulator.total_points))
        start_idx = max(0, position - duration // 2)
        end_idx = min(simulator.total_points, position + duration // 2)
        
        # Call original method
        original_spike(sensor_name, position_ratio, duration_ratio, amplitude_factor)
        
        # Track the anomaly
        simulator.anomaly_tracker.loc[start_idx:end_idx, f"{sensor_name}_anomaly"] = True
    
    def tracked_drift_anomaly(sensor_name, start_ratio, end_ratio, drift_factor=2.0):
        start_idx = int(start_ratio * simulator.total_points)
        end_idx = int(end_ratio * simulator.total_points)
        
        # Call original method
        original_drift(sensor_name, start_ratio, end_ratio, drift_factor)
        
        # Track the anomaly
        simulator.anomaly_tracker.loc[start_idx:end_idx, f"{sensor_name}_anomaly"] = True
    
    def tracked_intermittent_anomaly(sensor_name, start_ratio, end_ratio, frequency=0.2, amplitude_factor=2.0):
        start_idx = int(start_ratio * simulator.total_points)
        end_idx = int(end_ratio * simulator.total_points)
        
        # Call original method
        original_intermittent(sensor_name, start_ratio, end_ratio, frequency, amplitude_factor)
        
        # Track the anomaly - simulating random occurrence based on frequency
        for i in range(start_idx, end_idx):
            if np.random.random() < frequency:
                simulator.anomaly_tracker.loc[i, f"{sensor_name}_anomaly"] = True
    
    # Replace the methods with our tracked versions
    simulator.add_spike_anomaly = tracked_spike_anomaly
    simulator.add_drift_anomaly = tracked_drift_anomaly
    simulator.add_intermittent_anomaly = tracked_intermittent_anomaly

def evaluate_detector(simulator, detector_function, detector_name, **detector_params):
    """ Evaluation of a detector against all sensors and return results """
    results = {}
    
    for sensor_name in simulator.sensor_data.keys():
        # Create a wrapper for the specific detector with its parameters
        def detector_wrapper(df, sensor):
            return detector_function(df, sensor, **detector_params)
        
        # Evaluate the detector for this sensor
        detection_results = evaluate_anomaly_detection(simulator, detector_wrapper, sensor_name)
        results[sensor_name] = detection_results
    
    print(f"\nResults for {detector_name}:")
    for sensor, metrics in results.items():
        print(f"  {sensor}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1 Score: {metrics['f1']:.4f}")
    
    return results

def evaluate_anomaly_detection(simulator, detection_function, sensor_name):
    """ Evaluation of an anomaly detection algorithm against known anomalies """
    if not hasattr(simulator, 'anomaly_tracker'):
        raise ValueError("No anomaly tracking data available.")
    
    # Get the data as a DataFrame
    df = simulator.to_dataframe()
    
    # Call the detection function to get predictions for this sensor
    predicted_anomalies = detection_function(df, sensor_name)
    
    # Get the ground truth
    true_anomalies = simulator.anomaly_tracker[f"{sensor_name}_anomaly"].values
    
    # Calculate metrics
    true_positives = sum(predicted_anomalies & true_anomalies)
    false_positives = sum(predicted_anomalies & ~true_anomalies)
    false_negatives = sum(~predicted_anomalies & true_anomalies)
    true_negatives = sum(~predicted_anomalies & ~true_anomalies)
    
    # Calculate performance metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives
    }

def plot_detection_results(simulator, sensor_name, detector_function, detector_name, **detector_params):
    """ Visualization of the detection results for a specific sensor """
    df = simulator.to_dataframe()
    
    # Get detector results
    detected_anomalies = detector_function(df, sensor_name, **detector_params)
    
    # Get actual anomalies
    true_anomalies = simulator.anomaly_tracker[f"{sensor_name}_anomaly"].values
    
    # Create a new figure
    plt.figure(figsize=(15, 8))
    
    # Plot the sensor data
    plt.plot(df['timestamp'], df[sensor_name], label=f'{sensor_name} Data', color='blue', alpha=0.7)
    
    # Plot true anomalies in green
    anomaly_values = np.where(true_anomalies, df[sensor_name], np.nan)
    plt.scatter(df['timestamp'], anomaly_values, color='green', label='True Anomalies', s=50, alpha=0.7)
    
    # Plot detected anomalies in red
    detected_values = np.where(detected_anomalies, df[sensor_name], np.nan)
    plt.scatter(df['timestamp'], detected_values, color='red', label='Detected Anomalies', s=25, alpha=0.7)
    
    # Label the plot
    plt.title(f'Anomaly Detection Results for {sensor_name} using {detector_name}')
    plt.xlabel('Time')
    plt.ylabel(f'{sensor_name} ({simulator._get_sensor_unit(sensor_name)})')
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Main test function
def run_anomaly_detection_test():
    # Create a simulator for 7 days of data, with readings every minute
    start_date = datetime(2025, 3, 1)
    simulator = SensorDataSimulator(start_date=start_date, days=7, sample_rate=60)
    
    # Generate data for all sensors
    simulator.generate_all_sensor_data()
    
    # Set up anomaly tracking
    add_anomaly_tracking(simulator)
    
    # Add specific anomalies
    # Add a spike to temperature
    simulator.add_spike_anomaly('temperature', 0.3, duration_ratio=0.005, amplitude_factor=4.0)
    
    # Add a drift to pressure
    simulator.add_drift_anomaly('pressure', 0.5, 0.7, drift_factor=1.5)
    
    # Add intermittent anomalies to vibration
    simulator.add_intermittent_anomaly('vibration', 0.4, 0.5, frequency=0.3, amplitude_factor=3.0)
    
    # Add a second set of anomalies to test more complex patterns
    simulator.add_spike_anomaly('temperature', 0.7, duration_ratio=0.003, amplitude_factor=3.5)
    simulator.add_drift_anomaly('power_consumption', 0.6, 0.9, drift_factor=2.0)
    
    # Different detection algorithms
    
    # 1. Simple threshold detector
    threshold_results = evaluate_detector(simulator, threshold_detector, "Threshold Detector", stddev_factor=3)
    
    # 2. Moving average detector
    ma_results = evaluate_detector(simulator, moving_average_detector, "Moving Average Detector", window=60, threshold_factor=2.5)
    
    # 3. Z-score detector
    z_results = evaluate_detector(simulator, z_score_detector, "Z-Score Detector", window=60, threshold=3.0)
    
    # Plot results for each sensor with each detector
    sensors = list(simulator.sensor_data.keys())
    
    # Choose one sensor and one detector for visualization
    plot_detection_results(simulator, 'temperature', threshold_detector, "Threshold Detector", stddev_factor=3)
    plot_detection_results(simulator, 'pressure', moving_average_detector, "Moving Average Detector", window=60, threshold_factor=2.5)
    plot_detection_results(simulator, 'vibration', z_score_detector, "Z-Score Detector", window=60, threshold=3.0)
    
    # Compare performance across detectors
    print("\nComparison of F1 Scores across detectors:")
    for sensor in sensors:
        print(f"\n{sensor}:")
        print(f"  Threshold Detector: {threshold_results[sensor]['f1']:.4f}")
        print(f"  Moving Average Detector: {ma_results[sensor]['f1']:.4f}")
        print(f"  Z-Score Detector: {z_results[sensor]['f1']:.4f}")
        
        # Determine the best detector for this sensor
        best_f1 = max(threshold_results[sensor]['f1'], ma_results[sensor]['f1'], z_results[sensor]['f1'])
        if best_f1 == threshold_results[sensor]['f1']:
            best = "Threshold Detector"
        elif best_f1 == ma_results[sensor]['f1']:
            best = "Moving Average Detector"
        else:
            best = "Z-Score Detector"
            
        print(f"  Best detector: {best} (F1 = {best_f1:.4f})")

# Run the test
if __name__ == "__main__":
    run_anomaly_detection_test()