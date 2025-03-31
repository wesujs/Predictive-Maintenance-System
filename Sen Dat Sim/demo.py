import os
import time
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import subprocess
import sys

# Import custom modules
from data_simulation.sensor_data_simulator import SensorDataSimulator
from data_simulation.time_series_feature_engineering import TimeSeriesFeatureEngineering 
from data_simulation.advanced_anomaly_detection import AdvancedAnomalyDetection
from pipeline.real_time_processing_pipeline import RealTimeProcessingPipeline
from pipeline.setup_kafka import setup_kafka
print("Demo script starting")

def setup_results_folder():
    """Create a timestamped results folder for this run"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"runs/run_{timestamp}"
    
    # Create main results directory
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create subdirectories
    for subdir in ['data', 'models', 'logs', 'results']:
        path = os.path.join(results_dir, subdir)
        if not os.path.exists(path):
            os.makedirs(path)
    
    print(f"Created results directory: {results_dir}")
    return results_dir

def generate_training_data(days=7, add_anomalies=True, output_dir='data', plots_dir='results'):
    """Generate training data using the sensor data simulator"""
    print("Generating training data...")
    
    # Create simulator instance
    simulator = SensorDataSimulator(
        start_date=datetime.now(),
        days=days,
        sample_rate=60  # 1 reading per minute
    )
    
    # Generate base sensor data
    simulator.generate_all_sensor_data()
    
    # Add anomalies if requested
    if add_anomalies:
        # Add specific anomalies
        simulator.add_spike_anomaly('temperature', 0.3, duration_ratio=0.005, amplitude_factor=4.0)
        simulator.add_drift_anomaly('pressure', 0.6, 0.8, drift_factor=1.5)
        simulator.add_intermittent_anomaly('vibration', 0.4, 0.5, frequency=0.3, amplitude_factor=3.0)
        
        # Add some random anomalies to each sensor
        simulator.add_random_anomalies(count=5)
    
    # Save the data to CSV
    df = simulator.to_dataframe()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'training_data.csv'), index=False)
    
    # Plot the data
    plt.figure(figsize=(12, 8))
    simulator.plot_sensors()
    plt.savefig(os.path.join(plots_dir, 'training_data_plot.png'))
    
    print(f"Generated training data with {len(df)} rows")
    return df

def engineer_features(input_df=None, plot_features=True, output_dir='data', plots_dir='results'):
    """Run the feature engineering pipeline on training data"""
    print("Running feature engineering pipeline...")
    
    # Load data if not provided
    if input_df is None:
        data_path = os.path.join(output_dir, 'training_data.csv')
        if os.path.exists(data_path):
            input_df = pd.read_csv(data_path)
            # Make sure timestamp is datetime
            input_df['timestamp'] = pd.to_datetime(input_df['timestamp'])
        else:
            print("No training data found. Generating new data...")
            input_df = generate_training_data(output_dir=output_dir, plots_dir=plots_dir)
    
    # Create feature engineering pipeline
    pipeline = TimeSeriesFeatureEngineering()
    pipeline.load_data(input_df)
    
    # Run the full pipeline
    features_df = pipeline.run_full_pipeline()
    
    # Save the engineered features
    os.makedirs(output_dir, exist_ok=True)
    features_path = os.path.join(output_dir, 'engineered_features.csv')
    pipeline.save_features(features_path)
    
    # Plot some features
    if plot_features:
        fig = pipeline.plot_features(
            columns=[
                'temperature', 'temperature_roll_mean_30', 'temperature_roll_std_30',
                'vibration', 'vibration_roll_mean_30', 'vibration_roll_std_30',
                'pressure', 'pressure_dominant_freq',
                'power_consumption', 'power_consumption_trend'
            ]
        )
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, 'engineered_features_plot.png'))
    
    print(f"Generated {features_df.shape[1]} features from {input_df.shape[1]} original variables")
    return features_df

def create_labels(raw_df, output_dir='data'):
    """Create anomaly labels for training data"""
    print("Creating anomaly labels...")
    
    # Initialize labels (0 = normal, 1 = anomaly)
    anomaly_labels = pd.Series(0, index=range(len(raw_df)))
    
    # Create labels based on threshold method
    for col in ['temperature', 'pressure', 'vibration', 'power_consumption']:
        if col in raw_df.columns:
            # Calculate mean and std for each sensor
            mean = raw_df[col].mean()
            std = raw_df[col].std()
            
            # Mark anomalies where values deviate significantly
            anomaly_labels = anomaly_labels | (raw_df[col] > mean + 3*std) | (raw_df[col] < mean - 3*std)
    
    # Save labels
    os.makedirs(output_dir, exist_ok=True)
    labels_path = os.path.join(output_dir, 'anomaly_labels.csv')
    anomaly_labels.to_csv(labels_path, index=False)
    
    # Print statistics
    anomaly_count = anomaly_labels.sum()
    print(f"Created labels: {anomaly_count} anomalies and {len(anomaly_labels) - anomaly_count} normal points")
    
    return anomaly_labels

def train_anomaly_detection_models(features_df=None, labels=None, models_dir='models'):
    """Train various anomaly detection models"""
    print("Training anomaly detection models...")
    
    # Load data if not provided
    if features_df is None:
        features_path = os.path.join(models_dir, '..', 'data', 'engineered_features.csv')
        if os.path.exists(features_path):
            features_df = pd.read_csv(features_path)
        else:
            print("No engineered features found. Running feature engineering...")
            raw_df = generate_training_data()
            features_df = engineer_features(raw_df)
    
    if labels is None:
        labels_path = os.path.join(models_dir, '..', 'data', 'anomaly_labels.csv')
        if os.path.exists(labels_path):
            labels = pd.read_csv(labels_path).iloc[:, 0]
        else:
            print("No labels found. Creating labels...")
            raw_df = pd.read_csv(os.path.join(models_dir, '..', 'data', 'training_data.csv'))
            labels = create_labels(raw_df)
    
    # Select numeric features and handle missing values
    X = features_df.select_dtypes(include=[np.number]).fillna(0)
    y = labels
    
    # Make sure X and y have the same length
    min_len = min(len(X), len(y))
    X = X.iloc[:min_len]
    y = y.iloc[:min_len]
    
    # Create anomaly detector with specified models directory
    detector = AdvancedAnomalyDetection(model_dir=models_dir)
    
    # Train models
    print("Training Isolation Forest model...")
    detector.train_isolation_forest(X, contamination=0.05)
    
    print("Training One-Class SVM model...")
    detector.train_one_class_svm(X, nu=0.05)
    
    print("Training supervised Random Forest...")
    detector.train_supervised_rf(X, y)
    
    print("Training supervised XGBoost...")
    detector.train_supervised_xgboost(X, y)
    
    try:
        print("Training Autoencoder model...")
        detector.train_autoencoder(X, epochs=20, batch_size=32)
    except Exception as e:
        print(f"Could not train Autoencoder: {e}")
    
    # Save all models
    detector.save_models()
    
    print(f"Successfully trained and saved {len(detector.models)} models to {models_dir}")
    return detector

def test_models(detector=None, features_df=None, labels=None, logs_dir='logs', models_dir='models', data_dir='data'):
    """Test trained models on validation data"""
    print("Testing anomaly detection models...")
   
    # Load models if not provided
    if detector is None:
        detector = AdvancedAnomalyDetection(model_dir=models_dir)
        try:
            detector.load_models()
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Training new models...")
            detector = train_anomaly_detection_models(models_dir=models_dir)
   
    # Load data if not provided
    if features_df is None or labels is None:
        features_path = os.path.join(data_dir, 'engineered_features.csv')
        labels_path = os.path.join(data_dir, 'anomaly_labels.csv')
        if os.path.exists(features_path) and os.path.exists(labels_path):
            features_df = pd.read_csv(features_path)
            labels = pd.read_csv(labels_path).iloc[:, 0]
           
            # Select numeric features and handle missing values
            X = features_df.select_dtypes(include=[np.number])
            
            # Add robust NaN handling here
            print(f"Data shape before handling NaNs: {X.shape}")
            print(f"Number of NaN values: {X.isna().sum().sum()}")
            
            # Replace NaN values with 0 (or use another imputation strategy)
            X = X.fillna(0)
            
            # Double-check for any remaining NaNs
            if X.isna().sum().sum() > 0:
                print("Warning: NaN values still present after fillna. Using dropna.")
                X = X.dropna(axis=1)
                
            print(f"Data shape after handling NaNs: {X.shape}")
            
            y = labels
           
            # Make sure X and y have the same length
            min_len = min(len(X), len(y))
            X = X.iloc[:min_len]
            y = y.iloc[:min_len]
        else:
            print("No data found for testing. Skipping test...")
            return
    else:
        X = features_df
        y = labels
        
        # Also handle NaNs when direct data is provided
        if isinstance(X, pd.DataFrame) and X.isna().sum().sum() > 0:
            print(f"Handling {X.isna().sum().sum()} NaN values in provided features")
            X = X.fillna(0)
   
    # Make sure logs directory exists
    os.makedirs(logs_dir, exist_ok=True)
    
    # Split data for validation
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Evaluate models and save results to logs directory
    for model_name in detector.models.keys():
        try:
            results = detector.predict_anomalies(X_test, model_name=model_name)
            predictions = results[model_name]
            
            # Calculate basic metrics
            from sklearn.metrics import classification_report, confusion_matrix
            report = classification_report(y_test, predictions, output_dict=True)
            
            # Save results to logs
            with open(os.path.join(logs_dir, f'{model_name}_results.txt'), 'w') as f:
                f.write(f"Test results for {model_name}:\n\n")
                f.write(f"Classification Report:\n{classification_report(y_test, predictions)}\n\n")
                f.write(f"Confusion Matrix:\n{confusion_matrix(y_test, predictions)}\n")
            
            print(f"Tested {model_name} - Accuracy: {report['accuracy']:.4f}")
            
        except Exception as e:
            print(f"Error testing model {model_name}: {e}")
    
    print(f"Testing complete. Results saved to {logs_dir}")
    return detector


def run_kafka_pipeline(duration=300, logs_dir='logs'):
    """Run the real-time Kafka pipeline demo"""
    print("Checking Kafka availability...")
   
    # Make sure logs directory exists
    os.makedirs(logs_dir, exist_ok=True)
    
    # Check if Kafka is running
    import socket
    kafka_available = False
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect(('localhost', 9092))
        s.close()
        kafka_available = True
    except:
        print("Kafka doesn't appear to be running on localhost:9092.")
        return False
   
    if kafka_available:
        print("Kafka is available! Starting real-time pipeline...")
       
        # Create the pipeline with logs directed to the logs directory
        pipeline = RealTimeProcessingPipeline(
            bootstrap_servers='localhost:9092',
            window_size=60,
            sliding_step=10
        )
       
        try:
            # Save logs to the specified directory
            log_file = os.path.join(logs_dir, 'kafka_pipeline.log')
            with open(log_file, 'w') as f:
                f.write(f"Kafka pipeline run started at {datetime.now()}\n")
            
            pipeline.run_demo(duration_seconds=duration)
            
            # Save performance stats
            with open(log_file, 'a') as f:
                f.write("\nPerformance Statistics:\n")
                stats = pipeline.get_performance_stats()
                for stage, metrics in stats.items():
                    f.write(f"\n{stage}:\n")
                    for key, value in metrics.items():
                        f.write(f"  {key}: {value}\n")
            
            return True
        except Exception as e:
            print(f"Error running Kafka pipeline: {e}")
            with open(os.path.join(logs_dir, 'kafka_error.log'), 'w') as f:
                f.write(f"Error at {datetime.now()}: {e}")
            return False
    else:
        print("Real-time pipeline requires Kafka to be running.")
        print("Run with --kafka-setup to automatically start Kafka")
        return False

def run_kafka_demo_with_setup(logs_dir='logs'):
    """Run the Kafka demo with automatic setup if needed"""
    # Make sure logs directory exists
    os.makedirs(logs_dir, exist_ok=True)
    
    # Check if Kafka is running
    kafka_available = setup_kafka.check_kafka_running()
   
    if not kafka_available:
        print("Kafka is not running. Starting Kafka...")

        # Check if Docker is installed
        if not setup_kafka.check_docker_installed() or not setup_kafka.check_docker_compose_installed():
            print("Docker or Docker Compose is not installed. Cannot start Kafka.")
            print("Please install Docker and Docker Compose or start Kafka manually.")
            return False
       
        # Start Kafka
        if not setup_kafka.start_kafka():
            print("Failed to start Kafka. Exiting.")
            return False
       
        print("Kafka started successfully.")
    else:
        print("Kafka is already running on localhost:9092")
   
    # Now that Kafka is running, use the existing function to run the pipeline
    return run_kafka_pipeline(duration=180, logs_dir=logs_dir)
def run_complete_demo(results_dir=None):
    """Run a complete demonstration of the predictive maintenance system"""
    print("Starting Predictive Maintenance System Demo\n")
    
    # Use provided results_dir or create default
    if results_dir is None:
        # Create timestamped directory
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"runs/run_{timestamp}"
        
        # Create directory structure
        for subdir in ['data', 'models', 'logs', 'results']:
            path = os.path.join(results_dir, subdir)
            os.makedirs(path, exist_ok=True)
        
        print(f"Created results directory: {results_dir}")
    
    # Define paths for all outputs
    data_dir = os.path.join(results_dir, "data")
    models_dir = os.path.join(results_dir, "models")
    logs_dir = os.path.join(results_dir, "logs")
    plots_dir = os.path.join(results_dir, "results")
    
    # Generate training data
    print("\n=== Step 1: Generating Training Data ===")
    raw_df = generate_training_data(days=7, add_anomalies=True, output_dir=data_dir, plots_dir=plots_dir)
    
    # Engineer features
    print("\n=== Step 2: Engineering Features ===")
    features_df = engineer_features(raw_df, plot_features=True, output_dir=data_dir, plots_dir=plots_dir)
    
    # Create labels
    print("\n=== Step 3: Creating Anomaly Labels ===")
    labels = create_labels(raw_df, output_dir=data_dir)
    
    # Train models
    print("\n=== Step 4: Training Anomaly Detection Models ===")
    detector = train_anomaly_detection_models(features_df, labels, models_dir=models_dir)
    
    # Test models
    print("\n=== Step 5: Testing Models ===")
    test_models(detector, features_df, labels, logs_dir=logs_dir)
    
    # Run real-time pipeline (if Kafka is available)
    print("\n=== Step 6: Running Real-Time Pipeline ===")
    kafka_result = run_kafka_pipeline(duration=180, logs_dir=logs_dir)
    
    print(f"\nDemo Complete! All results saved to {results_dir}")
    if not kafka_result:
        print("Note: Real-time pipeline was skipped due to Kafka not being available.")
        print("You can run the full demo with Kafka later by setting up Kafka and running:")
        print("python demo.py --kafka-only")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predictive Maintenance System Demo")
    parser.add_argument('--output-dir', type=str, default=None, help='Custom output directory for all generated files')
    parser.add_argument('--generate-data', action='store_true', help='Generate training data only')
    parser.add_argument('--feature-engineering', action='store_true', help='Run feature engineering only')
    parser.add_argument('--train-models', action='store_true', help='Train anomaly detection models only')
    parser.add_argument('--test-models', action='store_true', help='Test trained models only')
    parser.add_argument('--kafka-only', action='store_true', help='Run Kafka pipeline only')
    parser.add_argument('--kafka-setup', action='store_true', help='Setup and run Kafka pipeline')
    parser.add_argument('--all', action='store_true', help='Run complete demo (default)')
    
    args = parser.parse_args()
    
    # If no args specified, run complete demo
    if not any(var for key, var in vars(args).items() if key != 'output_dir'):
        args.all = True
    
    # Set up results directory
    results_dir = None
    if args.output_dir:
        results_dir = args.output_dir
    else:
        # Create timestamped directory
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"runs/run_{timestamp}"
    
    # Create directory structure
    for subdir in ['data', 'models', 'logs', 'results']:
        path = os.path.join(results_dir, subdir)
        os.makedirs(path, exist_ok=True)
    
    print(f"Using output directory: {results_dir}")
    
    # Define paths for all outputs
    data_dir = os.path.join(results_dir, "data")
    models_dir = os.path.join(results_dir, "models")
    logs_dir = os.path.join(results_dir, "logs")
    plots_dir = os.path.join(results_dir, "results")
    
    # Run requested components
    if args.generate_data:
        generate_training_data(output_dir=data_dir, plots_dir=plots_dir)
    
    if args.feature_engineering:
        data_path = os.path.join(data_dir, 'training_data.csv')
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            engineer_features(df, output_dir=data_dir, plots_dir=plots_dir)
        else:
            print("No training data found. Generating data first...")
            df = generate_training_data(output_dir=data_dir, plots_dir=plots_dir)
            engineer_features(df, output_dir=data_dir, plots_dir=plots_dir)
    
    if args.train_models:
        data_path = os.path.join(data_dir, 'engineered_features.csv')
        labels_path = os.path.join(data_dir, 'anomaly_labels.csv')
        
        if os.path.exists(data_path) and os.path.exists(labels_path):
            features_df = pd.read_csv(data_path)
            labels = pd.read_csv(labels_path).iloc[:, 0]
            train_anomaly_detection_models(features_df, labels, models_dir=models_dir)
        else:
            print("Missing engineered features or labels. Running previous steps...")
            raw_df = generate_training_data(output_dir=data_dir, plots_dir=plots_dir)
            features_df = engineer_features(raw_df, output_dir=data_dir, plots_dir=plots_dir)
            labels = create_labels(raw_df, output_dir=data_dir)
            train_anomaly_detection_models(features_df, labels, models_dir=models_dir)
    
    if args.test_models:
        test_models(models_dir=models_dir, data_dir=data_dir, logs_dir=logs_dir)
    
    if args.kafka_only:
        run_kafka_pipeline(logs_dir=logs_dir)
    
    if args.kafka_setup:
        run_kafka_demo_with_setup(logs_dir=logs_dir)
    
    if args.all:
        run_complete_demo(results_dir=results_dir)