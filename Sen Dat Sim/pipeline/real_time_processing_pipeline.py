import json
import time
import threading
from datetime import datetime
import numpy as np
import pandas as pd
from kafka import KafkaProducer, KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic as ConfluentNewTopic

from data_simulation.time_series_feature_engineering import TimeSeriesFeatureEngineering
from data_simulation.advanced_anomaly_detection import AdvancedAnomalyDetection

class RealTimeProcessingPipeline:
    """
    Real-time processing pipeline for predictive maintenance using Kafka
    """
    
    def __init__(self, bootstrap_servers='localhost:9092', 
                 input_topic='sensor_data', 
                 feature_topic='engineered_features',
                 prediction_topic='anomaly_predictions',
                 group_id='predictive_maintenance',
                 model_dir='models',
                 window_size=60,  # 1 hour of data (assuming 1 reading per minute)
                 sliding_step=10,  # Process every 10 minutes
                 use_confluent=False):  # Whether to use confluent_kafka or kafka-python
        """
        Initialize the real-time processing pipeline
        
        Parameters:
        -----------
        bootstrap_servers : str
            Kafka bootstrap servers (comma-separated)
        input_topic : str
            Kafka topic for incoming sensor data
        feature_topic : str
            Kafka topic for feature-engineered data
        prediction_topic : str
            Kafka topic for anomaly predictions
        group_id : str
            Consumer group ID
        model_dir : str
            Directory with trained models
        window_size : int
            Window size for feature calculation (in number of data points)
        sliding_step : int
            How many data points to slide the window each time
        use_confluent : bool
            Whether to use confluent_kafka (True) or kafka-python (False)
        """
        self.bootstrap_servers = bootstrap_servers
        self.input_topic = input_topic
        self.feature_topic = feature_topic
        self.prediction_topic = prediction_topic
        self.group_id = group_id
        self.model_dir = model_dir
        self.window_size = window_size
        self.sliding_step = sliding_step
        self.use_confluent = use_confluent
        
        # Initialize components
        self.feature_engineering = TimeSeriesFeatureEngineering()
        self.anomaly_detector = AdvancedAnomalyDetection(model_dir=model_dir)
        
        # Load trained models
        try:
            self.anomaly_detector.load_models()
        except Exception as e:
            print(f"Warning: Could not load anomaly detection models: {e}")
            print("You will need to train models before making predictions.")
        
        # For storing incoming data in a buffer
        self.buffer = []
        self.buffer_lock = threading.Lock()
        
        # Flag to control threads
        self.running = False
        
        # Time tracker for performance monitoring
        self.processing_times = {
            'feature_engineering': [],
            'anomaly_detection': []
        }
        
    def create_topics(self):
        """Create required Kafka topics if they don't exist"""
        if self.use_confluent:
            admin_client = AdminClient({'bootstrap.servers': self.bootstrap_servers})
            
            # Get existing topics
            metadata = admin_client.list_topics(timeout=10)
            topics = metadata.topics
            
            # Create new topics
            new_topics = []
            for topic in [self.input_topic, self.feature_topic, self.prediction_topic]:
                if topic not in topics:
                    new_topics.append(ConfluentNewTopic(
                        topic=topic,
                        num_partitions=3,
                        replication_factor=1  # Adjust based on your Kafka cluster
                    ))
            
            if new_topics:
                admin_client.create_topics(new_topics)
                print(f"Created topics: {[t.topic for t in new_topics]}")
                
        else:
            try:
                admin_client = KafkaAdminClient(
                    bootstrap_servers=self.bootstrap_servers,
                    client_id='admin'
                )
                
                # Get existing topics
                existing_topics = admin_client.list_topics()
                
                # Create new topics
                new_topics = []
                for topic in [self.input_topic, self.feature_topic, self.prediction_topic]:
                    if topic not in existing_topics:
                        new_topics.append(NewTopic(
                            name=topic,
                            num_partitions=3,
                            replication_factor=1  # Adjust based on your Kafka cluster
                        ))
                
                if new_topics:
                    admin_client.create_topics(new_topics)
                    print(f"Created topics: {[t.name for t in new_topics]}")
                    
            except Exception as e:
                print(f"Warning: Could not create Kafka topics: {e}")
                print("Make sure Kafka is running and accessible.")

    def initialize_kafka_producer(self):
        """Initialize Kafka producer based on selected client"""
        if self.use_confluent:
            return Producer({
                'bootstrap.servers': self.bootstrap_servers,
                'queue.buffering.max.messages': 10000,
                'queue.buffering.max.ms': 100,
                'batch.num.messages': 1000
            })
        else:
            return KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',
                retries=3,
                max_in_flight_requests_per_connection=1
            )
            
    def initialize_kafka_consumer(self, topic, group_id=None):
        """Initialize Kafka consumer based on selected client"""
        if group_id is None:
            group_id = self.group_id
            
        if self.use_confluent:
            return Consumer({
                'bootstrap.servers': self.bootstrap_servers,
                'group.id': group_id,
                'auto.offset.reset': 'earliest',
                'enable.auto.commit': True,
                'auto.commit.interval.ms': 5000
            })
        else:
            return KafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=group_id,
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                auto_commit_interval_ms=5000,
                value_deserializer=lambda v: json.loads(v.decode('utf-8'))
            )
            
    def produce_message(self, producer, topic, message):
        """Send a message to Kafka topic based on producer type"""
        if self.use_confluent:
            producer.produce(
                topic,
                json.dumps(message).encode('utf-8'),
                callback=lambda err, msg: self._delivery_report(err, msg)
            )
            producer.poll(0)  # Trigger any callbacks
        else:
            # kafka-python producer has serializer built-in
            producer.send(topic, message)
            
    def _delivery_report(self, err, msg):
        """Callback for message delivery reports (Confluent Kafka)"""
        if err is not None:
            print(f"Message delivery failed: {err}")
        else:
            # Temp alternative debug statement
            # print(f"Message delivered to {msg.topic()} [{msg.partition()}]")
            pass
            
    def start_data_consumer(self):
        """Start a thread to consume sensor data from Kafka"""
        self.running = True
        consumer_thread = threading.Thread(target=self._consume_sensor_data)
        consumer_thread.daemon = True
        consumer_thread.start()
        
        # Start the processing thread that will analyze the buffer
        processing_thread = threading.Thread(target=self._process_data_buffer)
        processing_thread.daemon = True
        processing_thread.start()
        
        return consumer_thread, processing_thread
        
    def _consume_sensor_data(self):
        """Consume sensor data from Kafka and add to buffer"""
        print(f"Starting consumer for topic: {self.input_topic}")
        
        if self.use_confluent:
            consumer = self.initialize_kafka_consumer(None)
            consumer.subscribe([self.input_topic])
            
            try:
                while self.running:
                    msg = consumer.poll(timeout=1.0)
                    if msg is None:
                        continue
                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            # End of partition event
                            print(f"Reached end of partition {msg.partition()}")
                        else:
                            # Error
                            print(f"Error while consuming: {msg.error()}")
                    else:
                        # Process message
                        try:
                            value = json.loads(msg.value().decode('utf-8'))
                            with self.buffer_lock:
                                self.buffer.append(value)
                        except Exception as e:
                            print(f"Error processing message: {e}")
            finally:
                consumer.close()
                
        else:
            consumer = self.initialize_kafka_consumer(self.input_topic)
            
            try:
                for msg in consumer:
                    if not self.running:
                        break
                    try:
                        with self.buffer_lock:
                            self.buffer.append(msg.value)
                    except Exception as e:
                        print(f"Error processing message: {e}")
            finally:
                consumer.close()
                
    def _process_data_buffer(self):
        """Process data in the buffer at regular intervals"""
        print("Starting data processing thread")
        
        while self.running:
            # Wait for sliding_step amount of time
            time.sleep(5)  # Check buffer every 5 seconds
            
            with self.buffer_lock:
                buffer_size = len(self.buffer)
                
            if buffer_size >= self.window_size:
                print(f"Processing buffer with {buffer_size} records")
                self._run_processing_pipeline()
            else:
                print(f"Buffer has {buffer_size} records, waiting for {self.window_size}")
                
    def _run_processing_pipeline(self):
        """Run the full processing pipeline on buffered data"""
        # Get a copy of the current buffer
        with self.buffer_lock:
            # Take the most recent window_size records
            if len(self.buffer) > self.window_size:
                data = self.buffer[-self.window_size:]
                # Remove old records, keeping some overlap
                overlap = max(0, self.window_size - self.sliding_step)
                self.buffer = self.buffer[-(overlap):]
            else:
                data = self.buffer.copy()
                self.buffer = []
                
        # Convert to DataFrame
        try:
            df = pd.DataFrame(data)
        except Exception as e:
            print(f"Error creating DataFrame from buffer: {e}")
            print(f"Data sample: {data[:2] if data else None}")
            return
            
        # Make sure we have timestamp column
        if 'timestamp' not in df.columns:
            # Use current time
            df['timestamp'] = datetime.now()
            
        precise_start = time.perf_counter()  # More precise timing
        try:
            print(f"Features shape: {df.shape}")
            
            # Extract numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # Try using the full pipeline first
            try:
                # Use the full pipeline with a subset of methods
                feature_eng = TimeSeriesFeatureEngineering()
                feature_eng.load_data(df)
                
                feature_eng.generate_rolling_statistics(windows=[5, 10, 20])
                feature_eng.generate_lag_features(lags=[1, 5, 10])
                feature_eng.add_time_features()
                
                # Get features
                features_df = feature_eng.get_features()
                
            except Exception as e:
                print(f"Full pipeline failed: {e}, falling back to alternative version")
                
                # Create expanded set of features
                features = {}
                
                # Original values
                for col in numeric_cols:
                    features[col] = df[col]
                    
                    # Multiple rolling window statistics
                    windows = [5, 10, 20]
                    for window in windows:
                        if len(df) >= window:
                            features[f"{col}_roll_mean_{window}"] = df[col].rolling(window=window).mean()
                            features[f"{col}_roll_std_{window}"] = df[col].rolling(window=window).std()
                            features[f"{col}_roll_min_{window}"] = df[col].rolling(window=window).min()
                            features[f"{col}_roll_max_{window}"] = df[col].rolling(window=window).max()
                    
                    # Add lag features
                    for lag in [1, 5, 10]:
                        if lag < len(df):
                            features[f"{col}_lag_{lag}"] = df[col].shift(lag)
                            features[f"{col}_diff_{lag}"] = df[col].diff(lag)
                
                # Create features DataFrame with same index
                features_df = pd.DataFrame(features, index=df.index)
            
            # Fill NaN values
            features_df = features_df.fillna(0)
            
            # Track processing time
            precise_end = time.perf_counter()
            feature_time = precise_end - precise_start
            self.processing_times['feature_engineering'].append(feature_time)
            print(f"Feature engineering completed in {feature_time:.6f} seconds")
            print(f"Features shape: {features_df.shape}")
            print(f"Features calculated: {len(features_df.columns)}")
            
            # Publish engineered features to Kafka
            self._publish_features(features_df)
            
        except Exception as e:
            print(f"Error in feature engineering: {e}")
            import traceback
            traceback.print_exc()
            return
            
        # Anomaly Detection with improvements
        start_time = time.time()
        try:
            # Make predictions using trained models
            X = features_df.select_dtypes(include=[np.number])
            
            # Handle NaN values
            X = X.fillna(0)
            
            print(f"Data for prediction shape: {X.shape}")
            
            # Keep a sliding window of recent data for calibration
            if not hasattr(self, 'recent_data_buffer'):
                self.recent_data_buffer = X.copy()
            else:
                # Keep last 500 observations for calibration
                self.recent_data_buffer = pd.concat([self.recent_data_buffer, X]).tail(500)
            
            # Apply prediction with confidence and dynamic thresholds
            try:
                # Only start calibrating after collecting enough data
                calibration_data = self.recent_data_buffer if len(self.recent_data_buffer) > 100 else None
                
                # Check if we have enough processing history to make adjustments
                if hasattr(self, 'processing_count'):
                    self.processing_count += 1
                else:
                    self.processing_count = 1
                    
                # First few runs use standard prediction to establish baseline
                if self.processing_count < 5:
                    print(f"System warming up ({self.processing_count}/5), using standard prediction")
                    predictions = self.anomaly_detector.predict_anomalies(X)
                    ensemble_pred = self.anomaly_detector.ensemble_predictions(predictions, method='weighted_vote')
                else:
                    # Use advanced prediction with confidence after warm-up
                    predictions = self.anomaly_detector.predict_with_confidence(X, calibration_window=calibration_data)
                    ensemble_pred, ensemble_info = self.anomaly_detector.confidence_weighted_ensemble(predictions)
                    
                    # Track anomaly rate for monitoring
                    anomaly_rate = np.mean(ensemble_pred)
                    print(f"Current anomaly rate: {anomaly_rate:.2%}")
                    
                    # If anomaly rate is too high, increase threshold dynamically
                    if anomaly_rate > 0.3:  # More than 30% anomalies indicates miscalibration
                        print("Warning: High anomaly rate detected. Adjusting thresholds.")
                        for model_name in self.anomaly_detector.thresholds:
                            self.anomaly_detector.thresholds[model_name] *= 1.1  # Increase by 10%
            
            except Exception as e:
                print(f"Model-based prediction failed: {e}. Using threshold-based detection.")
                # Simple threshold-based detection as fallback
                thresholds = {}
                for col in numeric_cols:
                    thresholds[col] = np.mean(df[col]) + 3 * np.std(df[col])
                
                # Mark as anomaly if any value exceeds its threshold
                ensemble_pred = np.zeros(len(df))
                for col in numeric_cols:
                    ensemble_pred = np.logical_or(ensemble_pred, df[col] > thresholds[col])
                    ensemble_pred = np.logical_or(ensemble_pred, df[col] < (np.mean(df[col]) - 3 * np.std(df[col])))
                ensemble_pred = ensemble_pred.astype(int)
                
                # Create a minimalistic predictions dict
                predictions = {'threshold_detector': ensemble_pred}
            
            # Track processing time
            detection_time = time.time() - start_time
            self.processing_times['anomaly_detection'].append(detection_time)
            print(f"Anomaly detection completed in {detection_time:.2f} seconds")
            
            # Publish predictions to Kafka
            self._publish_predictions(df, features_df, ensemble_pred, predictions)
            
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            import traceback
            traceback.print_exc()
                
    def _publish_features(self, features_df):
        """Publish engineered features to Kafka"""
        producer = self.initialize_kafka_producer()
        
        # Convert features DataFrame to JSON-serializable format
        features_data = features_df.reset_index()
        
        # Convert timestamps to string if present
        if 'timestamp' in features_data.columns:
            features_data['timestamp'] = features_data['timestamp'].astype(str)
            
        # Send to Kafka as individual messages
        for _, row in features_data.iterrows():
            message = row.to_dict()
            self.produce_message(producer, self.feature_topic, message)
            
        if hasattr(producer, 'flush'):
            producer.flush()
            
    def _publish_predictions(self, original_df, features_df, ensemble_pred, model_predictions):
        """Publish anomaly predictions to Kafka"""

        if len(ensemble_pred) == 0:
            print("No predictions to publish - skipping")
            return
    
        producer = self.initialize_kafka_producer()
        
        # Get timestamps from original data
        timestamps = original_df['timestamp'].reset_index(drop=True)
        
        # Prepare messages
        for i, is_anomaly in enumerate(ensemble_pred):
            # Create a message with relevant info
            prediction = {
                'timestamp': str(timestamps[i]) if isinstance(timestamps[i], datetime) else timestamps[i],
                'is_anomaly': bool(is_anomaly),
                'ensemble_method': 'weighted_vote',
                'confidence': float(np.mean([predictions[i] for predictions in model_predictions.values() 
                                   if isinstance(predictions, np.ndarray)])),
                'model_predictions': {name: bool(predictions[i]) 
                                     for name, predictions in model_predictions.items()
                                     if not name.endswith('_prob') and not name.endswith('_score')}
            }
            
            # Add original sensor readings
            if i < len(original_df):
                row = original_df.iloc[i]
                for col in original_df.columns:
                    if col != 'timestamp' and col not in prediction:
                        prediction[f"sensor_{col}"] = float(row[col]) if np.isscalar(row[col]) else None
            
            # Send to Kafka
            self.produce_message(producer, self.prediction_topic, prediction)
            
        if hasattr(producer, 'flush'):
            producer.flush()
            
    def start_feature_consumer(self):
        """Start a consumer for the feature-engineered data"""
        self.running = True
        consumer_thread = threading.Thread(target=self._consume_features)
        consumer_thread.daemon = True
        consumer_thread.start()
        return consumer_thread
        
    def _consume_features(self):
        """Consume and handle feature-engineered data (for demo/monitoring)"""
        print(f"Starting consumer for topic: {self.feature_topic}")
        
        if self.use_confluent:
            consumer = self.initialize_kafka_consumer(None, f"{self.group_id}_features")
            consumer.subscribe([self.feature_topic])
            
            try:
                while self.running:
                    msg = consumer.poll(timeout=1.0)
                    if msg is None:
                        continue
                    if msg.error():
                        if msg.error().code() != KafkaError._PARTITION_EOF:
                            print(f"Error while consuming features: {msg.error()}")
                    else:
                        # Just print the feature data
                        print(f"Received feature data: {msg.value()[:100]}...")
            finally:
                consumer.close()
                
        else:
            consumer = self.initialize_kafka_consumer(self.feature_topic, f"{self.group_id}_features")
            
            try:
                for msg in consumer:
                    if not self.running:
                        break
                    # Just print the feature data
                    print(f"Received feature data: {str(msg.value)[:100]}...")
            finally:
                consumer.close()
                
    def start_prediction_consumer(self, callback=None):
        """
        Start a consumer for the anomaly predictions, with optional callback
        
        Parameters:
        -----------
        callback : callable, optional
            Function to call with each prediction message
        """
        self.running = True
        self.prediction_callback = callback
        consumer_thread = threading.Thread(target=self._consume_predictions)
        consumer_thread.daemon = True
        consumer_thread.start()
        return consumer_thread
        
    def _consume_predictions(self):
        """Consume and handle anomaly predictions"""
        print(f"Starting consumer for topic: {self.prediction_topic}")
        
        if self.use_confluent:
            consumer = self.initialize_kafka_consumer(None, f"{self.group_id}_predictions")
            consumer.subscribe([self.prediction_topic])
            
            try:
                while self.running:
                    msg = consumer.poll(timeout=1.0)
                    if msg is None:
                        continue
                    if msg.error():
                        if msg.error().code() != KafkaError._PARTITION_EOF:
                            print(f"Error while consuming predictions: {msg.error()}")
                    else:
                        # Process prediction
                        prediction = json.loads(msg.value().decode('utf-8'))
                        self._handle_prediction(prediction)
            finally:
                consumer.close()
                
        else:
            consumer = self.initialize_kafka_consumer(
                self.prediction_topic, f"{self.group_id}_predictions")
            
            try:
                for msg in consumer:
                    if not self.running:
                        break
                    self._handle_prediction(msg.value)
            finally:
                consumer.close()
                
    def _handle_prediction(self, prediction):
        """Handle a prediction message"""
        # Print anomaly predictions
        is_anomaly = prediction.get('is_anomaly', False)
        timestamp = prediction.get('timestamp', 'unknown')
        
        if is_anomaly:
            print(f"ANOMALY DETECTED at {timestamp}")
            # Print model votes
            model_votes = prediction.get('model_predictions', {})
            for model, vote in model_votes.items():
                print(f"  - {model}: {'ANOMALY' if vote else 'normal'}")
        else:
            print(f"Normal operation at {timestamp}")
            
        # Call the callback if provided
        if hasattr(self, 'prediction_callback') and self.prediction_callback:
            self.prediction_callback(prediction)
    def simulate_sensor_data(self, duration_seconds=300, interval_seconds=1):
            """
            Simulate sensor data and produce to Kafka
            
            Parameters:
            -----------
            duration_seconds : int
                How long to run the simulation in seconds
            interval_seconds : int
                Interval between messages in seconds
            """
            from data_simulation.sensor_data_simulator import SensorDataSimulator
            
            # Create a simulator
            simulator = SensorDataSimulator(
                start_date=datetime.now(),
                days=1,
                sample_rate=60
            )
            
            # Generate sensor data
            simulator.generate_all_sensor_data()
            
            # Add some anomalies
            simulator.add_spike_anomaly('temperature', 0.3, duration_ratio=0.005, amplitude_factor=4.0)
            simulator.add_drift_anomaly('pressure', 0.6, 0.8, drift_factor=1.5)
            simulator.add_intermittent_anomaly('vibration', 0.4, 0.5, frequency=0.3, amplitude_factor=3.0)
            
            # Get data as DataFrame
            df = simulator.to_dataframe()
            
            # Initialize Kafka producer
            producer = self.initialize_kafka_producer()
            
            # Calculate total messages to send
            total_messages = min(len(df), duration_seconds // interval_seconds)
            
            print(f"Starting sensor data simulation: {total_messages} messages over {duration_seconds} seconds")
            
            # Send each row to Kafka
            start_time = time.time()
            for i in range(total_messages):
                # Get the row as a dictionary
                row = df.iloc[i].to_dict()
                
                # Convert timestamp to string
                if 'timestamp' in row and isinstance(row['timestamp'], datetime):
                    row['timestamp'] = row['timestamp'].isoformat()
                    
                # Send to Kafka
                self.produce_message(producer, self.input_topic, row)
                
                # Sleep to control message rate
                elapsed = time.time() - start_time
                target_time = i * interval_seconds
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)
                    
                # Progress update
                if i % 10 == 0:
                    print(f"Sent {i}/{total_messages} messages")
                    
            # Make sure all messages are sent
            if hasattr(producer, 'flush'):
                producer.flush()
                
            print(f"Simulation complete: sent {total_messages} messages")
            
    def stop(self):
        """Stop all running threads"""
        self.running = False
        print("Stopping all processing threads...")
        time.sleep(2)  # Give threads time to clean up
        print("Processing pipeline stopped")
        
    def get_performance_stats(self):
        """Get performance statistics for the pipeline"""
        stats = {}
        
        for stage, times in self.processing_times.items():
            if times:
                stats[stage] = {
                    'mean_time': np.mean(times),
                    'max_time': np.max(times),
                    'min_time': np.min(times),
                    'std_time': np.std(times),
                    'total_processed': len(times)
                }
            else:
                stats[stage] = {
                    'mean_time': 0,
                    'max_time': 0,
                    'min_time': 0,
                    'std_time': 0,
                    'total_processed': 0
                }
                
        return stats
        
    def print_performance_stats(self):
        """Print performance statistics for the pipeline"""
        stats = self.get_performance_stats()
        
        print("\n===== Performance Statistics =====")
        for stage, metrics in stats.items():
            print(f"\n{stage.replace('_', ' ').title()}:")
            print(f"  Mean processing time: {metrics['mean_time']:.2f} seconds")
            print(f"  Max processing time: {metrics['max_time']:.2f} seconds")
            print(f"  Min processing time: {metrics['min_time']:.2f} seconds")
            print(f"  Standard deviation: {metrics['std_time']:.2f} seconds")
            print(f"  Total processed: {metrics['total_processed']}")
            
            if metrics['total_processed'] > 0:
                throughput = metrics['total_processed'] / sum(self.processing_times[stage])
                print(f"  Throughput: {throughput:.2f} items/second")
        
        # Overall pipeline stats
        if stats['feature_engineering']['total_processed'] > 0 and stats['anomaly_detection']['total_processed'] > 0:
            total_time = sum(self.processing_times['feature_engineering']) + sum(self.processing_times['anomaly_detection'])
            total_processed = stats['feature_engineering']['total_processed']
            print(f"\nOverall Pipeline:")
            print(f"  Total processing time: {total_time:.2f} seconds")
            print(f"  Total items processed: {total_processed}")
            print(f"  Overall throughput: {total_processed / total_time:.2f} items/second")
            
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.processing_times = {
            'feature_engineering': [],
            'anomaly_detection': []
        }
        print("Performance statistics reset")
        
    def run_demo(self, duration_seconds=300):
        """
        Run a complete demo of the real-time processing pipeline
        
        Parameters:
        -----------
        duration_seconds : int
            How long to run the demo in seconds
        """
        # Create Kafka topics
        self.create_topics()
        
        # Start consumers for features and predictions
        self.start_feature_consumer()
        self.start_prediction_consumer()
        
        # Start the data consumer and processing threads
        self.start_data_consumer()
        
        print("Pipeline started. Beginning sensor data simulation...")
        
        # Simulate sensor data
        self.simulate_sensor_data(duration_seconds=duration_seconds)
        
        # Wait until the simulation is complete plus some extra time for processing
        time.sleep(10)
        
        # Print performance statistics
        self.print_performance_stats()
        
        # Stop the pipeline
        self.stop()
        
# Example usage
def run_kafka_demo():
    # Check if Kafka is available
    import socket
    kafka_available = False
    try:
        # Try to connect to Kafka (assuming default port)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect(('localhost', 9092))
        s.close()
        kafka_available = True
    except:
        print("Kafka doesn't appear to be running on localhost:9092")
        
    if kafka_available:
        # Create and run the pipeline
        pipeline = RealTimeProcessingPipeline(
            bootstrap_servers='localhost:9092',
            window_size=60,
            sliding_step=30
        )
        
        # Load or train models
        try:
            pipeline.anomaly_detector.load_models()
            print("Successfully loaded anomaly detection models")
        except:
            print("No pre-trained models found. Training new models...")
            # Generate some training data
            from data_simulation.sensor_data_simulator import SensorDataSimulator
            from sklearn.model_selection import train_test_split
            
            simulator = SensorDataSimulator(
                start_date=datetime.now(),
                days=7,
                sample_rate=60
            )
            simulator.generate_all_sensor_data()
            
            # Add anomalies for training data
            simulator.add_spike_anomaly('temperature', 0.3, amplitude_factor=4.0)
            simulator.add_drift_anomaly('pressure', 0.6, 0.8, drift_factor=1.5)
            simulator.add_intermittent_anomaly('vibration', 0.4, 0.5, frequency=0.3)
            simulator.add_random_anomalies(count=5)
            
            # Get data as DataFrame
            df = simulator.to_dataframe()
            
            # Create feature engineering pipeline
            feature_eng = TimeSeriesFeatureEngineering()
            feature_eng.load_data(df)
            features_df = feature_eng.run_full_pipeline()
            
            # Create labels (simplified version - in real world this would come from labeled data)
            # Here we're using a simple threshold approach to create synthetic labels
            anomaly_labels = pd.Series(0, index=features_df.index)
            for col in ['temperature', 'pressure', 'vibration', 'power_consumption']:
                mean = df[col].mean()
                std = df[col].std()
                anomaly_labels = anomaly_labels | (df[col] > mean + 3*std) | (df[col] < mean - 3*std)
            
            # Split data for training
            X = features_df.select_dtypes(include=[np.number]).fillna(0)
            y = anomaly_labels
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Train models
            pipeline.anomaly_detector.train_isolation_forest(X_train, contamination=0.05)
            pipeline.anomaly_detector.train_supervised_rf(X_train, y_train)
            pipeline.anomaly_detector.train_supervised_xgboost(X_train, y_train)
            
            # Save models
            pipeline.anomaly_detector.save_models()
            
        # Run the demo
        pipeline.run_demo(duration_seconds=180)  # 3 minutes demo
    else:
        print("Demo requires Kafka to be running. Please start Kafka and try again.")
        print("You can run Kafka using Docker with: docker-compose up -d")

if __name__ == "__main__":
    run_kafka_demo()