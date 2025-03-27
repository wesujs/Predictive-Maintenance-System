import numpy as np
import pandas as pd
from scipy import signal
import random
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json

class SensorDataSimulator:
    def __init__(self, 
                 start_date=datetime.now(),
                 days=7, 
                 sample_rate=60):  # Default: 1 reading per minute
        """
        Initialize the sensor data simulator.
        
        Parameters:
        -----------
        start_date : datetime
            The starting date for the simulation
        days : int
            Number of days to simulate
        sample_rate : int
            Seconds between readings
        """
        self.start_date = start_date
        self.days = days
        self.sample_rate = sample_rate
        
        # Calculate total number of data points
        self.total_points = int((days * 24 * 60 * 60) / sample_rate)
        
        # Create timestamp array
        self.timestamps = [start_date + timedelta(seconds=i*sample_rate) 
                           for i in range(self.total_points)]
        
        # Initialize empty dictionary to store sensor data
        self.sensor_data = {}
    def generate_custom_sensor_data(self, 
                                sensor_name,
                                mean,
                                amplitude,
                                day_amplitude=0,
                                week_amplitude=0,
                                noise_level=0.1,
                                unit=""):
        """
        Generate data for a custom sensor type.
        
        Parameters:
        -----------
        sensor_name : str
            Name for the custom sensor
        mean : float
            Mean value for the sensor
        amplitude : float
            General amplitude of variations
        day_amplitude : float
            Daily cycle amplitude
        week_amplitude : float
            Weekly cycle amplitude
        noise_level : float
            Noise level as a proportion of amplitude
        unit : str
            Unit of measurement for the sensor
        """
        # Generate base signal
        signal = self.generate_base_signal(mean, amplitude, noise_level)
        
        # Add daily cycle if specified
        if day_amplitude > 0:
            signal = self.add_daily_pattern(signal, day_amplitude)
        
        # Add weekly variation if specified
        if week_amplitude > 0:
            signal = self.add_weekly_pattern(signal, week_amplitude)
        
        # Store in the sensor data dictionary
        self.sensor_data[sensor_name] = signal
        
        # Add the unit to our units dictionary
        if unit:
            self._units[sensor_name] = unit
        
    def load_config_from_json(self, config_file):
        """
        Load sensor configuration from a JSON file.
        
        Parameters:
        -----------
        config_file : str
            Path to the JSON configuration file
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            # Generate sensors based on configuration
            for sensor_name, params in config.get('sensors', {}).items():
                if sensor_name == 'temperature':
                    self.generate_temperature_data(**params)
                elif sensor_name == 'vibration':
                    self.generate_vibration_data(**params)
                elif sensor_name == 'pressure':
                    self.generate_pressure_data(**params)
                elif sensor_name == 'power_consumption':
                    self.generate_power_consumption_data(**params)
                    
            # Apply anomalies if specified
            for anomaly in config.get('anomalies', []):
                anomaly_type = anomaly.get('type')
                sensor = anomaly.get('sensor')
                
                if anomaly_type == 'spike':
                    self.add_spike_anomaly(
                        sensor, 
                        anomaly.get('position_ratio', 0.5),
                        anomaly.get('duration_ratio', 0.002),
                        anomaly.get('amplitude_factor', 3.0)
                    )
                elif anomaly_type == 'drift':
                    self.add_drift_anomaly(
                        sensor,
                        anomaly.get('start_ratio', 0.6),
                        anomaly.get('end_ratio', 0.8),
                        anomaly.get('drift_factor', 2.0)
                    )
                elif anomaly_type == 'intermittent':
                    self.add_intermittent_anomaly(
                        sensor,
                        anomaly.get('start_ratio', 0.3),
                        anomaly.get('end_ratio', 0.4),
                        anomaly.get('frequency', 0.2),
                        anomaly.get('amplitude_factor', 2.0)
                    )
                    
            print(f"Successfully loaded configuration from {config_file}")
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
        
    def generate_base_signal(self, 
                            mean, 
                            amplitude, 
                            noise_level=0.1):
        """
        Generate a base signal with random noise.
        
        Parameters:
        -----------
        mean : float
            The center value of the signal
        amplitude : float
            The amplitude of variations
        noise_level : float
            Standard deviation of noise as a proportion of amplitude
            
        Returns:
        --------
        numpy.ndarray
            Array of signal values
        """
        # Generate random noise
        noise = np.random.normal(0, noise_level * amplitude, self.total_points)
        
        # Create base signal: just the mean value plus noise
        base_signal = np.ones(self.total_points) * mean + noise
        
        return base_signal

    def add_daily_pattern(self, signal, day_amplitude):
        """
        Add daily cyclical pattern to the signal.
        
        Parameters:
        -----------
        signal : numpy.ndarray
            Base signal to modify
        day_amplitude : float
            Amplitude of the daily cycle
            
        Returns:
        --------
        numpy.ndarray
            Modified signal with daily pattern
        """
        # Create time indices from 0 to 2π for each day
        days_indices = np.linspace(0, 2*np.pi*self.days, self.total_points)
        
        # Create daily pattern (24-hour cycle)
        hours_in_day = 24
        daily_pattern = day_amplitude * np.sin(days_indices * hours_in_day / (24))
        
        # Add pattern to signal
        return signal + daily_pattern

    def add_weekly_pattern(self, signal, week_amplitude):
        """
        Add weekly cyclical pattern to the signal.
        
        Parameters:
        -----------
        signal : numpy.ndarray
            Base signal to modify
        week_amplitude : float
            Amplitude of the weekly cycle
            
        Returns:
        --------
        numpy.ndarray
            Modified signal with weekly pattern
        """
        # Create time indices from 0 to 2π for the entire period
        days_indices = np.linspace(0, 2*np.pi*self.days/7, self.total_points)
        
        # Create weekly pattern
        weekly_pattern = week_amplitude * np.sin(days_indices)
        
        # Add pattern to signal
        return signal + weekly_pattern
        
    def generate_temperature_data(self, 
                                 mean=22.0,          # 22°C average
                                 amplitude=5.0,      # ±5°C variation
                                 day_amplitude=3.0,  # Daily cycle of ±3°C
                                 week_amplitude=1.0, # Weekly cycle of ±1°C
                                 noise_level=0.05):  # 5% noise
        """
        Generate realistic temperature sensor data.
        
        Parameters:
        -----------
        mean : float
            Mean temperature value in °C
        amplitude : float
            General amplitude of variations
        day_amplitude : float
            Daily cycle amplitude
        week_amplitude : float
            Weekly cycle amplitude
        noise_level : float
            Noise level as a proportion of amplitude
        """
        # Generate base signal
        signal = self.generate_base_signal(mean, amplitude, noise_level)
        
        # Add daily cycle (temperature rises during day, falls at night)
        signal = self.add_daily_pattern(signal, day_amplitude)
        
        # Add weekly variation (e.g., less activity on weekends)
        signal = self.add_weekly_pattern(signal, week_amplitude)
        
        # Store in the sensor data dictionary
        self.sensor_data['temperature'] = signal

    def generate_vibration_data(self, 
                               mean=0.5,            # 0.5g average
                               amplitude=0.3,       # ±0.3g variation
                               day_amplitude=0.2,   # Daily cycle
                               noise_level=0.2):    # 20% noise (vibration is noisy)
        """
        Generate realistic vibration sensor data.
        
        Parameters:
        -----------
        mean : float
            Mean vibration value in g
        amplitude : float
            General amplitude of variations
        day_amplitude : float
            Daily cycle amplitude
        noise_level : float
            Noise level as a proportion of amplitude
        """
        # Generate base signal - vibration is typically more noisy
        signal = self.generate_base_signal(mean, amplitude, noise_level)
        
        # Add daily cycle (more activity during working hours)
        signal = self.add_daily_pattern(signal, day_amplitude)
        
        # Store in the sensor data dictionary
        self.sensor_data['vibration'] = signal

    def generate_pressure_data(self, 
                              mean=101.3,         # 101.3 kPa (atmospheric pressure)
                              amplitude=1.0,      # ±1 kPa variation
                              day_amplitude=0.2,  # Small daily cycle
                              noise_level=0.03):  # 3% noise
        """
        Generate realistic pressure sensor data.
        
        Parameters:
        -----------
        mean : float
            Mean pressure value in kPa
        amplitude : float
            General amplitude of variations
        day_amplitude : float
            Daily cycle amplitude
        noise_level : float
            Noise level as a proportion of amplitude
        """
        # Generate base signal
        signal = self.generate_base_signal(mean, amplitude, noise_level)
        
        # Add subtle daily cycle
        signal = self.add_daily_pattern(signal, day_amplitude)
        
        # Store in the sensor data dictionary
        self.sensor_data['pressure'] = signal

    def generate_power_consumption_data(self, 
                                      mean=75.0,          # 75 kW average
                                      amplitude=25.0,     # ±25 kW variation
                                      day_amplitude=20.0, # Strong daily cycle
                                      week_amplitude=15.0,# Strong weekly cycle
                                      noise_level=0.1):   # 10% noise
        """
        Generate realistic power consumption sensor data.
        
        Parameters:
        -----------
        mean : float
            Mean power consumption in kW
        amplitude : float
            General amplitude of variations
        day_amplitude : float
            Daily cycle amplitude
        week_amplitude : float
            Weekly cycle amplitude
        noise_level : float
            Noise level as a proportion of amplitude
        """
        # Generate base signal
        signal = self.generate_base_signal(mean, amplitude, noise_level)
        
        # Add strong daily cycle (more power during working hours)
        signal = self.add_daily_pattern(signal, day_amplitude)
        
        # Add weekly variation (less power on weekends)
        signal = self.add_weekly_pattern(signal, week_amplitude)
        
        # Ensure power consumption doesn't go below a minimum threshold
        minimum_power = mean * 0.2  # 20% of mean as minimum
        signal = np.maximum(signal, minimum_power)
        
        # Store in the sensor data dictionary
        self.sensor_data['power_consumption'] = signal
        
    def add_spike_anomaly(self, 
                         sensor_name, 
                         position_ratio, 
                         duration_ratio=0.002, 
                         amplitude_factor=3.0):
        """
        Add a spike anomaly to a sensor reading.
        
        Parameters:
        -----------
        sensor_name : str
            The name of the sensor to modify
        position_ratio : float
            Where to place the anomaly (0-1 ratio of total duration)
        duration_ratio : float
            How long the anomaly lasts (ratio of total duration)
        amplitude_factor : float
            How severe the spike is (multiplier of normal amplitude)
        """
        if sensor_name not in self.sensor_data:
            raise ValueError(f"Sensor {sensor_name} does not exist")
        
        # Get the signal
        signal = self.sensor_data[sensor_name]
        
        # Determine anomaly position and duration
        position = int(position_ratio * self.total_points)
        duration = max(1, int(duration_ratio * self.total_points))
        
        # Calculate anomaly start and end indices
        start_idx = max(0, position - duration // 2)
        end_idx = min(self.total_points, position + duration // 2)
        
        # Determine the mean and amplitude of the signal
        signal_mean = np.mean(signal)
        signal_amplitude = np.std(signal)
        
        # Create the spike
        for i in range(start_idx, end_idx):
            # Gradually increase and then decrease for a more natural spike
            distance_from_center = abs(i - position) / (duration / 2)
            if distance_from_center <= 1:
                # Parabolic shape for the spike
                spike_factor = amplitude_factor * (1 - distance_from_center**2)
                signal[i] += spike_factor * signal_amplitude
        
        # Update the sensor data
        self.sensor_data[sensor_name] = signal

    def add_drift_anomaly(self, 
                         sensor_name, 
                         start_ratio, 
                         end_ratio, 
                         drift_factor=2.0):
        """
        Add a gradual drift anomaly to a sensor reading.
        
        Parameters:
        -----------
        sensor_name : str
            The name of the sensor to modify
        start_ratio : float
            Where the drift begins (0-1 ratio of total duration)
        end_ratio : float
            Where the drift ends (0-1 ratio of total duration)
        drift_factor : float
            How severe the drift is (multiplier of normal amplitude)
        """
        if sensor_name not in self.sensor_data:
            raise ValueError(f"Sensor {sensor_name} does not exist")
        
        # Get the signal
        signal = self.sensor_data[sensor_name]
        
        # Determine drift start and end positions
        start_idx = int(start_ratio * self.total_points)
        end_idx = int(end_ratio * self.total_points)
        
        # Calculate drift duration
        drift_duration = end_idx - start_idx
        
        if drift_duration <= 0:
            raise ValueError("End ratio must be greater than start ratio")
        
        # Determine the amplitude of the signal
        signal_amplitude = np.std(signal)
        
        # Create the drift
        for i in range(start_idx, end_idx):
            # Calculate how far into the drift period we are (0 to 1)
            progress = (i - start_idx) / drift_duration
            # Apply gradual drift
            signal[i] += progress * drift_factor * signal_amplitude
        
        # Update the sensor data
        self.sensor_data[sensor_name] = signal

    def add_intermittent_anomaly(self, 
                                sensor_name, 
                                start_ratio, 
                                end_ratio, 
                                frequency=0.2, 
                                amplitude_factor=2.0):
        """
        Add intermittent failures/spikes to a sensor reading.
        
        Parameters:
        -----------
        sensor_name : str
            The name of the sensor to modify
        start_ratio : float
            Where the anomalies begin (0-1 ratio of total duration)
        end_ratio : float
            Where the anomalies end (0-1 ratio of total duration)
        frequency : float
            How often the anomalies occur (probability 0-1)
        amplitude_factor : float
            How severe the anomalies are (multiplier of normal amplitude)
        """
        if sensor_name not in self.sensor_data:
            raise ValueError(f"Sensor {sensor_name} does not exist")
        
        # Get the signal
        signal = self.sensor_data[sensor_name]
        
        # Determine anomaly start and end positions
        start_idx = int(start_ratio * self.total_points)
        end_idx = int(end_ratio * self.total_points)
        
        # Calculate the signal properties
        signal_mean = np.mean(signal)
        signal_amplitude = np.std(signal)
        
        # Create intermittent anomalies
        for i in range(start_idx, end_idx):
            # Randomly apply anomalies based on frequency
            if random.random() < frequency:
                # Random sign for the anomaly
                sign = 1 if random.random() > 0.5 else -1
                # Apply the anomaly
                signal[i] += sign * amplitude_factor * signal_amplitude * random.random()
        
        # Update the sensor data
        self.sensor_data[sensor_name] = signal
    
    def to_dataframe(self):
        """
        Convert the simulated sensor data to a pandas DataFrame.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with timestamps and sensor readings
        """
        # Create a DataFrame with timestamps
        df = pd.DataFrame({'timestamp': self.timestamps})
        
        # Add each sensor's data
        for sensor_name, values in self.sensor_data.items():
            df[sensor_name] = values
            
        return df

    def to_csv(self, filename='sensor_data.csv'):
        """
        Export the simulated sensor data to a CSV file.
        
        Parameters:
        -----------
        filename : str
            Name of the output file
        """
        df = self.to_dataframe()
        df.to_csv(filename, index=False)
        print(f"Data exported to {filename}")

    def plot_sensors(self, figsize=(15, 10)):
        """
        Plot the simulated sensor data.
        
        Parameters:
        -----------
        figsize : tuple
            Size of the figure (width, height)
        """
        # Create a DataFrame for easier plotting
        df = self.to_dataframe()
        
        # Calculate number of subplots needed
        n_sensors = len(self.sensor_data)
        
        # Create the figure and axes
        fig, axes = plt.subplots(n_sensors, 1, figsize=figsize, sharex=True)
        
        # If there's only one sensor, axes won't be an array
        if n_sensors == 1:
            axes = [axes]
        
        # Plot each sensor
        for i, sensor_name in enumerate(self.sensor_data.keys()):
            axes[i].plot(df['timestamp'], df[sensor_name])
            axes[i].set_title(f'{sensor_name.replace("_", " ").title()} Sensor Data')
            axes[i].set_ylabel(self._get_sensor_unit(sensor_name))
            axes[i].grid(True)
        
        # Set the x-axis label for the bottom subplot
        axes[-1].set_xlabel('Time')
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()
        
    def _get_sensor_unit(self, sensor_name):
        """
        Get the appropriate unit for a sensor type.
        
        Parameters:
        -----------
        sensor_name : str
            The name of the sensor
            
        Returns:
        --------
        str
            The unit for the sensor
        """
        units = {
            'temperature': '°C',
            'vibration': 'g',
            'pressure': 'kPa',
            'power_consumption': 'kW'
        }
        return units.get(sensor_name, '')
    
    def create_api_endpoint(self, port=5000):
        """
        Create a simple Flask API endpoint to stream sensor data.
        Note: This requires Flask to be installed.
        
        Parameters:
        -----------
        port : int
            Port to run the API on
        """
        try:
            from flask import Flask, jsonify
            import threading
            
            app = Flask(__name__)
            
            # Convert data to DataFrame once
            df = self.to_dataframe()
            
            @app.route('/api/sensors', methods=['GET'])
            def get_all_data():
                """Return all sensor data."""
                return jsonify(df.to_dict(orient='records'))
                
            @app.route('/api/sensors/latest', methods=['GET'])
            def get_latest_data():
                """Return only the latest sensor reading."""
                return jsonify(df.iloc[-1].to_dict())
                
            @app.route('/api/sensors/<sensor_name>', methods=['GET'])
            def get_sensor_data(sensor_name):
                """Return data for a specific sensor."""
                if sensor_name not in self.sensor_data:
                    return jsonify({"error": f"Sensor {sensor_name} not found"}), 404
                    
                return jsonify(df[['timestamp', sensor_name]].to_dict(orient='records'))
            
            # Run the API in a separate thread
            threading.Thread(target=lambda: app.run(port=port, debug=False)).start()
            
            print(f"API running on http://localhost:{port}/api/sensors")
            
        except ImportError:
            print("Flask is not installed. To use this feature, run: pip install flask")
            
    def generate_all_sensor_data(self):
        """
        Generate data for all sensor types with default parameters.
        """
        self.generate_temperature_data()
        self.generate_vibration_data()
        self.generate_pressure_data()
        self.generate_power_consumption_data()
        
        print(f"Generated {self.total_points} data points for each sensor type.")

    def add_random_anomalies(self, count=3):
        """
        Add random anomalies to the generated sensor data.
        
        Parameters:
        -----------
        count : int
            Number of anomalies to add per sensor
        """
        if not self.sensor_data:
            raise ValueError("Generate sensor data before adding anomalies")
        
        anomaly_types = [
            self.add_spike_anomaly,
            self.add_drift_anomaly,
            self.add_intermittent_anomaly
        ]
        
        for sensor_name in self.sensor_data.keys():
            for _ in range(count):
                # Choose a random anomaly type
                anomaly_func = random.choice(anomaly_types)
                
                if anomaly_func == self.add_spike_anomaly:
                    # Add a spike at a random position
                    position = random.uniform(0.1, 0.9)
                    anomaly_func(sensor_name, position)
                    
                elif anomaly_func == self.add_drift_anomaly:
                    # Add a drift in a random segment
                    start = random.uniform(0.2, 0.7)
                    end = start + random.uniform(0.1, 0.3)
                    anomaly_func(sensor_name, start, end)
                    
                elif anomaly_func == self.add_intermittent_anomaly:
                    # Add intermittent anomalies in a random segment
                    start = random.uniform(0.2, 0.7)
                    end = start + random.uniform(0.1, 0.3)
                    anomaly_func(sensor_name, start, end)
        
        print(f"Added {count} random anomalies to each sensor.")


# Example usage of the SensorDataSimulator
if __name__ == "__main__":
    # Create a simulator for 7 days of data, with readings every minute
    start_date = datetime(2025, 3, 1)
    simulator = SensorDataSimulator(start_date=start_date, days=7, sample_rate=60)
    
    # Generate data for all sensors
    simulator.generate_all_sensor_data()
    
    # Add some specific anomalies
    # 1. Add a temperature spike at around 40% of the time period
    simulator.add_spike_anomaly('temperature', 0.4, amplitude_factor=4.0)
    
    # 2. Add a pressure drift starting at 60% and ending at 80% of the time period
    simulator.add_drift_anomaly('pressure', 0.6, 0.8, drift_factor=1.5)
    
    # 3. Add intermittent anomalies to vibration between 30% and 40% of the time period
    simulator.add_intermittent_anomaly('vibration', 0.3, 0.4, frequency=0.3)
    
    # 4. Add random anomalies to all sensors (3 per sensor by default)
    simulator.add_random_anomalies()
    
    # Plot the data to visualize the patterns and anomalies
    simulator.plot_sensors()
    
    # Export the data to CSV
    simulator.to_csv('industrial_sensor_data.csv')
    
    # Optional: Start a REST API to serve the data
    # Uncomment the line below if Flask is installed
    # simulator.create_api_endpoint()
    
    # Example of accessing the data programmatically
    df = simulator.to_dataframe()
    
    # Print some basic statistics
    print("\nSensor Data Statistics:")
    for column in df.columns:
        if column != 'timestamp':
            print(f"\n{column.replace('_', ' ').title()}:")
            print(f"  Mean: {df[column].mean():.2f} {simulator._get_sensor_unit(column)}")
            print(f"  Min: {df[column].min():.2f} {simulator._get_sensor_unit(column)}")
            print(f"  Max: {df[column].max():.2f} {simulator._get_sensor_unit(column)}")
            print(f"  Std Dev: {df[column].std():.2f} {simulator._get_sensor_unit(column)}")