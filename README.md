# Predictive Maintenance System

## Project Overview

This project aims to create an end-to-end predictive maintenance system that makes use of machine learning models to forecast equipment failures before they happen. The system will process sensor data from industrial equipment, indentify patterns and anomalies, and provide maintenance recommendations through an intuitive online dashboard.

This project comes from the urge to increase my knowledge in machine learning, data science, and software engineering as a whole. It also seems like it could be a good investment for people starting businesses to have once the features, data implemented, and accuracy are all fine-tuned.

## Completed Components ✅

### Sensor Data Simulator
A Python-based tool that generates realistic industrial sensor data with configurable parameters:
- Multiple sensor types (temperature, vibration, pressure, power consumption)
- Normal operational patterns with appropriate noise and variance
- Seasonal and cyclical variations
- Configurable anomaly patterns (gradual degradation, sudden failures, intermittent issues)
- Time-stamped output in multiple formats (CSV, JSON)
- Streaming API for integration with processing pipeline
- Visualization tools for data inspection and validation

### Anomaly Injection and Testing Framework
Inclusion of complex anomaly injection capabilities
- **Spike Anomalies**: Sudden, short-duration deviations representing power surges or momentary failures
- **Drift Anomalies**: Gradual changes over time that simulate sensor calibration issues or equipment degradation
- **Intermittent Anomalies**: Random, sporadic fluctuations that represent loose connections or interference

### Anomaly Detection Evaluation System
A testing framework to measure the performance of different anomaly detection algorithms
- Precision, recall, and F1 score calculation
- Visualization of detected vs. actual anomalies
- Comparative analysis of detection methods

### Generated Anomalies from Sensor Data Simulator
1. Simple Threshold Detector

This detector will establish static upper and lower boundaries based on the mean and standard   deviation of the entire dataset. It will flag values that exceed these set boundaries as anomalies.

![alt text](https://cdn.discordapp.com/attachments/1051835950902288435/1354638520701947985/Figure_1.png?ex=67e604ef&is=67e4b36f&hm=a0fe1431e49d732076f3cd942d5b1a5b5771ffd34289d7e670b9b6a17fcf6968& "Simple Threshold Detector")

2. Moving Average Detector

This detector compares the data point to a local moving average to identify points that deviate significantly from recent trends. This approach will be more adaptive when changing baseline conditions.

![alt text](https://media.discordapp.net/attachments/1051835950902288435/1354638543951237151/Figure_2.png?ex=67e604f4&is=67e4b374&hm=d82b4a208efc94aa525cc5252389e051b4eaf31fed1a69572934db6ee9e94908&=&format=webp&quality=lossless&width=1610&height=858 "Moving Aveg Detector")

- Z-Score Detector

This detector uses a robust statistical approach based on median values instead of mean. This reduces the influence of extreme values and can be better suited for detecting more subtle anomalies in noisy data.

![alt text](https://media.discordapp.net/attachments/1051835950902288435/1354638570698178570/Figure_3.png?ex=67e604fb&is=67e4b37b&hm=321e10a018b7823255f29c3f329dd0dcdfbc502fe32f950ca95ea0086b6d3b18&=&format=webp&quality=lossless&width=1610&height=858 "Z-Score Detector")
### Coming Soon 🔜

### Time Series Feature Engineering Pipeline
- Rolling statistics calculation
- Frequency domain feature extraction
- Lag and trend indicators
- Trend and seasonality decomposition
- Normalization and scaling

### Advanced Anomaly Detection & Predictive Models
- Multivariate anomaly detection using correlation between sensors
- Deep learning models (LSTM, autoencoder) for complex pattern recognition
- Ensemble methods combining multiple detection algorithms
- Remaining Useful Life (RUL) prediction models
- Confidence scoring and uncertainty quantification

### Real-Time Data Processing Pipeline
- Kafka-based message queue for sensor data ingestion
- Stream processing with windowing and aggregation
- Feature store for real-time feature computation
- Low-latency analytics for immediate anomaly detection

### Interactive Dashboard
- Equipment health overview with status indicators
- Detailed sensor visualizations with anomaly highlighting
- Prediction timeline with confidence intervals
- Maintenance recommendation interface
- Historical performance tracking and trend analysis

### Model Deployment & Monitoring
- RESTful API for predictions and inferences
- Model versioning and A/B testing framework
- Continuous performance tracking with alerts
- Concept drift detection and automatic retraining triggers

### Maintenance Optimization System
- Cost-benefit analysis of maintenance actions
- Optimized scheduling algorithms considering dependencies
- Resource allocation and inventory management
- Feedback loop for maintenance effectiveness evaluation

## Technical Stack

- **Languages:** Python, JavaScript
- **Data Processing:** Apache Kafka, Apache Spark
- **Machine Learning:** Scikit-learn, PyTorch, MLflow
- **Backend:** FastAPI
- **Frontend:** React, D3.js
- **Deployment:** Docker, GitHub Actions

## Getting Started

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- Node.js 16+

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/predictive-maintenance.git
cd predictive-maintenance

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the sensor data simulator
python src/data_generation/simulator.py
```

## Project Roadmap

1. **Phase 1 (Completed):** Data generation, anomaly injection, and detection evaluation
2. **Phase 2 (Current):** Data processing pipeline implementation
3. **Phase 3:** Machine learning model development
4. **Phase 4:** Dashboard and advanced visualization creation
5. **Phase 5:** System integration and deployment
6. **Phase 6:** Optimization and additional features

## Contributing

This project is currently under active development. Contributions, suggestions, and feedback are welcome. Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project aims to provide an open-source program for new companies to monitor their equipment and prevent long-term issues.
- Special thanks to the open-source community for the tools and libraries that make this project possible
