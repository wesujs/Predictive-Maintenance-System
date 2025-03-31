import os
import subprocess
import sys
import time
import signal
import atexit

def create_docker_compose_file():
    """Create a docker-compose.yml file for Kafka and Zookeeper"""
    docker_compose_content = """version: '3'
services:
  zookeeper:
    image: wurstmeister/zookeeper
    ports:
      - "2181:2181"
  
  kafka:
    image: wurstmeister/kafka
    ports:
      - "9092:9092"
    environment:
      KAFKA_ADVERTISED_HOST_NAME: localhost
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_CREATE_TOPICS: "sensor_data:3:1,engineered_features:3:1,anomaly_predictions:3:1"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
"""
    
    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose_content)
    
    print("Created docker-compose.yml file")

def check_docker_installed():
    """Check if Docker is installed"""
    try:
        subprocess.run(["docker", "--version"], check=True, stdout=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def check_docker_compose_installed():
    """Check if Docker Compose is installed"""
    try:
        subprocess.run(["docker-compose", "--version"], check=True, stdout=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def setup_kafka():
    """Setup and ensure Kafka is running for the application"""
    if not check_docker_installed():
        print("Docker is not installed. Please install Docker first.")
        return False
   
    if not check_docker_compose_installed():
        print("Docker Compose is not installed. Please install Docker Compose first.")
        return False
   
    if check_kafka_running():
        print("Kafka is already running on localhost:9092")
        return True
    else:
        return start_kafka()

def start_kafka():
    """Start Kafka using Docker Compose"""
    if not os.path.exists("docker-compose.yml"):
        create_docker_compose_file()
    
    try:
        # Start containers
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        print("Starting Kafka. This might take a moment...")
        time.sleep(10)
        
        # Check if Kafka is running
        result = subprocess.run(
            ["docker-compose", "ps", "kafka"], 
            check=True, 
            stdout=subprocess.PIPE, 
            text=True
        )
        
        if "Up" in result.stdout:
            print("Kafka is now running on localhost:9092")
            return True
        else:
            print("Kafka failed to start properly")
            return False
    
    except subprocess.SubprocessError as e:
        print(f"Error starting Kafka: {e}")
        return False

def stop_kafka():
    """Stop Kafka and Zookeeper containers"""
    try:
        print("Stopping Kafka and Zookeeper...")
        subprocess.run(["docker-compose", "down"], check=True)
        print("Kafka and Zookeeper stopped")
    except subprocess.SubprocessError as e:
        print(f"Error stopping Kafka: {e}")

def check_kafka_running():
    """Check if Kafka is running on localhost:9092"""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', 9092))
        sock.close()
        return result == 0 
    except:
        return False

if __name__ == "__main__":
    if not check_docker_installed():
        print("Docker is not installed. Please install Docker first.")
        sys.exit(1)
    
    if not check_docker_compose_installed():
        print("Docker Compose is not installed. Please install Docker Compose first.")
        sys.exit(1)
    
    if check_kafka_running():
        print("Kafka is already running on localhost:9092")
    else:
        if start_kafka():
            # Register the stop function to be called on exit
            atexit.register(stop_kafka)
            
            # Also handle SIGINT
            def signal_handler(sig, frame):
                stop_kafka()
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            
            print("Press Ctrl+C to stop Kafka and exit")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass