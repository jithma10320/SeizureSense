import time
import board
import busio
import adafruit_adxl34x
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fft import fft
import joblib
import RPi.GPIO as GPIO
import serial


phone_number=+94765570719
# Initialize the sensor
try:
    i2c = busio.I2C(board.SCL, board.SDA)
    accelerometer = adafruit_adxl34x.ADXL345(i2c)
except ValueError as e:
    print(f"Error: {e}")
    print("Make sure the ADXL345 device is connected to the I2C bus.")
    exit(1)

# Initialize GSM module
try:
    gsm_serial = serial.Serial(port="/dev/serial0", baudrate=9600, timeout=1)  
    gsm_serial.write(b"AT\r\n")
    time.sleep(1)
    if "OK" not in gsm_serial.read_all().decode():
        raise Exception("GSM module not responding.")
except Exception as e:
    print(f"Error initializing GSM module: {e}")
    exit(1)

# Function to send AT commands
def send_at_command(command, delay=1):
    gsm_serial.write((command + "\r\n").encode())
    time.sleep(delay)
    return gsm_serial.read_all().decode()

# Function to make a phone call
def make_call(phone_number):
    try:
        print(f"Dialing {phone_number}...")
        response = send_at_command(f"ATD{phone_number};")
        if "OK" in response or "CONNECT" in response:
            print("Call initiated successfully.")
        else:
            print("Error: Failed to initiate call.")
    except Exception as e:
        print(f"Error during call: {e}")

# Function to hang up call
def hang_up_call():
    try:
        print("Hanging up...")
        send_at_command("ATH")
    except Exception as e:
        print(f"Error ending call: {e}")

# Parameters
data_size = 500
sampling_interval = 30.06 / data_size  # Interval in seconds between each data point

# Load the trained model
model = joblib.load('seizure_detection_model.pkl')

# Setup for the buzzer
BUZZER_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

# Function to activate buzzer
def activate_buzzer(duration=10):
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(duration)
    GPIO.output(BUZZER_PIN, GPIO.LOW)

# Function to extract features from the accelerometer data
def extract_features(data):
    features = {}
    
    for axis in ['gyro_x', 'gyro_y', 'gyro_z']:
        values = data[axis].values
        
        features[f'{axis}_mean'] = np.mean(values)
        features[f'{axis}_std'] = np.std(values)
        features[f'{axis}_min'] = np.min(values)
        features[f'{axis}_max'] = np.max(values)
        features[f'{axis}_range'] = np.max(values) - np.min(values)
        features[f'{axis}_variance'] = np.var(values)
        features[f'{axis}_skew'] = skew(values)
        features[f'{axis}_kurtosis'] = kurtosis(values)
        features[f'{axis}_rms'] = np.sqrt(np.mean(values**2))
        
        fft_values = fft(values)
        fft_magnitude = np.abs(fft_values)
        
        features[f'{axis}_dominant_freq'] = np.argmax(fft_magnitude[1:]) + 1 
        features[f'{axis}_spectral_entropy'] = -np.sum((fft_magnitude*2) * np.log(fft_magnitude*2 + 1e-10)) / len(fft_magnitude)
        features[f'{axis}_energy'] = np.sum(fft_magnitude**2)
    
    return features

# Infinite loop to collect data continuously
while True:
    # Lists to store x, y, z data
    x_data = []
    y_data = []
    z_data = []
    timestamp_data = []
    timecard = 0

    # Collect 500 data points over 30 seconds
    for _ in range(data_size):
        x, y, z = accelerometer.acceleration
        timecard += sampling_interval
        
        x_data.append(x)
        y_data.append(y)
        z_data.append(z)
        timestamp_data.append(timecard)
        time.sleep(sampling_interval)

    # Convert lists to DataFrame
    data = pd.DataFrame({
        'timestamp': timestamp_data,
        'gyro_x': x_data,
        'gyro_y': y_data,
        'gyro_z': z_data
    })

    # Save the collected data to a CSV file (timestamp-based naming)
    timestamp = int(time.time())
    data.to_csv(f'data/accelerometer_data_{timestamp}.csv', index=False)
    print(f"Data saved to data/accelerometer_data_{timestamp}.csv")

    time.sleep(2) 
    
    # Load the latest data and extract features 
    new_data = pd.read_csv(f'data/accelerometer_data_{timestamp}.csv')  
    new_data.columns = ["timestamp", "gyro_x", "gyro_y", "gyro_z"]  

    # Extract features from the new data
    new_features = extract_features(new_data)
    new_features_df = pd.DataFrame([new_features])  

    # Make a prediction using the trained model
    prediction = model.predict(new_features_df)

    if prediction[0] == 1:
        print("Seizure detected")
        activate_buzzer()
        make_call("+94771234567")  # Replace with the recipient's phone number
        hang_up_call()
    else:
        print("No seizure detected")