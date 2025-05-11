import RPi.GPIO as GPIO
import time
import supabase
import datetime
import serial
import requests

# GPIO setup
GPIO.setmode(GPIO.BCM)

# Motor Driver Pins
ENA = 12
ENB = 13
IN1 = 5
IN2 = 6
IN3 = 20
IN4 = 21

# Ultrasonic Sensor Pins
TRIG_LEFT = 23
ECHO_LEFT = 24
TRIG_RIGHT = 18
ECHO_RIGHT = 27

# Soil Moisture Sensor Pin
SOIL_MOISTURE_PIN = 22

# Servo Pins
WEED_SERVO_PIN = 17
PEST_SERVO_PIN = 4

# GPS Serial
GPS_PIN = 15
gps = serial.Serial('/dev/ttyS0', 9600, timeout=1)

# Supabase credentials
url = "https://wrktsadpbrocmuyuztxl.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Indya3RzYWRwYnJvY211eXV6dHhsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDEzNjcwMzMsImV4cCI6MjA1Njk0MzAzM30.EDH66keNiSQB6wvsYJ-STKPQgIBRNj9QG7sAH0Xvm3s"
supabase_client = supabase.create_client(url, key)

# GPIO Setup
GPIO.setup([ENA, ENB], GPIO.OUT)
GPIO.setup([IN1, IN2, IN3, IN4], GPIO.OUT)
GPIO.setup(TRIG_LEFT, GPIO.OUT)
GPIO.setup(ECHO_LEFT, GPIO.IN)
GPIO.setup(TRIG_RIGHT, GPIO.OUT)
GPIO.setup(ECHO_RIGHT, GPIO.IN)
GPIO.setup(SOIL_MOISTURE_PIN, GPIO.IN)
GPIO.setup([WEED_SERVO_PIN, PEST_SERVO_PIN], GPIO.OUT)

# Servo & Motor setup
weed_servo = GPIO.PWM(WEED_SERVO_PIN, 50)
pest_servo = GPIO.PWM(PEST_SERVO_PIN, 50)
left_motor = GPIO.PWM(ENA, 100)
right_motor = GPIO.PWM(ENB, 100)

# Function to read soil moisture
def read_soil_moisture():
    return 0 if GPIO.input(SOIL_MOISTURE_PIN) == GPIO.LOW else 1

# Ultrasonic sensor reading
def read_distance(trigger_pin, echo_pin):
    GPIO.output(trigger_pin, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(trigger_pin, GPIO.LOW)

    pulse_start = time.time()
    while GPIO.input(echo_pin) == GPIO.LOW:
        pulse_start = time.time()
    while GPIO.input(echo_pin) == GPIO.HIGH:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    return pulse_duration * 17150  # in cm

# Trigger image capture from Flask
def trigger_image_capture_from_flask():
    try:
        response = requests.get("http://192.168.69.155:5000/capture_image")
        if response.status_code == 200:
            data = response.json()
            print("Image captured and uploaded via Flask.")
            return data.get("image_filename")
        else:
            print("Failed to capture image from Flask.")
            return None
    except Exception as e:
        print(f"Error contacting Flask server: {e}")
        return None

# Supabase upload
def send_data_to_supabase(level, image_filename=None, gps_coords=None):
    try:
        # Insert into current_moisture_level (use 'level' column)
        supabase_client.from_('current_moisture_level').insert({
            "level": level,
            "created_at": datetime.datetime.now().isoformat(),
            "species": "unknown"  # update dynamically if needed
        }).execute()
    except Exception as e:
        print("Error sending moisture to Supabase:", e)
    
    if image_filename:
        try:
            prediction_data = supabase_client.from_('prediction').select('*').eq('image_filename', image_filename).single().execute()
            if prediction_data['data']:
                pest_or_weed = prediction_data['data']['prediction']
                if gps_coords:
                    supabase_client.from_('coordinates').insert({
                        "longitude": gps_coords[0],
                        "latitude": gps_coords[1],
                        "created_at": datetime.datetime.now().isoformat()
                    }).execute()

                if pest_or_weed == 'pest':
                    move_servo(pest_servo, 90)
                    time.sleep(1)
                    move_servo(pest_servo, 0)
                elif pest_or_weed == 'weed':
                    move_servo(weed_servo, 90)
                    time.sleep(1)
                    move_servo(weed_servo, 0)
        except Exception as e:
            print("Error sending prediction to Supabase:", e)

# Servo control
def move_servo(servo, angle):
    duty = angle / 18 + 2
    servo.ChangeDutyCycle(duty)
    time.sleep(1)
    servo.ChangeDutyCycle(0)

# Main logic
try:
    weed_servo.start(0)
    pest_servo.start(0)
    left_motor.start(50)
    right_motor.start(50)

    while True:
        GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)
        time.sleep(5)

        level = read_soil_moisture()
        print(f"Moisture Level: {level}")
        send_data_to_supabase(level)

        left_distance = read_distance(TRIG_LEFT, ECHO_LEFT)
        right_distance = read_distance(TRIG_RIGHT, ECHO_RIGHT)

        if left_distance < 30 or right_distance < 30:
            image_filename = trigger_image_capture_from_flask()
            if image_filename:
                gps_coords = (gps.readline().decode().strip(), gps.readline().decode().strip())
                send_data_to_supabase(level, image_filename, gps_coords)

        GPIO.output([IN1, IN2, IN3, IN4], GPIO.HIGH)
        time.sleep(1)

finally:
    GPIO.cleanup()
    pest_servo.stop()
    weed_servo.stop()
    left_motor.stop()
    right_motor.stop()
