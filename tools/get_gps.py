import serial
import time
from datetime import datetime

PORT = 'COM16'
BAUDRATE = 9600

def main():
    try:
        with serial.Serial(PORT, BAUDRATE, timeout=0.05) as ser:
            print(f"Listening to GPS on {PORT}...")
            count = 0
            start_time = time.time()

            while True:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                now = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                if line.startswith('$GNR'):
                    print(f"[{now}] {line}")
                    count += 1

                # Every second, print how many messages were received
                if time.time() - start_time >= 1.0:
                    print(f"-> GPS messages this second: {count}")
                    count = 0
                    start_time = time.time()

    except serial.SerialException as e:
        print(f"Could not open serial port {PORT}: {e}")

if __name__ == "__main__":
    main()
