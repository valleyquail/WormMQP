import serial
import csv
import time

# Configure the serial port
import serial.tools.list_ports

# List available serial ports
ports = serial.tools.list_ports.comports()
available_ports = [port.device for port in ports]

print("Available serial ports:")
for i, port in enumerate(available_ports):
    print(f"{i}: {port}")

# Prompt user to select a serial port
port_index = int(input("Select a serial port by index: "))
serial_port = available_ports[port_index]
baud_rate = 115200  # Replace with your baud rate

# Open the serial port
ser = serial.Serial(serial_port, baud_rate)

# Open the CSV file for writing
timestamp = time.strftime('%Y%m%d_%H%M%S')
csv_file = open(f'data_{timestamp}.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)

# Write the header to the CSV file
csv_writer.writerow(['Timestamp', 'Data'])

try:
    while True:
        # Read a line from the serial port
        line = ser.readline().decode('utf-8').strip()
        
        # Get the current timestamp
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Write the data to the CSV file
        csv_writer.writerow([timestamp, line])
        print(f'{timestamp}, {line}')
        
except serial.SerialException:
    print("Data collection stopped.")

finally:
    # Close the serial port and CSV file
    ser.close()
    csv_file.close()