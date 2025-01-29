import serial
import csv
import time
import pyperclip

# Configure the serial port
import serial.tools.list_ports

# List available serial ports
ports = serial.tools.list_ports.comports()
available_ports = [port.device for port in ports]

ports = 0
print("Available serial ports:")
for i, port in enumerate(available_ports):
    print(f"{i}: {port}")
    ports += 1

if ports == 0:
    print("No serial ports available. Exiting program.")
    exit()

if ports == 1:
    port_index = 0
else:
    # Prompt user to select a serial port
    port_index = int(input("Select a serial port by index: "))
serial_port = available_ports[port_index]
baud_rate = 500000  # Replace with your baud rate

# Open the serial port
ser = serial.Serial(serial_port, baud_rate,timeout=1)
# ser = Serial(port='COM1', baudrate=115200, timeout=1, writeTimeout=1)
ser.set_buffer_size(rx_size = 12800, tx_size = 12800)

# Open the CSV file for writing
timestamp = time.strftime('%m%d_%H%M%S')
print(f"Data collection started. Saving data to data_{timestamp}.csv")
# Data\raw_data\data_20250123_160441.csv
csv_file = open(f'C:/Users/camca/WormMQP/Data/raw_data/data_{timestamp}.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)

# Write the header to the CSV file
csv_writer.writerow(['Index', 'Data'])
i = 0


# Request cycles
cycles = 100
ser.write(f"cycles={cycles}\n".encode('utf-8'))


try:
    while True:
        # Read a line from the serial port
        line = ser.readline().decode('utf-8').strip()

        # ser.flushInput()
        
        # Get the current timestamp (not needed; time from teensy)
        # timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Write the data to the CSV file
        csv_writer.writerow([i, line])
        print(line)
        # print(ser.in_waiting)
        i += 1
        
except serial.SerialException:
    print("Data collection stopped.")
    print(f"Saving data to data_{timestamp}.csv")
    pyperclip.copy(f'data_{timestamp}.csv')
    print("File path copied to clipboard.")

finally:
    # Close the serial port and CSV file
    ser.close()
    csv_file.close()




    # # csv_file = open(f'C:/Users/camca/WormMQP/Data/raw_data/data_{timestamp}.csv', 'r', newline='')
    # # csv_reader = csv.reader(csv_file)

    # # # Open a new CSV file for writing formatted data
    # # formatted_csv_file = open(f'C:/Users/camca/WormMQP/Data/raw_data/formatted_data_{timestamp}.csv', 'w', newline='')
    # # csv_writer = csv.writer(formatted_csv_file)

    # # # Write the header to the new CSV file
    # # csv_writer.writerow(['Index', 'Data'])

    # # # Iterate through the lines of the original CSV file and write to the new CSV file
    # # i = 0
    # # for row in csv_reader:
    # #     if i > 0:  # Skip the header row
    # #         csv_writer.writerow([i - 1, row[1].decode('utf-8').strip()])
    # #     i += 1

    # # print("Data formatted")
    # # formatted_csv_file.close()
    # csv_file.close()