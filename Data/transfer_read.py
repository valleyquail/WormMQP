import serial
import serial.tools.list_ports
import struct

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
baud_rate = 500000  

# Open the serial port
ser = serial.Serial(serial_port, baud_rate,timeout=None)

num_bytes = 12
r = 0
prev = [None]

data = None
while True:
    ser.write(b'\n')
    data = ser.read_until(size=num_bytes+2)
    if len(data) == 14:
        results = struct.unpack("LLLcc", data)
        print(results)
        if results[0] == prev[0]:
            r += 1
        else:
            r = 0
        # print(r)
        prev = results
    else:
        ser.reset_input_buffer()
        print("reset")
    

    # print(data[:])
    # integer_value = int.from_bytes(data[4:8], byteorder='little')
    # print(f"Integer value: {integer_value}")

    # if ser.in_waiting:
    #     data = ser.read(1)
    #     if data == b'\r':
    #         ser.read(1)
    #         i == 0
    #     else:
    #         buffer[i] = ord(data)
    #         i += 1
    #     if i == 4:
    #         integer_value = int.from_bytes(buffer)#, byteorder='little')
    #         print(f"Integer value: {integer_value}")
    #         i = 0


    #     # Convert binary data to integer
    #     integer_value = int.from_bytes(data, byteorder='little')
    #     print(f"Integer value: {integer_value}")
