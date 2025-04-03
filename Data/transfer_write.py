import serial
import serial.tools.list_ports
import struct
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt

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

# num_bytes = 12
# r = 0
# prev = [None]

# data = None

ser.write(struct.pack("ciiiic", b'S', 0, 0, 0, 0, b'E'))
ser.flush()
print("Sent")
# while True:
#     print(ser.read_until(size=1))

# Initial positions
pos1 = 0
pos2 = 0
pos3 = 0

# Create a figure and a set of subplots
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

# Create sliders
ax_pos1 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_pos2 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_pos3 = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor='lightgoldenrodyellow')

motor_max_pos = 8000

slider_pos1 = Slider(ax_pos1, 'Pos1', 0, motor_max_pos, valinit=pos1)
slider_pos2 = Slider(ax_pos2, 'Pos2', 0, motor_max_pos, valinit=pos2)
slider_pos3 = Slider(ax_pos3, 'Pos3', 0, motor_max_pos, valinit=pos3)

def update(val):
    pos1 = int(slider_pos1.val)
    pos2 = int(slider_pos2.val)
    pos3 = int(slider_pos3.val)
    ser.write(struct.pack("ciiiic", b'S', pos1, pos2, pos3, 0, b'E'))
    ser.flush()
    print(f"Sent: Pos1={pos1}, Pos2={pos2}, Pos3={pos3}")

slider_pos1.on_changed(update)
slider_pos2.on_changed(update)
slider_pos3.on_changed(update)

plt.show()

# while True:
#     ser.write(b'\n')
#     data = ser.read_until(size=num_bytes+2)
#     if len(data) == 14:
#         results = struct.unpack("LLLcc", data)
#         print(results)
#         if results[0] == prev[0]:
#             r += 1
#         else:
#             r = 0
#         # print(r)
#         prev = results
#     else:
#         ser.reset_input_buffer()
#         print("reset")
    

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
