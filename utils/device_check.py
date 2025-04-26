import samna

# Using the device controller we can retrieve a list of the supported devices
# currently connected to the system
devices = samna.device.get_unopened_devices()

# We can select a device based on the type of device, or serial number
# If we only have 1 device connected we can simply choose that one
if len(devices) == 0:
    raise Exception("No devices found")

chosen_board = devices[0]

# open the board and give it the name "my_board"
my_board = samna.device.open_device(chosen_board)

print(f"Opened device: {my_board}")