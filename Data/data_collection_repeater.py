import subprocess
import time
import os

n = 5  # Number of times to run the script
delay = 30  # Delay in seconds
script_dir = os.path.dirname(__file__)
venv_path = "/home/c/WormMQP/.venv/bin/python"
script_path = os.path.join(script_dir, "data_collection_v2.py")

for _ in range(n):
    subprocess.run([venv_path, script_path])
    time.sleep(delay)