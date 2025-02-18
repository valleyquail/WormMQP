from Nodes import *
from Arm import *
from SensorEdge import *

import numpy as np
from multiprocessing import shared_memory as shm

try:
    data = shm.SharedMemory(name='arm_data')
except:
    print("No data found")
    quit()

numpy_faces = np.ndarray(shape=(36,),buffer=data.buf)
print(numpy_faces)