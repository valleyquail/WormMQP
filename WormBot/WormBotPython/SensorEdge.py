import numpy as np
import matplotlib.pyplot as plt
from Nodes import SensorNode

class SensorEdge:

    def __init__(self, node1: SensorNode, node2: SensorNode, sensorID: int):
        self.node1 = node1
        self.node2 = node2
        self.sensorID = sensorID
        self.strain = 0

    def getNodes(self):
        return self.node1, self.node2

    def getEndPoints(self):
        return self.node1.getPosition(), self.node2.getPosition()

    def getSensorID(self):
        return self.sensorID

    def getStrain(self):
        return self.strain

    def setStrain(self, strain):
        self.strain = strain