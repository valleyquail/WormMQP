import numpy as np
import matplotlib.pyplot as plt
from Nodes import VertexNode

class SensorEdge:

    def __init__(self, node1: VertexNode, node2: VertexNode, sensorID: int=None):
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

    def setSensorID(self, sensorID):
        self.sensorID = sensorID

    def getStrain(self):
        return self.strain

    def setStrain(self, strain):
        self.strain = strain