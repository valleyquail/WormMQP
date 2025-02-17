import numpy as np

from Nodes import VertexNode


class SensorEdge:

    def __init__(self, node1: VertexNode, node2: VertexNode, sensor_id: int = None):
        self.node1: VertexNode = node1
        self.node2: VertexNode = node2
        self.sensorID: int = sensor_id
        self.strain: float = 0

    def getNodes(self) -> tuple[VertexNode, VertexNode]:
        return self.node1, self.node2

    def getEndPoints(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return self.node1.getPosition(), self.node2.getPosition()

    def getSensorID(self) -> int:
        return self.sensorID

    def setSensorID(self, sensor_id) -> None:
        self.sensorID = sensor_id

    def getStrain(self) -> float:
        return self.strain

    def setStrain(self, strain) -> None:
        self.strain = strain
