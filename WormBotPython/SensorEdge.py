import numpy as np

class SensorEdge:
    """
    Represents a sensor edge between two vertex nodes in the continuum arm.
    This edge measures the distance between vertices to track deformation.
    """

    def __init__(self, node1, node2, sensor_id=None):
        # The two vertex nodes this sensor connects
        self.node1 = node1
        self.node2 = node2
        # Unique identifier for this sensor
        self.sensor_id = sensor_id
        # Store the initial distance between nodes as reference
        self.initial_distance = self._calculate_distance()
        # Current distance between nodes
        self.current_distance = self.initial_distance

    def _calculate_distance(self):
        """Calculate the Euclidean distance between the two nodes."""
        pos1 = self.node1.getPosition()
        pos2 = self.node2.getPosition()
        return np.sqrt(
            (pos2[0] - pos1[0]) ** 2 +
            (pos2[1] - pos1[1]) ** 2 +
            (pos2[2] - pos1[2]) ** 2
        )

    def updatePositions(self):
        """
        Update the current distance between nodes.
        Called after any movement to track deformation.
        """
        self.current_distance = self._calculate_distance()

    def getEndPoints(self):
        """Return the current positions of both end nodes."""
        return self.node1.getPosition(), self.node2.getPosition()

    def getStrain(self):
        """
        Calculate the strain (deformation) in the edge.
        Strain is the relative change in length.
        """
        return (self.current_distance - self.initial_distance) / self.initial_distance

    def setSensorID(self, sensor_id):
        """Set the unique identifier for this sensor."""
        self.sensor_id = sensor_id

    def getSensorID(self):
        """Get the unique identifier for this sensor."""
        return self.sensor_id