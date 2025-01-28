import numpy as np
import matplotlib.pyplot as plt
import enum as Enum


class Node:

    def __init__(self, initial_x: float, initial_y: float, initial_z: float,
                 node_level: int, node_type: str = None, node_id=None):
        self.x: float = initial_x
        self.y: float = initial_y
        self.z: float = initial_z
        self.level: int = node_level
        self.id = node_id
        self.type: str = node_type

    def getPosition(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z

    def getLevel(self):
        return self.level

    def get_id(self):
        return self.id

    def getType(self):
        return self.type

    def set_position(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"Node at position ({self.x:.2f}, {self.y:.2f}, {self.z:.2f}, level {self.level}, type {self.type})"


class MidpointNode(Node):
    def __init__(self, initial_x: float, initial_y: float, initial_z: float, node_level: int):
        super().__init__(initial_x, initial_y, initial_z, node_level)
        self.type = 'crease'


class VertexNode(Node):
    def __init__(self, initial_x: float, initial_y: float, initial_z: float, node_level: int):
        super().__init__(initial_x, initial_y, initial_z, node_level)
        self.type = 'vertex'
