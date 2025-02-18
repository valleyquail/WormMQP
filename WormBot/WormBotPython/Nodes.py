import numpy as np
import matplotlib.pyplot as plt
from multipledispatch import dispatch

class Node:
    def __init__(self, initial_x: float, initial_y: float, initial_z: float,
                 node_level: int, node_type: str = None, node_id=None):
        self.x: float = initial_x
        self.y: float = initial_y
        self.z: float = initial_z
        self.level: int = node_level
        self.id: int = node_id
        self.type: str = node_type

    def getPosition(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z

    def getLevel(self) -> int:
        return self.level

    def get_id(self) -> int:
        return self.id

    def getType(self) -> str:
        return self.type

    @dispatch(float, float, float)
    def set_position(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    @dispatch(tuple)  # Changed from tuple[float, float, float] to just tuple
    def set_position(self, pos: tuple) -> None:
        # Added validation and fixed the indexing
        if len(pos) != 3:
            raise ValueError("Position tuple must have exactly 3 elements")
        self.x, self.y, self.z = pos  # Removed [0] as pos is already the tuple

    def __str__(self):  # Fixed the double underscores
        return f"Node id {self.id}, level {self.level}, type {self.type})"

class MidpointNode(Node):
    def __init__(self, initial_x: float, initial_y: float, initial_z: float, node_level: int):
        super().__init__(initial_x, initial_y, initial_z, node_level)
        self.type = 'midpoint'

class VertexNode(Node):
    def __init__(self, initial_x: float, initial_y: float, initial_z: float, node_level: int):
        super().__init__(initial_x, initial_y, initial_z, node_level)
        self.type = 'vertex'