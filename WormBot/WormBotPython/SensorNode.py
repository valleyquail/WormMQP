import numpy as np
import matplotlib.pyplot as plt
import enum as Enum


class SensorNode():

    def __init__(self, initial_x: float, initial_y: float, initial_z: float, node_level: int, node_type: str,
                 node_id=None):
        self.x: float = initial_x
        self.y: float = initial_y
        self.z: float = initial_z
        self.level: int = node_level
        self.id = node_id
        self.neighbours: list[SensorNode] = []
        self.type: str = node_type

    def add_neighbour(self, neighbour):
        self.neighbours.append(neighbour)

    def get_neighbours(self):
        return self.neighbours

    def get_position(self)-> tuple[float, float, float]:
        return self.x, self.y, self.z

    def get_level(self):
        return self.level

    def get_id(self):
        return self.id

    def get_type(self):
        return self.type

    def set_position(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"Node at position ({self.x:.2f}, {self.y:.2f}, {self.z:.2f}, level {self.level}, type {self.type})"


