import numpy as np
import matplotlib.pyplot as plt
import enum as Enum

class SensorNode():

    def __init__(self, initial_x, initial_y, initial_z, node_level, node_id, node_type):
        self.x = initial_x
        self.y = initial_y
        self.z = initial_z
        self.level = node_level
        self.id = node_id
        self.neighbours = []
        self.type = node_type

        def add_neighbour(self, neighbour):
            self.neighbours.append(neighbour)

        def get_neighbours(self):
            return self.neighbours

        def get_position(self):
            return (self.x, self.y, self.z)

        def get_level(self):
            return self.level

        def get_id(self):
            return self.id

        def set_position(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
