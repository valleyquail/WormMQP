import numpy as np
import matplotlib.pyplot as plt
from SensorNode import SensorNode

# Reference Papers:
# Title: Parametric design of developable structure based on Yoshimura origami pattern
# DOI: 10.54113/j.sust.2022.000019

# Title: Rigid-flexible coupled origami robots via multimaterial 3D printing
# DOI: 10.1088/1361-665X/ad212c


def generate_unit(beta, major_sl, minor_sl, num_sides):
    """ Generate a unit cell
    Generates a unit cell with the given parameters. A unit cell comprises a set of vertices that outline the bottom
    and top faces of a cell as well as the vertices in the middle that form the creases. The vertices are returned in a list.
    @:param alpha: the angle between the major side and the x-axis
    @:param beta: the angle between the minor side and the x-axis
    @:param major_sl: the length of the major side
    @:param minor_sl: the length of the minor side
    @:param num_sides: the number of sides of the unit cell
    return: the coordinates of the vertices of the unit cell
   """
    angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    x_angles_base = np.cos(angles)
    y_angles_base = np.sin(angles)
    vertex_coords_base = np.array([[x * major_sl, y * major_sl, 0] for x, y in zip(x_angles_base, y_angles_base)])
    vertex_coords_base = np.reshape(vertex_coords_base, (num_sides, 3))
    # Total height of a unit
    h_1 = minor_sl / 2 * np.tan(beta)
    mid_h = h_1 / 2
    vertex_coords_top = vertex_coords_base + np.array([0, 0, h_1])
    # C is the point located halfway between the two vertices on the major side, let's call them A and B
    theta_ACB = np.arccos((h_1 * np.sqrt(2)) / (2 * minor_sl))
    # The inset norm is the scalar distance from the midpoint of the major side to the crease
    inset_norm = np.sin(theta_ACB) * minor_sl / np.sqrt(2)

    # Define the first points of the crease coordinates
    # X position is the x position of the first vertex on the major side minus the inset norm
    # Y position is the y position of the first vertex on the major side minus half the minor side length
    # Z position is the mid height

    inset_pos_one = np.array([vertex_coords_base[0, 0] -inset_norm, minor_sl/2])
    inset_neg_one = np.array([vertex_coords_base[0, 0] -inset_norm, -minor_sl/2])
    inset_pos = []
    inset_neg = []
    for angle in angles:
        # Define basic rotation matrix
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        # Rotate the first point of the crease
        inset_pos.append(np.dot(rotation, inset_pos_one))
        inset_neg.append(np.dot(rotation, inset_neg_one))

    mid_heights = np.ones(num_sides) * mid_h
    insets_pos = np.array(inset_pos).reshape(num_sides, 2)
    insets_pos = np.hstack((insets_pos, mid_heights.reshape(num_sides, 1)))
    insets_neg = np.array(inset_neg).reshape(num_sides, 2)
    insets_neg = np.hstack((insets_neg, mid_heights.reshape(num_sides, 1)))

    # Make classes
    vertices = np.vstack((vertex_coords_base, vertex_coords_top))
    vertex_nodes = []
    for i in range(vertices.shape[0]):
        node_level = 0
        if vertices[i, 2] > 0:
            node_level += 1
        node = SensorNode(vertices[i, 0], vertices[i, 1], vertices[i, 2], 0, i, "vertex")
        vertex_nodes.append(node)

    creases = np.vstack((insets_pos, insets_neg))
    crease_nodes = []
    for i in range(creases.shape[0]):
        node = SensorNode(creases[i, 0], creases[i, 1], creases[i, 2], 1, i, "crease")
        crease_nodes.append(node)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    #
    # ax.scatter(vertex_coords_base[:, 0], vertex_coords_base[:, 1], vertex_coords_base[:, 2], c="b", s=50)
    # ax.scatter(vertex_coords_top[:, 0], vertex_coords_top[:, 1], vertex_coords_top[:, 2], c="b", s=50)
    # ax.scatter(insets_pos[:, 0], insets_pos[:, 1], insets_pos[:, 2], c="g", s=50)
    # ax.scatter(insets_neg[:, 0], insets_neg[:, 1], insets_neg[:, 2], c="g", s=50)
    #
    # ax.set_xlim(-100, 100)
    # ax.set_ylim(-100, 100)
    # ax.set_zlim(0, 40)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    #
    # plt.show()
    return vertex_nodes, crease_nodes

class Arm():
    def __init__(self):
        pass


if __name__ == '__main__':
    generate_unit(0.5, 60, 28, 4)
    pass
