import numpy as np
import matplotlib.pyplot as plt
from SensorNode import SensorNode


# Reference Papers:
# Title: Parametric design of developable structure based on Yoshimura origami pattern
# DOI: 10.54113/j.sust.2022.000019

# Title: Rigid-flexible coupled origami robots via multimaterial 3D printing
# DOI: 10.1088/1361-665X/ad212c


def generate_unit(beta, major_sl, minor_sl, num_sides, height_index=0):
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
    h_1 = minor_sl / 2 * np.tan(beta)
    # Offset the base height by the height index
    base_height = height_index * h_1
    mid_h = h_1 / 2 + base_height
    h_1 *= (height_index + 1)

    angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    x_angles_base = np.cos(angles)
    y_angles_base = np.sin(angles)
    vertex_coords_base = np.array(
        [[x * major_sl, y * major_sl, base_height] for x, y in zip(x_angles_base, y_angles_base)])
    vertex_coords_base = np.reshape(vertex_coords_base, (num_sides, 3))
    # Total height of a unit

    vertex_coords_top = vertex_coords_base + np.array([0, 0, h_1])
    # C is the point located halfway between the two vertices on the major side, let's call them A and B
    theta_ACB = np.arccos((h_1 * np.sqrt(2)) / (2 * minor_sl))
    # The inset norm is the scalar distance from the midpoint of the major side to the crease
    inset_norm = np.sin(theta_ACB) * minor_sl / np.sqrt(2)

    # Define the first points of the crease coordinates
    # X position is the x position of the first vertex on the major side minus the inset norm
    # Y position is the y position of the first vertex on the major side minus half the minor side length
    # Z position is the mid height

    inset_pos_one = np.array([vertex_coords_base[0, 0] - inset_norm, minor_sl / 2])
    inset_neg_one = np.array([vertex_coords_base[0, 0] - inset_norm, -minor_sl / 2])
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
        node_level = height_index
        # If this is not the base of the unit, increment the level by one
        if vertices[i, 2] - base_height > 0:
            node_level += 1
        node = SensorNode(vertices[i, 0], vertices[i, 1], vertices[i, 2], node_level, "vertex")
        vertex_nodes.append(node)

    creases = np.vstack((insets_pos, insets_neg))
    crease_nodes = []
    for i in range(creases.shape[0]):
        node = SensorNode(creases[i, 0], creases[i, 1], creases[i, 2], height_index, "crease")
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


def generate_unit2(beta, major_sl, minor_sl, num_sides, height_index=0, prev_top_nodes=None):
    """ Generate a unit cell
    Generates a unit cell with the given parameters. A unit cell comprises a set of vertices that outline the bottom
    and top faces of a cell as well as the vertices in the middle that form the creases. The vertices are returned in a list.
    @:param alpha: the angle between the major_sl side and the x-axis
    @:param beta: the angle between the minor_sl side and the x-axis
    @:param major_sl: the length of the major_sl side
    @:param minor_sl: the length of the minor_sl side
    @:param num_sides: the number of sides of the unit cell
    return: the coordinates of the vertices of the unit cell
   """

    def parseArrayToNodes(array, nodeType, heightIndex, euclidianRange):
        nodes = []
        nodeEdgePairs = []
        for i in range(array.shape[0]):
            # If this is not the base of the unit, increment the level by one
            node = SensorNode(array[i, 0], array[i, 1], array[i, 2], heightIndex, nodeType)
            nodes.append(node)
        # Add the other base nodes as neighbours
        for i, node in enumerate(nodes):
            for other_node in nodes[i:]:
                if node.get_position() != other_node.get_position():
                    if np.linalg.norm(
                            np.array(node.get_position()) - np.array(other_node.get_position())) < euclidianRange * 1.1:
                        nodeEdgePairs.append((node, other_node))
        return nodes, nodeEdgePairs

    nodeEdges = []
    # Total height of a unit
    h_unit = minor_sl / 2 * np.tan(beta)
    # Offset the base height by the height index
    base_height = height_index * h_unit
    mid_h = h_unit / 2

    # The radius of the circumscribed circle of the polygon
    startingRadius = major_sl / (2 * np.sin(np.pi / num_sides))

    angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    vertex_coords_base = []
    for angle in angles:
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        vertex_coords_base.append(np.array([startingRadius, 0, 0]).dot(rotation))

    vertex_coords_base = np.reshape(vertex_coords_base, (num_sides, 3))
    vertex_coords_top = vertex_coords_base + np.array([0, 0, base_height + h_unit])
    # Create the base nodes and add them as neighbors
    if prev_top_nodes == None:
        baseNodes, baseEdges = parseArrayToNodes(vertex_coords_base, "vertex", height_index, major_sl)
        nodeEdges += baseEdges
    else:
        baseNodes = prev_top_nodes
    topNodes, topEdges = parseArrayToNodes(vertex_coords_top, "vertex", height_index + 1, major_sl)
    nodeEdges += topEdges

    # C is the point located halfway between the two vertices on the major_sl side, let's call them A and B
    theta_ACB = np.arccos((h_unit * np.sqrt(2)) / (2 * minor_sl))
    # The inset norm is the scalar distance from the midpoint of the major_sl side to the crease
    inset_norm = np.sin(theta_ACB) * minor_sl / np.sqrt(2)

    # Define the first points of the crease coordinates
    # X position is the x position of the first vertex on the major_sl side minus the inset norm
    # Y position is the y position of the first vertex on the major_sl side minus half the minor_sl side length
    # Z position is the mid height

    inset_pos_one = np.array([vertex_coords_base[0, 0] - inset_norm, minor_sl / 2])
    inset_neg_one = np.array([vertex_coords_base[0, 0] - inset_norm, -minor_sl / 2])
    inset_pos = []
    inset_neg = []
    for angle in angles:
        # Define basic rotation matrix
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        # Rotate the first point of the crease
        inset_pos.append(np.dot(rotation, inset_pos_one))
        inset_neg.append(np.dot(rotation, inset_neg_one))

    mid_heights = np.ones(num_sides) * (mid_h + base_height)
    insets_pos = np.array(inset_pos).reshape(num_sides, 2)
    insets_pos = np.hstack((insets_pos, mid_heights.reshape(num_sides, 1)))
    insets_neg = np.array(inset_neg).reshape(num_sides, 2)
    insets_neg = np.hstack((insets_neg, mid_heights.reshape(num_sides, 1)))

    # creaseToCreaseNorm = np.linalg.norm(np.array([vertex_coords_base[0, :] - insets_pos[0, :]]))

    creases = np.vstack((insets_pos, insets_neg))

    creaseNodes, creaseEdges = parseArrayToNodes(creases, "crease", height_index, major_sl - minor_sl)
    nodeEdges += creaseEdges

    # Add the creases as neighbours to the vertices
    vertexNodes = baseNodes + topNodes
    for vertex in vertexNodes:
        for crease in creaseNodes:
            if np.linalg.norm(
                    np.array(vertex.get_position()) - np.array(crease.get_position())) < minor_sl * 1.1:
                nodeEdges += [(vertex, crease)]

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
    return baseNodes, topNodes, creaseNodes, nodeEdges


class Arm():
    def __init__(self, beta: float, major_sl: float, minor_sl: float, num_sides: int, num_units: int):

        # List of creases
        creases = []
        vertices = []
        edgePairs = []
        # Array used to be fed back the the generate_unit function to make
        # the next unit's base it the top of the previous unit
        topNodes = None
        old_base = None
        # Create all the units
        for i in range(num_units):
            baseNodes, topNodes, crease_nodes, edges = generate_unit2(beta, major_sl, minor_sl, num_sides, i, topNodes)
            old_base = topNodes
            creases += crease_nodes
            vertices += baseNodes
            edgePairs += edges
        vertices += topNodes

        self.creases: list[SensorNode] = creases
        self.vertices: list[SensorNode] = vertices
        self.edges: list[tuple[SensorNode, SensorNode]] = edgePairs
        self.beta: float = beta
        self.major_sl: float = major_sl
        self.minor_sl: float = minor_sl
        self.num_sides: int = num_sides
        self.num_units: int = num_units

    def drawArm(self) -> None:
        def extractPoints() -> (np.ndarray, np.ndarray):
            vertexPoints = []
            creasePoints = []
            for vertex in self.vertices:
                vertexPoints.append(vertex.get_position())
            for crease in self.creases:
                creasePoints.append(crease.get_position())
            creases = np.reshape(creasePoints, (-1, 3))
            vertices = np.reshape(vertexPoints, (-1, 3))
            return vertices, creases

        vertices, creases = extractPoints()
        fig = plt.figure()

        ax = fig.add_subplot(projection='3d')

        ax.scatter(vertices[:, 0],
                   vertices[:, 1],
                   vertices[:, 2], c="b", s=50)
        ax.scatter(creases[:, 0],
                   creases[:, 1],
                   creases[:, 2], c="g", s=50)

        for node, neighbour in self.edges:
            nodePos = node.get_position()
            neighbourPos = neighbour.get_position()
            ax.plot([nodePos[0], neighbourPos[0]],
                    [nodePos[1], neighbourPos[1]],
                    [nodePos[2], neighbourPos[2]], c="k")

        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_zlim(0, 70)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()


if __name__ == '__main__':
    arm = Arm(0.2, 60, 28, 4, 5)
    arm.drawArm()
    pass
