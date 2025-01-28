import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from Nodes import Node, SensorNode, VertexNode
from SensorEdge import SensorEdge


# Reference Papers:
# Title: Parametric design of developable structure based on Yoshimura origami pattern
# DOI: 10.54113/j.sust.2022.000019

# Title: Rigid-flexible coupled origami robots via multimaterial 3D printing
# DOI: 10.1088/1361-665X/ad212c


def generate_unit(beta, major_sl, minor_sl, num_sides, height_index=0, prev_top_nodes=None):
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

    def parseArrayToNodes(array: np.ndarray, heightIndex: int, euclidianRange: float):
        nodes = []
        nodeEdgePairs = []
        for i in range(array.shape[0]):
            # If this is not the base of the unit, increment the level by one
            node = VertexNode(array[i, 0], array[i, 1], array[i, 2], heightIndex)
            nodes.append(node)
        # Add the other base nodes as neighbours
        for i, node in enumerate(nodes):
            for other_node in nodes[i:]:
                if node.getPosition() != other_node.getPosition():
                    dist = norm(np.array(node.getPosition()) - np.array(other_node.getPosition()))
                    if dist < euclidianRange * 1.1:
                        nodeEdgePairs.append((node, other_node))
        return nodes, nodeEdgePairs

    def parseArrayToCreases(array: np.ndarray, heightIndex: int, euclidianRange: float, minor_sl: float):
        nodes, long_edges, sensor_edges = [], [], []
        for i in range(array.shape[0]):
            # If this is not the base of the unit, increment the level by one
            node = SensorNode(array[i, 0], array[i, 1], array[i, 2], heightIndex)
            nodes.append(node)
        # Add the other base nodes as neighbours
        for i, node in enumerate(nodes):
            for neighbor in nodes[i:]:
                # If the node is very close/the minor side length away from the neighbor, add it as a sensor edge
                if node.getPosition() != neighbor.getPosition():
                    dist = norm(np.array(node.getPosition()) - np.array(neighbor.getPosition()))
                    if minor_sl * 0.95 < dist < minor_sl * 1.05:
                        sensor_edges.append((node, neighbor))
                    # If the node is very close/the euclidian range away from the neighbor, add it as a normal edge
                    elif dist < euclidianRange * 1.1:
                        long_edges.append((node, neighbor))

        return nodes, long_edges, sensor_edges

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
    # Rotate the base vertices to get the coordinates
    for angle in angles:
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        vertex_coords_base.append(np.array([startingRadius, 0, 0]).dot(rotation))

    vertex_coords_base = np.reshape(vertex_coords_base, (num_sides, 3))
    vertex_coords_top = vertex_coords_base + np.array([0, 0, base_height + h_unit])

    # Create the base nodes and add them as neighbors
    if prev_top_nodes == None:
        baseNodes, baseEdges = parseArrayToNodes(vertex_coords_base, height_index, major_sl)
        nodeEdges += baseEdges
    else:
        baseNodes = prev_top_nodes

    topNodes, topEdges = parseArrayToNodes(vertex_coords_top, height_index + 1, major_sl)
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

    creases = np.vstack((insets_pos, insets_neg))

    creaseNodes, creaseEdges, sensorEdges = parseArrayToCreases(creases, height_index, major_sl - minor_sl, minor_sl)
    nodeEdges += creaseEdges

    # Add the creases as neighbours to the vertices
    vertexNodes = baseNodes + topNodes
    for vertex in vertexNodes:
        for crease in creaseNodes:
            dist = norm(np.array(vertex.getPosition()) - np.array(crease.getPosition()))
            if dist < minor_sl * 1.1:
                nodeEdges += [(vertex, crease)]

    return baseNodes, topNodes, creaseNodes, nodeEdges, sensorEdges


class Arm():
    def __init__(self, beta: float, major_sl: float, minor_sl: float, numSides: int, num_units: int):
        # Sorts the vertices and corners into an organized dictionary based on the height index of the node

        def organizeByLayer(nodes: list[Node]) -> dict[int, list[SensorNode]]:
            """
             Organize the nodes into a dictionary based on their height index
             Makes it very convenient for iterating through the nodes in a layer by layer
             fashion and applying transformations
            :param nodes:
            :return: a dictionary that maps the height index into a list of nodes at that index
            """
            organizedData = dict()
            for nodes in nodes:
                height_idx = nodes.getLevel()
                if height_idx not in organizedData:
                    organizedData[height_idx] = []
                    organizedData[height_idx].append(nodes)
            return organizedData

        def assignSensorIds(sensorEdgePairs: list[SensorNode, SensorNode]) -> dict[int, SensorEdge]:
            """
            Organize the edges into crease and vertex edges
            @:param edges: the edges to be organized
            @:return: a tuple containing the crease edges and the vertex edges
            """
            sensorEdges = {}
            for i, edge in enumerate(sensorEdgePairs):
                sensorEdge = SensorEdge(edge[0], edge[1], i)
                sensorEdges[i] = sensorEdge
            return sensorEdges

        # List of creases
        creases: list[SensorNode] = []
        vertices: list[VertexNode] = []
        edgePairs: list[tuple[Node, Node]] = []
        sensorEdges: list[tuple[SensorNode, SensorNode]] = []
        # Array used to be fed back the the generate_unit function to make
        # the next unit's base it the top of the previous unit
        tNodes = None
        # Create and stack all the units into the organized data
        for i in range(num_units):
            bNodes, tNodes, cNodes, edges, sensors = generate_unit(beta, major_sl, minor_sl, numSides, i, tNodes)
            creases += cNodes
            vertices += bNodes
            edgePairs += edges
            sensorEdges += sensors
        vertices += tNodes

        # Assign ids to each vertex node
        for i, node in enumerate(vertices):
            node.id = i

        # Assign ids to each crease node
        for i, node in enumerate(creases):
            node.id = i + len(vertices)

        self.vertices = vertices
        self.creases = creases
        self.edges = edgePairs
        self.organized: dict[int, list[SensorNode]] = organizeByLayer(creases + vertices)
        self.edges: list[tuple[SensorNode, SensorNode]] = edgePairs
        self.sensorEdges: dict[int, SensorEdge] = assignSensorIds(sensorEdges)
        self.beta: float = beta
        self.major_sl: float = major_sl
        self.minor_sl: float = minor_sl
        self.num_sides: int = numSides
        self.num_units: int = num_units

    def drawArm(self) -> None:
        def extractPoints() -> (np.ndarray, np.ndarray):
            vertexPoints = []
            creasePoints = []
            nodes = self.vertices + self.creases
            for node in nodes:
                if node.getType() == 'crease':
                    creasePoints.append(node.getPosition())
                else:
                    vertexPoints.append(node.getPosition())
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
            nodePos = node.getPosition()
            neighbourPos = neighbour.getPosition()
            ax.plot([nodePos[0], neighbourPos[0]],
                    [nodePos[1], neighbourPos[1]],
                    [nodePos[2], neighbourPos[2]], c="k")

        for sensorEdge in self.sensorEdges.values():
            node1, node2 = sensorEdge.getNodes()
            node1Pos = node1.getPosition()
            node2Pos = node2.getPosition()
            ax.plot([node1Pos[0], node2Pos[0]],
                    [node1Pos[1], node2Pos[1]],
                    [node1Pos[2], node2Pos[2]], c="r")

        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_zlim(0, 70)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()


if __name__ == '__main__':
    arm = Arm(0.6, 60, 28, 4, 5)
    arm.drawArm()
    pass
