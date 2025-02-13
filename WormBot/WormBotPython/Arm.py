import copy

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from Nodes import Node, MidpointNode, VertexNode
from SensorEdge import SensorEdge
import itertools


# Reference Papers:
# Title: Parametric design of developable structure based on Yoshimura origami pattern
# DOI: 10.54113/j.sust.2022.000019

# Title: Rigid-flexible coupled origami robots via multimaterial 3D printing
# DOI: 10.1088/1361-665X/ad212c

# Title: Energy absorption of thin-walled tubes with pre-folded
# origami patterns: Numerical simulation and experimental verification
# DOI: https://doi.org/10.1016/j.tws.2016.02.007

def generate_unit(beta, major_sl, minor_sl, num_sides, num_units, height_index=0, prev_top_nodes=None):
    """ Generate a unit cell
    Generates a unit cell with the given parameters. A unit cell comprises a set of vertices that outline the bottom
    and top faces of a cell as well as the vertices in the middle that form the creases. The vertices are returned in a list.
    @:param alpha: the angle between the major_sl side and the x-axis
    @:param beta: the angle between the minor_sl side and the x-axis
    @:param major_sl: the length of the major_sl side
    @:param minor_sl: the length of the minor_sl side
    @:param num_sides: the number of sides of the unit cell
    @param num_units: the number of unit cells to generate
    return: the coordinates of the vertices of the unit cell
   """

    def parseArrayToNodes(array: np.ndarray, height_index: int, euclidian_range: float, type: str):
        nodes = []
        nodeEdgePairs = []
        for i in range(array.shape[0]):
            if type == 'midpoint':
                node = MidpointNode(array[i, 0], array[i, 1], array[i, 2], height_index)
            else:
                node = VertexNode(array[i, 0], array[i, 1], array[i, 2], height_index)
            nodes.append(node)
        # Add the other base nodes as neighbours
        for i, node in enumerate(nodes):
            for other_node in nodes[i:]:
                if node.getPosition() != other_node.getPosition():
                    dist = norm(np.array(node.getPosition()) - np.array(other_node.getPosition()))

                    if dist < euclidian_range * 1.01:
                        nodeEdgePairs.append((node, other_node))
        return nodes, nodeEdgePairs

    def parseArrayToSensors(array: np.ndarray, heightIndex: int, euclidianRange: float):
        nodes: list[VertexNode] = []
        sensorEdges: list[SensorEdge] = []
        for i in range(array.shape[0]):
            # If this is not the base of the unit, increment the level by one
            node = VertexNode(array[i, 0], array[i, 1], array[i, 2], heightIndex)
            nodes.append(node)
        # Add the other base nodes as neighbours
        for i, node in enumerate(nodes):
            for neighbor in nodes[i:]:
                if node.getPosition() != neighbor.getPosition():
                    dist = norm(np.array(node.getPosition()) - np.array(neighbor.getPosition()))
                    if dist < euclidianRange * 1.05:
                        # Make a sensor edge without an id since it will be assigned later
                        sensorEdges.append(SensorEdge(node, neighbor))
        return nodes, sensorEdges

    nodeEdges = []
    sensorEdges = []

    # Total height of a unit when unfolded
    h_flat = 2 * (minor_sl / 2) * np.tan(beta)
    # C is the point located halfway between the top and bottom vertices on the major_sl side, let's call them A and B
    # Calculate the one half the dihedral angle ACB
    dihedral = np.arccos((minor_sl / h_flat) * np.tan(np.pi / (2 * num_sides)))

    # The height of a unit when folded
    h_unit = h_flat * np.sin(dihedral)
    # Offset the base height by the height index
    base_height = height_index * h_unit
    mid_h = h_unit / 2

    # The inset norm is the scalar distance from the line AB and C
    inset_norm = h_flat / 2 * np.cos(dihedral)

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
        baseNodes, baseEdges = parseArrayToNodes(vertex_coords_base, height_index, major_sl, 'vertex')
        nodeEdges += baseEdges
    else:
        baseNodes = prev_top_nodes

    if height_index == num_units - 1:
        topNodes, edges = parseArrayToNodes(vertex_coords_top, height_index + 1, major_sl, 'vertex')
        nodeEdges += edges
    else:
        topNodes, sensorEdges = parseArrayToSensors(vertex_coords_top, height_index + 1, major_sl)

    # Define the first points of the midpoint coordinates
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
        # Rotate the first point of the midpoint
        inset_pos.append(np.dot(rotation, inset_pos_one))
        inset_neg.append(np.dot(rotation, inset_neg_one))

    mid_heights = np.ones(num_sides) * (mid_h + base_height)

    insets_pos = np.array(inset_pos).reshape(num_sides, 2)
    insets_pos = np.hstack((insets_pos, mid_heights.reshape(num_sides, 1)))
    insets_neg = np.array(inset_neg).reshape(num_sides, 2)
    insets_neg = np.hstack((insets_neg, mid_heights.reshape(num_sides, 1)))

    midpoints = np.vstack((insets_pos, insets_neg))



    midpointNodes, midpointEdges = parseArrayToNodes(midpoints, height_index, minor_sl, 'midpoint')
    nodeEdges += midpointEdges

    # The midpoint limiting distance is the sidelength of one of the diamonds in the pattern
    # This is used to prevent the drawing of extra lines between the midpoints
    midpoint_to_vertex_dist = (minor_sl / 2) / np.cos(beta)

    # Add the midpoints as neighbours to the vertices
    vertexNodes = baseNodes + topNodes
    for vertex in vertexNodes:
        for midpoint in midpointNodes:
            dist = norm(np.array(vertex.getPosition()) - np.array(midpoint.getPosition()))
            # Midpoint limiter helps parse the edges between the vertices
            # and the midpoints to prevent extra lines from being drawn
            if  dist < midpoint_to_vertex_dist * 1.1:
                nodeEdges += [(vertex, midpoint)]

    return baseNodes, topNodes, midpointNodes, nodeEdges, sensorEdges


class Arm():
    def __init__(self, beta: float, major_sl: float, minor_sl: float, num_sides: int, num_units: int):
        # Sorts the vertices and corners into an organized dictionary based on the height index of the node

        def organizeByLayer(nodes: list[Node]) -> dict[int, list[MidpointNode]]:
            """
             Organize the nodes into a dictionary based on their height index
             Makes it very convenient for iterating through the nodes in a layer by layer
             fashion and applying transformations

             Also assigns the id to each node
            :param nodes:
            :return: a dictionary that maps the height index into a list of nodes at that index
            """
            organizedData = dict()
            curr_id = 0
            for node in nodes:
                height_idx = node.getLevel()
                if height_idx not in organizedData:
                    organizedData[height_idx] = []
                node.id = curr_id
                organizedData[height_idx].append(nodes)
                curr_id += 1
            return organizedData

        def assignSensorIds(sensor_edges: list[SensorEdge]) -> dict[int, SensorEdge]:
            """
            Organize the edges into midpoint and vertex edges
            @:param edges: the edges to be organized
            @:return: a tuple containing the midpoint edges and the vertex edges
            """
            sensors = {}
            for i, edge in enumerate(sensor_edges):
                edge.setSensorID(i)
                sensors[i] = edge
            return sensors

        # List of midpoints
        midpoint: list[MidpointNode] = []
        _vertices: list[VertexNode] = []
        _edgePairs: list[tuple[Node, Node]] = []
        _sensor_edges: list[SensorEdge] = []
        # Array used to be fed back to the generate_unit function to make
        # the next unit's base it the top of the previous unit
        tNodes = None
        # Create and stack all the units into the organized data
        for i in range(num_units):
            bNodes, tNodes, cNodes, edges, sensors = generate_unit(beta, major_sl, minor_sl, num_sides, num_units, i,
                                                                   tNodes)
            midpoint += cNodes
            _vertices += bNodes
            _edgePairs += edges
            _sensor_edges += sensors
        _vertices += tNodes

        # Assign ids to each vertex node
        for i, node in enumerate(_vertices):
            node.id = i

        # Assign ids to each midpoint node
        for i, node in enumerate(midpoint):
            node.id = i + len(_vertices)

        self.vertices = _vertices
        self.midpoint = midpoint
        self.edges = _edgePairs
        self.arm_dict: dict[int, list[Node]] = organizeByLayer(midpoint + _vertices)
        self._default_pose = copy.deepcopy(self.arm_dict)
        self.edges: list[tuple[Node, MidpointNode]] = _edgePairs
        self.sensorEdges: dict[int, SensorEdge] = assignSensorIds(_sensor_edges)

        self._beta: float = beta
        self._major_sl: float = major_sl
        self._minor_sl: float = minor_sl
        self._num_sides: int = num_sides
        self._num_units: int = num_units

    # TODO: Rewrite this function to use the arm_dict to reassign the stuff in the edges and the sensorEdges
    def resetPose(self) -> None:
        def sort_key(node: Node):
            return node.get_id()
        default_nodes = list(itertools.chain.from_iterable(self._default_pose.values()))[0]
        default_nodes = sorted(default_nodes, key=sort_key)
        nodes = list(itertools.chain.from_iterable(self.arm_dict.values()))[0]
        nodes = sorted(nodes, key=sort_key)
        for i in range(len(nodes)):
            nodes[i].set_position(default_nodes[i].getPosition())

    def forwardKinematics(self, theta: float) -> None:
        pass

    def drawArm(self) -> None:
        def extractPoints() -> (np.ndarray, np.ndarray):
            vertex_points = []
            midpoint_points = []
            nodes = list(itertools.chain.from_iterable(self.arm_dict.values()))[0]
            for n in nodes:
                if n.getType() == 'midpoint':
                    midpoint_points.append(n.getPosition())
                else:
                    vertex_points.append(n.getPosition())
            midpoints = np.reshape(midpoint_points, (-1, 3))
            vertices = np.reshape(vertex_points, (-1, 3))
            return vertices, midpoints

        vertices, midpoints = extractPoints()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(vertices[:, 0],
                   vertices[:, 1],
                   vertices[:, 2], c="b", s=50)
        ax.scatter(midpoints[:, 0],
                   midpoints[:, 1],
                   midpoints[:, 2], c="g", s=50)

        for node, neighbour in self.edges:
            nodePos = node.getPosition()
            neighbourPos = neighbour.getPosition()
            ax.plot([nodePos[0], neighbourPos[0]],
                    [nodePos[1], neighbourPos[1]],
                    [nodePos[2], neighbourPos[2]], c="k")

        for sensorEdge in self.sensorEdges.values():
            node1Pos, node2Pos = sensorEdge.getEndPoints()
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
    arm = Arm(np.pi * 35 / 180, 60, 40, 4, 1)
    arm.drawArm()
    pass
