import copy
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib import cm
import itertools
from Nodes import Node, MidpointNode, VertexNode
from SensorEdge import SensorEdge


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

    startingRadius = major_sl / (2 * np.sin(np.pi / num_sides))

    angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    vertex_coords_base = []
    for angle in angles:
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        vertex_coords_base.append(np.array([startingRadius, 0, 0]).dot(rotation))

    vertex_coords_base = np.reshape(vertex_coords_base, (num_sides, 3))
    vertex_coords_top = vertex_coords_base + np.array([0, 0, base_height + h_unit])

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

    inset_pos_one = np.array([vertex_coords_base[0, 0] - inset_norm, minor_sl / 2])
    inset_neg_one = np.array([vertex_coords_base[0, 0] - inset_norm, -minor_sl / 2])
    inset_pos = []
    inset_neg = []
    for angle in angles:
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
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

    midpoint_to_vertex_dist = (minor_sl / 2) / np.cos(beta)

    vertexNodes = baseNodes + topNodes
    for vertex in vertexNodes:
        for midpoint in midpointNodes:
            dist = norm(np.array(vertex.getPosition()) - np.array(midpoint.getPosition()))
            if dist < midpoint_to_vertex_dist * 1.1:
                nodeEdges += [(vertex, midpoint)]

    return baseNodes, topNodes, midpointNodes, nodeEdges, sensorEdges


def organizeByLayer(nodes: list[Node]) -> dict[int, list[Node]]:
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
    sensors = {}
    for i, edge in enumerate(sensor_edges):
        edge.setSensorID(i)
        sensors[i] = edge
    return sensors


def forward_kinematics(vertices: list[Node], theta: float, side: str = 'right') -> None:

    def get_level_nodes(nodes: list[Node]) -> dict[int, list[Node]]:
        levels = {}
        for node in nodes:
            level = node.getLevel()
            if level not in levels:
                levels[level] = []
            levels[level].append(node)
        return levels

    def get_level_center(nodes: list[Node]) -> np.ndarray:
        positions = [np.array(node.getPosition()) for node in nodes]
        return np.mean(positions, axis=0)

    def create_backbone_curve(base_center: np.ndarray, height: float, theta: float) -> callable:

        def curve(h: float) -> np.ndarray:
            t = h / height  # Normalized height
            x = base_center[0] + height * np.sin(theta * t) * t
            y = base_center[1]
            z = h * np.cos(theta * t)
            return np.array([x, y, z])

        return curve

    def transform_cross_section(nodes: list[Node], center: np.ndarray,
                                new_center: np.ndarray, theta: float, t: float) -> None:
        rotation = np.array([
            [np.cos(theta * t), 0, -np.sin(theta * t)],
            [0, 1, 0],
            [np.sin(theta * t), 0, np.cos(theta * t)]
        ])

        for node in nodes:
            pos = np.array(node.getPosition())
            relative_pos = pos - center

            new_pos = rotation @ relative_pos + new_center
            node.set_position(tuple(new_pos))

    level_groups = get_level_nodes(vertices)
    if not level_groups:
        return

    base_nodes = level_groups[0]
    base_center = get_level_center(base_nodes)
    max_height = max(node.getPosition()[2] for node in vertices)

    backbone = create_backbone_curve(base_center, max_height, theta)

    for level, nodes in level_groups.items():
        if level == 0:
            continue

        current_center = get_level_center(nodes)
        current_height = current_center[2]
        t = current_height / max_height

        new_center = backbone(current_height)

        transform_cross_section(nodes, current_center, new_center, theta, t)


def asymmetric_forward_kinematics(vertices: list[Node], theta: float, side: str = 'right') -> None:

    def get_segment_centers(nodes: list[Node]) -> tuple[np.ndarray, np.ndarray]:
        base_points = []
        top_points = []

        for node in nodes:
            pos = np.array(node.getPosition())
            if node.getLevel() == 0:
                base_points.append(pos)
            elif node.getType() == 'vertex':
                top_points.append(pos)

        base_center = np.mean(np.array(base_points), axis=0)
        top_center = np.mean(np.array(top_points), axis=0)
        return base_center, top_center

    def get_node_side(node_pos: np.ndarray, base_center: np.ndarray, side: str) -> float:

        relative_pos = node_pos - base_center

        if side == 'right':
            return (np.tanh(relative_pos[0] / 50) + 1) / 2  # Scaled factor for smoother transition
        elif side == 'left':
            return (-np.tanh(relative_pos[0] / 50) + 1) / 2
        elif side == 'front':
            return (np.tanh(relative_pos[1] / 50) + 1) / 2
        elif side == 'back':
            return (-np.tanh(relative_pos[1] / 50) + 1) / 2
        else:
            return 1.0

    def get_height_factor(node: Node, max_height: float) -> float:
        node_height = node.getPosition()[2]
        return node_height / max_height if max_height > 0 else 0

    def create_rotation_matrix(theta: float) -> np.ndarray:
        return np.array([
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)]
        ])

    def transform_nodes(nodes: list[Node], base_center: np.ndarray,
                        rotation_matrix: np.ndarray, side: str, max_height: float) -> None:
        for node in nodes:
            if node.getLevel() == 0:
                continue

            pos = np.array(node.getPosition())

            height_factor = get_height_factor(node, max_height)
            side_factor = get_node_side(pos, base_center, side)

            total_factor = height_factor * side_factor

            scaled_rotation = R.from_matrix(rotation_matrix).as_rotvec() * total_factor
            scaled_rotation_matrix = R.from_rotvec(scaled_rotation).as_matrix()

            pos_centered = pos - base_center
            pos_rotated = scaled_rotation_matrix @ pos_centered
            pos_final = pos_rotated + base_center

            node.set_position(tuple(pos_final))

    max_height = max(node.getPosition()[2] for node in vertices)


    base_center, _ = get_segment_centers(vertices)
    rotation_matrix = create_rotation_matrix(theta)

    transform_nodes(vertices, base_center, rotation_matrix, side, max_height)


def trapezoid_interface_kinematics(vertices: list[Node], midpoints: list[Node], interface_angle: float) -> None:

    def find_interface_nodes(all_nodes: list[Node]) -> tuple[list[Node], list[Node]]:

        all_heights = [node.z for node in all_nodes]
        mid_height = (max(all_heights) + min(all_heights)) / 2

        interface_nodes = []
        other_nodes = []
        height_tolerance = 0.1

        for node in all_nodes:
            if abs(node.z - mid_height) < height_tolerance:
                interface_nodes.append(node)
            else:
                other_nodes.append(node)

        return interface_nodes, other_nodes

    def rotate_point(x: float, y: float, z: float,
                     center_x: float, center_y: float, center_z: float,
                     angle: float) -> tuple[float, float, float]:

        translated_x = x - center_x
        translated_z = z - center_z

        # Rotate around Y-axis
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        new_x = translated_x * cos_angle - translated_z * sin_angle + center_x
        new_z = translated_x * sin_angle + translated_z * cos_angle + center_z

        return (new_x, y, new_z)


    all_nodes = vertices + midpoints

    interface_nodes, other_nodes = find_interface_nodes(all_nodes)

    if len(interface_nodes) < 2:
        return


    center_x = sum(node.x for node in interface_nodes) / len(interface_nodes)
    center_y = sum(node.y for node in interface_nodes) / len(interface_nodes)
    center_z = sum(node.z for node in interface_nodes) / len(interface_nodes)

    half_angle = interface_angle / 2


    for node in other_nodes:
        if node.z > center_z:
            new_pos = rotate_point(node.x, node.y, node.z,
                                   center_x, center_y, center_z,
                                   half_angle)
        else:
            new_pos = rotate_point(node.x, node.y, node.z,
                                   center_x, center_y, center_z,
                                   -half_angle)

        node.set_position(new_pos)


class Arm():
    def __init__(self, beta: float, major_sl: float, minor_sl: float, num_sides: int, num_units: int):
        # List of midpoints
        _midpoints: list[MidpointNode] = []
        _vertices: list[VertexNode] = []
        _edgePairs: list[tuple[Node, Node]] = []
        _sensor_edges: list[SensorEdge] = []

        tNodes = None
        self.current_theta = 0.0

        for i in range(num_units):
            bNodes, tNodes, cNodes, edges, sensors = generate_unit(beta, major_sl, minor_sl, num_sides, num_units, i,
                                                                   tNodes)
            _midpoints += cNodes
            _vertices += bNodes
            _edgePairs += edges
            _sensor_edges += sensors
        _vertices += tNodes

        # Assign ids to each vertex node
        for i, node in enumerate(_vertices):
            node.id = i

        # Assign ids to each midpoint node
        for i, node in enumerate(_midpoints):
            node.id = i + len(_vertices)

        self._vertices = _vertices
        self._midpoints = _midpoints
        self.edges = _edgePairs
        self.arm_dict: dict[int, list[Node]] = organizeByLayer(_midpoints + _vertices)
        self._default_pose = copy.deepcopy(self.arm_dict)
        self.edges: list[tuple[Node, MidpointNode]] = _edgePairs
        self.sensorEdges: dict[int, SensorEdge] = assignSensorIds(_sensor_edges)

        self._beta: float = beta
        self._major_sl: float = major_sl
        self._minor_sl: float = minor_sl
        self._num_sides: int = num_sides
        self._num_units: int = num_units

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
        current_nodes = self._vertices + self._midpoints
        forward_kinematics(current_nodes, theta)
        self.current_theta = theta

        for sensor in self.sensorEdges.values():
            sensor.updatePositions()

    def setInterfaceAngle(self, angle: float) -> None:
        # Save current angle for reference
        self.current_angle = angle

        # Apply kinematics
        trapezoid_interface_kinematics(self._vertices, self._midpoints, angle)

        # Update all sensor edges to track deformation
        for sensor in self.sensorEdges.values():
            sensor.updatePositions()

    def getInterfaceStrain(self) -> dict[int, float]:
        return {sensor_id: sensor.getStrain()
                for sensor_id, sensor in self.sensorEdges.items()}
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
                    [nodePos[2], neighbourPos[2]], c="k", lw=1)

        for sensorEdge in self.sensorEdges.values():
            node1Pos, node2Pos = sensorEdge.getEndPoints()
            ax.plot([node1Pos[0], node2Pos[0]],
                    [node1Pos[1], node2Pos[1]],
                    [node1Pos[2], node2Pos[2]], c="r", lw=1.5)

        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_zlim(0, 70)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()


if __name__ == '__main__':
    # Create arm with default parameters
    arm = Arm(np.pi * 35 / 180, 60, 40, 4, 2)

    # Set interface angle and visualize
    arm.setInterfaceAngle(np.pi / 9)

    arm.drawArm()