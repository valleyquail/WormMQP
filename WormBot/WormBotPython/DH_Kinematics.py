import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def dh_transformation_matrix(theta, d, a, alpha):
    """
    Calculate the Denavit-Hartenberg transformation matrix.

    Args:
        theta: Rotation around Z-axis (joint angle)
        d: Translation along Z-axis
        a: Translation along X-axis
        alpha: Rotation around X-axis (link twist)

    Returns:
        4x4 transformation matrix
    """
    # Create the transformation matrix
    matrix = np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])

    return matrix


def apply_dh_kinematics(vertices, unit_height, bend_angles, bend_axes):
    """
    Apply DH forward kinematics to the vertices of a continuum arm.

    Args:
        vertices: List of all node objects in the arm
        unit_height: Height of a single unit (used for all interfaces)
        bend_angles: List of bend angles to apply at each interface
        bend_axes: List of axes to bend around ('x', 'y', or 'z')
    """
    # Group nodes by level
    levels = {}
    for node in vertices:
        level = node.getLevel()
        if level not in levels:
            levels[level] = []
        levels[level].append(node)

    # Get the minimum level (base)
    base_level = min(levels.keys())

    # Store original positions
    original_positions = {node: np.array(node.getPosition()) for node in vertices}

    # Start with identity transformation matrix
    current_transform = np.eye(4)

    # Apply transformations for each interface
    for level in sorted(levels.keys())[1:]:  # Skip the base level
        interface_idx = level - base_level - 1

        # Skip if this interface is out of range
        if interface_idx < 0 or interface_idx >= len(bend_angles):
            continue

        # Get parameters for this interface
        bend_angle = bend_angles[interface_idx]
        bend_axis = bend_axes[interface_idx]

        # Create the appropriate DH parameters based on bend axis
        if bend_axis == 'x':
            # Rotation around X-axis
            theta = 0
            d = unit_height
            a = 0
            alpha = bend_angle
        elif bend_axis == 'y':
            # Rotation around Y-axis (need to compose two rotations)
            # First rotate 90 degrees around X to align Z with Y
            dh1 = dh_transformation_matrix(0, 0, 0, np.pi / 2)
            # Then rotate around the new Z (which was Y)
            dh2 = dh_transformation_matrix(bend_angle, unit_height, 0, 0)
            # Then rotate -90 degrees around X to restore orientation
            dh3 = dh_transformation_matrix(0, 0, 0, -np.pi / 2)

            # Combine the transformations
            dh_matrix = dh1 @ dh2 @ dh3
            current_transform = current_transform @ dh_matrix

            # Continue to next interface (we've already applied the transform)
            continue
        else:  # 'z' or default
            # Rotation around Z-axis
            theta = bend_angle
            d = unit_height
            a = 0
            alpha = 0

        # Create and apply the DH transformation matrix
        dh_matrix = dh_transformation_matrix(theta, d, a, alpha)
        current_transform = current_transform @ dh_matrix

        # Apply the current transformation to all nodes at this level and above
        for l in sorted(levels.keys()):
            if l >= level:
                for node in levels[l]:
                    # Get original position and convert to homogeneous coordinates
                    orig_pos = np.append(original_positions[node], 1)

                    # Apply transformation
                    new_pos = current_transform @ orig_pos

                    # Update node position
                    node.set_position(tuple(new_pos[:3]))