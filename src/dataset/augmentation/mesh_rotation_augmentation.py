from random import randint
import numpy as np
import trimesh

class MeshRotationAugmentation:
    def __init__(self, mesh):
        self.mesh = mesh

    @staticmethod
    def rotate_points(points: np.ndarray, angle: int):
        R = MeshRotationAugmentation.create_rotation_matrix(angle)
        rotated_points = np.dot(points, R.T)
        return rotated_points.__array__()

    @staticmethod
    def augment_single_element(mesh: trimesh.Trimesh, rotation_scope: int = 179):
        random_rotation = randint(-rotation_scope, rotation_scope)
        rotated_vertices = MeshRotationAugmentation.rotate_points(mesh.vertices, random_rotation)
        mesh.vertices = rotated_vertices
        return mesh

    @staticmethod
    def create_rotation_matrix(angle):
        angle_radians = np.radians(angle)
        r_cos, r_sin = np.cos(angle_radians), np.sin(angle_radians)
        return np.matrix([[r_cos, -r_sin, 0],
                          [r_sin, r_cos, 0],
                          [0, 0, 1]])
