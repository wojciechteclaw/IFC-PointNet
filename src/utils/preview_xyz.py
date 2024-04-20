import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def preview_xyz(path):
    xyz = np.loadtxt(path, dtype=np.float32)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="black", s=0.03, alpha=0.9)

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    # Set the limits for each axis to 0.0 to 1.0
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_zlim([0.0, 1.0])

    plt.show()


if __name__ == "__main__":
    PATH = r"C:\Code\IFC-PointNet\data\sample_1000_XYZ\IfcBeam\000083_IfcBeam_x3.94.xyz"
    preview_xyz(PATH)
