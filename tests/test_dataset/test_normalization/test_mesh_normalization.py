import os
import shutil

import numpy as np
import pytest
import trimesh
from matplotlib import pyplot as plt

from src.dataset.normalization.mesh_normalizer import MeshNormalizer
from src.dataset.normalization.enums.normalization_strategy import NormalizationStrategy

current_dir = os.path.dirname(os.path.realpath(__file__))
sample_element_name = "norm_test_duct.obj"
assets_path = os.path.join(current_dir, "../../assets")
correct_obj_file = os.path.join(assets_path, sample_element_name)


def test_normalize_zero_to_one():
    mesh = trimesh.load(correct_obj_file)
    result, _ = MeshNormalizer.normalize_zero_to_one(mesh)
    min_vals = result.min(axis=0)
    max_vals = result.max(axis=0)
    assert isinstance(result, np.ndarray)
    assert max_vals.max() == 1
    assert np.equal(min_vals, 0).any()
    mesh.vertices = result
    resulting_dir = os.path.join(assets_path, "results")
    if not os.path.exists(resulting_dir):
        os.makedirs(resulting_dir)
    mesh.export(os.path.join(resulting_dir, "test_result_1.obj"))


def test_normalize_minus_one_to_one():
    # mesh should be centered around 0
    mesh = trimesh.load(correct_obj_file)
    result, _ = MeshNormalizer.normalize_minus_one_to_one(mesh)
    min_vals = result.min(axis=0)
    max_vals = result.max(axis=0)
    assert isinstance(result, np.ndarray)
    assert max_vals.max() == 1
    assert min_vals.min() == -1
    assert np.isclose(max_vals + min_vals, 0).all()
    mesh.vertices = result
    resulting_dir = os.path.join(assets_path, "results")
    if not os.path.exists(resulting_dir):
        os.makedirs(resulting_dir)
    mesh.export(os.path.join(resulting_dir, "test_result_2.obj"))

@pytest.mark.parametrize("strategy, max_coord, min_coord", [
    (NormalizationStrategy.ZERO_TO_ONE, 1, 0),
    (NormalizationStrategy.MINUS_ONE_TO_ONE, 1, -1)
])
def test_normalize(strategy, max_coord, min_coord):
    mesh = trimesh.load(correct_obj_file)
    ifc_normalization_result = MeshNormalizer.normalize(mesh, strategy)
    result = ifc_normalization_result.mesh
    sample_points = result.sample(2048)
    min_vals = result.vertices.min(axis=0)
    max_vals = result.vertices.max(axis=0)
    assert max_vals.max() == max_coord
    assert min_vals.min() == min_coord
    fig1 = plt.figure(figsize=(8, 6))
    a = fig1.add_subplot(111, projection='3d')
    a.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2], color='g')
    fig1.suptitle(f'Normalization {strategy.value}', fontsize=20)
    plt.show()
    print('test')


def test_normalize__when_strategy_not_defined():
    mesh = trimesh.load(correct_obj_file)
    result = MeshNormalizer.normalize(mesh)
    min_vals = result.mesh.vertices.min(axis=0)
    max_vals = result.mesh.vertices.max(axis=0)
    assert np.any(max_vals - min_vals > 2)


def test_cleanup():
    resulting_dir = os.path.join(assets_path, "results")
    shutil.rmtree(resulting_dir)