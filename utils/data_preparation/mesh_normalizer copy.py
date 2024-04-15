class MeshNormalizer:

    @staticmethod
    def normalize(mesh:trimesh.Trimesh, strategy:NormalizationStrategy=NormalizationStrategy.NO_NORMALIZATION):
        if strategy == NormalizationStrategy.ZERO_TO_ONE:
            new_vertices = MeshNormalizer.normalize_zero_to_one(mesh)
            mesh.vertices = new_vertices
        elif strategy == NormalizationStrategy.MINUS_ONE_TO_ONE:
            new_vertices =  MeshNormalizer.normalize_minus_one_to_one(mesh)
            mesh.vertices = new_vertices
        return mesh

    @staticmethod
    def normalize_zero_to_one(mesh:trimesh.Trimesh):
        vertices_coordinates = mesh.vertices
        min_vals = vertices_coordinates.min(axis=0)
        max_vals = vertices_coordinates.max(axis=0)
        ranges = max_vals - min_vals
        normalization_dimension = ranges.max()
        return (vertices_coordinates - min_vals) / normalization_dimension

    @staticmethod
    def normalize_minus_one_to_one(mesh:trimesh.Trimesh):
        normalized = MeshNormalizer.normalize_zero_to_one(mesh) * 2
        translation = normalized.max(axis=0) / 2
        return normalized - translation