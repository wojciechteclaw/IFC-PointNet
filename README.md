
# IFC Entity Recognition

## What it is

IFC... Often misclassified... To ensure quality of the data... Based on geometry the algorithm is trying to recognize the object and assign an IFC entity. This can be compared to the original entity and in case of a mismatch - notify the user of possible error.  

## How it works

1. Extract pure geometry from IFC model using IfcOpenShell.
2. Save each object to indyvidual obj file. (we published our dataset here: TODO)
3. Normalise the geometry...

### Option 1: using PointNet
4. Place 1000 points equally distributed along the surface of an object, creating a point cloud.

### Option 2: using voxels
4. Divide the bounding box into 64x64x64 voxels (cubes), and depending on the geometry, assign each voxel a label:
   - 0 - if the voxel is outside the geometry (empty)
   - 0.5 - if the geometry crosses the voxel (partial)
   - 1 - if the voxel is fully immersed in the geometry (full)

6. Run the neural network train to detect IFC entities
7. Compare results with the original.

### How to use it

1. To be developed...

### Dataset we used to train the model

The full dataset used for the training of the model is published on Zenodo: https://doi.org/10.5281/zenodo.10730758

To obtain the dataset, we extracted 884008 IFC elements from 245 publicly available IFC files mostly from libraries of: 
- The University of Auckland, https://openifcmodel.cs.auckland.ac.nz/
- RWTH Aachen University, https://github.com/RWTH-E3D/DigitalHub
- Karlsruhe Institute of Technology, https://www.ifcwiki.org/index.php?title=KIT_IFC_Examples
- buildingSMART International, https://github.com/buildingSMART/Sample-Test-Files
- OSArch https://wiki.osarch.org/index.php?title=AECO_Workflow_Examples

Each OBJ file contains vertices and faces of a triangular mesh, as well as a header with metadata, for example:

```
# Source file: 'OpenIfcModel.Auckland.NZ_0912101-02wall_opening_straight_ac_1.ifc'
# GlobalId: '3m$GcQbwH0b9V0tqrRtMu_'
# entity: 'IfcDoor'
# Name: D02
# ObjectType: None
# location: [-22.,26.148,0.]
# rotation: [[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]]
```

The geometry of IfcDoor example visualized in Blender:

<img src="https://github.com/wojciechteclaw/IFC-PointNet/assets/22922395/9c1ed5ee-cf1e-40ea-9052-1e0893b6456c" width="500">

The distribution of elements in the dataset:

<img src="https://github.com/wojciechteclaw/IFC-PointNet/assets/22922395/cf940db4-4e62-4f55-9b89-499835e2c27e" width="600">

The average resolution of vertices per element:

<img src="https://github.com/wojciechteclaw/IFC-PointNet/assets/22922395/872425e3-f3be-464f-be3f-83ddbf2d0f69" width="900">
