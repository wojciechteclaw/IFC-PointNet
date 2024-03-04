
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
