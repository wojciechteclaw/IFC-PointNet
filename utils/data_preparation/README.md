
# Data preparation

This code is meant to take as input IFC files, extract indyvidual objects that contain geometry, and convert them to normalised point clouds and store in folders corresponding to their IFC entity (category).

The procedure consist of three steps:
- extract IFC elements and convert to OBJ mesh files with metadata such as: Source file, GlobalID, IFC entity, Name, ObjectType, location in the model, rotation.
- organising files in folders

- create PyTorch DataSet and DataLoader objects (https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- normalising the geometry, and saving the scale factor
- 
#TODO normalizacja do 0.0 - 1.0
#TODO zapisz info o tym ile się przeskalował (w obiekcie DATA jest parametr na to...)

- split into train/valid/test set
#TODO podziel na train/valid/test 6/2/2
