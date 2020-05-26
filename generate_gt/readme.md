# Generating Spherical Images and Voxel Ground Truth for training GenRe

This code assumes that ShapeNetCore.v2 is downloaded and data has been rendered as described [here](link).

## Downloading

1. Download canonical voxel grids by running 
```
wget https://www.dropbox.com/s/l4scu01763edyao/ShapeNet55_TuDF_3.tar
```
2. Download `shapenet_info.json` file 
```
cd jsons; wget *add link* ; cd ..;
```

## Edit `paths.json`

- `renders_path` - path to rendered data
- `tudf_path` - path to downloaded voxel grids from the downloading step (TuDF stands for truncated unsigned distance fields).
- `shapenet_path` - path to downloaded ShapeNetCore.v2
- `output_path` - directory to output data

## Run `convert_to_genre.py`

`convert_to_genre.py` converts the outputs from the rendering step into the GenRe file structure, extracts full spherical projections and generates rotated voxel grids for each view.

This script takes `-start` and `-end` arguments so the total 55K shapenet objects can be processed in separate chunks. It's straightforward to add functionality to only include certain synsets or objects.

