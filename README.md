# Training Generalizable Reconstruction (GenRe) on 13 ShapeNet Classes and Testing on 42 Unseen Classes

This is a repository with some minimal extensions of the original [GenRe Repository](https://github.com/xiumingzhang/GenRe-ShapeHD) to allow for training on different ShapeNet splits. Please refer to the [original paper](http://genre.csail.mit.edu/papers/genre_nips.pdf) as well. The main focus of this repository is generating ground truth data for training. For setup please follow the instructions in the original repository.

To run our evaluation code, please compile OccNet extension modules in `./eval/mesh_gen_utils`
```bash
python setup.py build_ext --inplace
```

## Training Data Download

Download the training data for the 13/42 split on ShapeNet using the following command

```
tbd
```

## Training the Model

In the directory where the repository is cloned, make a symlink to where the data was extracted

```
ln -s path/to/data ./downloads/data/shapenet
```

Then as described in the original repository, follow [the steps](https://github.com/xiumingzhang/GenRe-ShapeHD/blob/master/README.md#genre-1) to train the GenRe model specifying `13_seen` for the `class` argument.

## Testing the Model

1. Download `data_split.json` by running `cd eval; wget https://www.dropbox.com/s/7shqu6krvs9x1ib/data_split.json; cd ..`
2. Generate symlinks for testing data by running 

        python util_scripts/make_genre_test_13_seen_symlink.py
        python util_scripts/make_genre_test_42_unseen_symlink.py
        
3. Run the testing scripts

        bash scripts/test_genre_seen_13.sh
        bash scripts/test_genre_unseen_42.sh
        
4. Postprocess the data to be in `synset/object` directory structure, since it isn't so by default.

        python util_scripts/postprocess_13_seen_pred.py
        python util_scripts/postprocess_42_unseen_pred.py

5. Run the testing scripts in the `eval` directory

        python eval_seen_13_GenRe.py
        python eval_unseen_42_GenRe.py
   
   and run the following scripts to load the results
        
        python load_seen_13_GenRe.py
        python load_unseen_42_GenRe.py
        
## Training Data Generation 

The pages below contain information to generate ground truth data for GenRe.
1. [Image Rendering](md/rendering.md)
2. [Full Spherical Images](md/spherical.md)
3. [TDF Voxel Grids](md/voxel.md)
