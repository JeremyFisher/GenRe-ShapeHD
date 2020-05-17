import numpy as np
import os
import trimesh
import utils
import time
import json

from multiprocessing import Pool
from skimage import measure
from joblib import Parallel, delayed
from collections import defaultdict

import argparse

home = os.path.expanduser("~")

GT_vox_dir = os.path.join(home, 'path_to_GT')
pred_dir = os.path.join(home, 'path_to_predictions')

pointcloud_size=100000

mc_thres_gt = 0.5
mc_thres_pred = 0.25
GT_TYPE = 'VOX_MESH'

def get_mesh_grid(size):
    x = np.arange(size) - size / 2.
    y = np.arange(size) - size / 2.
    z = np.arange(size) - size / 2.
    gridx, gridy, gridz = np.meshgrid(x, y, z, indexing='ij')

    grid = np.array([gridx.reshape(-1), gridy.reshape(-1), gridz.reshape(-1)])
    grid = grid/size

    return grid.T

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def eval_single(arg):
    scale = True
    verbose = True

    cat, obj, view = arg
    print('running eval for: ', cat, obj, 'with GT type', GT_TYPE)
    
    #######################
    # loading gt mesh from voxel
    gt_vox_dir = os.path.join(GT_vox_dir, cat, obj)
    gt_vox_file = next(x for x in os.listdir(gt_vox_dir) if x.endswith('{:03d}_gt_rotvox_samescale_128.npz'.format(view)))
    gt_vox_fpath = os.path.join(gt_vox_dir, gt_vox_file)
    
    try:
        gt_vox = np.load(gt_vox_fpath)['voxel'].squeeze()
    except:
        return 0
    
    try:
        verts, faces, normals, values = measure.marching_cubes_lewiner(
                                    gt_vox, mc_thres_gt, spacing=(1 / 128, 1 / 128, 1 / 128))
        gt_mesh_vox = trimesh.Trimesh(vertices=verts - 0.5, faces=faces)
    except:
        return 0

    ## bbox scaling 
    bbox = gt_mesh_vox.bounding_box.bounds
    if scale:
        # Compute location and scale
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max()

        # Transform input mesh
        try:
            gt_mesh_vox.apply_scale(1 / scale)
        except Exception:
            gt_mesh_vox.vertices = gt_mesh_vox.vertices * 1/scale
    
    gt_mesh = gt_mesh_vox
    del bbox

    # manually extracting gt pcloud and normals
    ptcl, face_idx = gt_mesh.sample(pointcloud_size, return_index=True)
    normals = gt_mesh.face_normals[face_idx]

    pointcloud_gt = ptcl.astype(np.float32)
    normals_gt = normals.astype(np.float32)
    
    #######################
    #loading prediction
    pred_item_dir = os.path.join(pred_dir, cat, obj)
    vox_file = next(x for x in os.listdir(pred_item_dir) if x.endswith('{:03d}_pred_vox.npz'.format(view)))
    vox_file = os.path.join(pred_item_dir, vox_file)
     
    #loading voxel
    pred_vox = np.load(vox_file)['voxel'].squeeze()
    pred_vox = sigmoid(pred_vox)
    
    #running marching cube to extract mesh 
    verts, faces, normals, values = measure.marching_cubes_lewiner(
                                pred_vox, mc_thres_pred, spacing=(1 / 128, 1 / 128, 1 / 128))
    pred_mesh_vox = trimesh.Trimesh(vertices=verts - 0.5, faces=faces)
    
    #rotating to align
    R_x = trimesh.transformations.rotation_matrix(90/180*np.pi,[1,0,0])
    pred_mesh_vox.apply_transform(R_x)
    
    ## bbox scaling 
    bbox = pred_mesh_vox.bounding_box.bounds
    
    if scale:
        # Compute location and scale
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max()

        # Transform input mesh
        try:
            pred_mesh_vox.apply_scale(1 / scale)
        except Exception:
            pred_mesh_vox.vertices = pred_mesh_vox.vertices * 1/scale
    
    out_dict = utils.eval_mesh_genre(pred_mesh_vox,  
                                     pointcloud_gt, 
                                     normals_gt, 
                                     n_points=300000)
    
    if verbose:
        print('cd', out_dict['cd'])
        print('normal', out_dict['normals'])
        print('fscore', out_dict['fscore'])
    
    return out_dict

def main():
    seen_synsets = ['02691156', '02828884', '02933112',
                    '02958343', '03001627', '03211117',
                    '03636649', '03691459', '04090263',
                    '04256520', '04379243', '04401088',
                    '04530566']
    
    out_dir = 'seen_13_output_dicts'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
   
    for syn in seen_synsets:

        arg_list = []
        print(syn, len(arg_list))
        objects = sorted(os.listdir(os.path.join(pred_dir, syn)))

        for idx, obj in enumerate(objects):
            
            #string munging to get views
            gt_path = os.path.join(GT_vox_dir, syn, obj)
            gt_items = [x for x in os.listdir(gt_path) if 'rotvox' in x]
            views = [int(x.split('view')[-1][:3]) for x in gt_items]
            
            for i in sorted(views):
                arg = (syn, obj, i)
                arg_list.append(arg)

        print("total items:", len(arg_list))
        results = Parallel(n_jobs=12, verbose=10, backend="multiprocessing")(map(delayed(eval_single), arg_list))
        complete_output = defaultdict(list)

        for res, arg in zip(results, arg_list):
            syn, obj, view  = arg
            if res != 0:
                complete_output[(obj, view)].append(res)

        np.savez('./{}/{}_eval_output_vox.npz'.format(out_dir, syn), output=complete_output)

if __name__ == '__main__':
   print('running main') 
   main()
