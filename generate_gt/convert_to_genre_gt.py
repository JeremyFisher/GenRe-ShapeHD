import os
import copy
import misc_utils
import trimesh
import numpy as np
import shutil
import cv2
import json
import argparse

from joblib import Parallel, delayed 
from util_sph import render_spherical_simple, render_spherical
from copy import deepcopy
from scipy.io import savemat

with open('./jsons/paths.json','r') as f:
    path_dict = json.load(f)

our_path = path_dict['renders_path']
target_dir = path_dict['output_path']
shapenet_path = path_dict['shapenet_path']
tudf_path = path_dict['tudf_path']

# specify what needs to get converted
to_convert = ['rgb', 'sil', 'depth', 'normal', 'full_spherical', 'TuDF']

def arr2grid(arr):
    revox_enc = trimesh.voxel.encoding.DenseEncoding(arr.astype(bool))
    gt_grid_revox = trimesh.voxel.VoxelGrid(revox_enc)

    return gt_grid_revox

def generate_tudf_gt_metadata(synset_id, obj_id):
    obj_gt_path = os.path.join(our_path, synset_id, obj_id)
    target_path = os.path.join(target_dir, synset_id, obj_id)

    obj_tudf_path = os.path.join(tudf_path, "{}_{}_tudf.npz".format(synset_id, obj_id))

    angles = np.radians(
                np.loadtxt(os.path.join(obj_gt_path, 'metadata.txt')
                ))

    arr = np.load(obj_tudf_path)['tudf']

    #preprocessing tudf

    arr = np.transpose(arr, (0,2,1))
    arr  = np.flip(arr, (0))
    arr  = np.flip(arr, (1))

    shapenet_mesh_path = os.path.join(shapenet_path,
                                   synset_id,
                                   obj_id,
                                   'models',
                                   'model_normalized.obj')

    #loading the mesh
    canonical_mesh = misc_utils.load_obj(shapenet_mesh_path)

    #scaling it to fit in unit box WITHOUT altering the origin
    canonical_mesh = misc_utils.preprocess_shapenet_mesh(canonical_mesh)

    box = canonical_mesh.bounding_box
    shift2 = (box.vertices[0] + box.vertices[-1])/2

    ### resampling the tudf grid to "fill" the inside of the voxel grid
    grid1 = arr2grid(arr)
    S1 = np.max(grid1.bounding_box.extents)
    S1 = 128/S1

    arr = misc_utils.transform(arr, scales=(S1,S1,S1))

    #scaling down so the output rarely leaks outside the voxel grid
    scale_arr = np.array([0.6,0.6,0.6])

    #transposing the origin shift
    shift2 = shift2[[1,0,2]]
    shift2 = shift2[[0,2,1]]
    shift = -shift2*128 * np.mean(scale_arr)

    arr = misc_utils.transform(arr, scales=scale_arr)
    arr = misc_utils.transform(arr, offset=shift)

    gt_TuDF = arr

    gt_TuDF_grids = []
    for idx, angle in enumerate(angles):

        arr = copy.deepcopy(gt_TuDF)

        azim, elev = np.pi - angle[0], angle[1]
        azim *= -1
        angle_arr = np.array([0, elev, azim])

        arr = misc_utils.transform(arr, angles=angle_arr)
        
        gt_TuDF_grids.append(arr)

    # genre naming convention
    for idx, gt_vox_grid in enumerate(gt_TuDF_grids):
        target_fname = "{}_{}_view{:03d}_gt_rotvox_samescale_128.npz".format(synset_id, obj_id, idx)
        np.savez_compressed(os.path.join(target_path, target_fname), voxel=gt_vox_grid, compressed=True)

def generate_full_spherical(synset_id, obj_id, output=None):
    shapenet_mesh_path = os.path.join(shapenet_path,
                                      synset_id, 
                                      obj_id, 
                                      'models', 
                                      'model_normalized.obj')

    target_path = os.path.join(target_dir, synset_id, obj_id)
    obj_gt_path = os.path.join(our_path, synset_id, obj_id) 

    canonical_mesh = misc_utils.load_obj(shapenet_mesh_path)
    canonical_mesh = misc_utils.preprocess_shapenet_mesh(canonical_mesh)
    
    angles = np.radians(
                np.loadtxt(os.path.join(obj_gt_path, 'metadata.txt')
                ))
    
    full_sph_imgs = []
    for idx, (azim, elev) in enumerate(angles):
        # aligning with our meta file
        azim = np.pi - azim
        elev = np.pi/2 - elev

        mesh = deepcopy(canonical_mesh)
        
        azim_mat = trimesh.transformations.rotation_matrix(azim, [1,0,0])
        elev_mat = trimesh.transformations.rotation_matrix(elev, [0,0,1])
        
        mesh.apply_transform(azim_mat)
        mesh.apply_transform(elev_mat)

        im_depth = render_spherical_simple(mesh)
        im_depth = np.flip(im_depth)
        full_sph_imgs.append(im_depth)
    
    if output is None:
        return full_sph_imgs
        
    elif output == 'print_to_file':
        for idx, full_sph_img in enumerate(full_sph_imgs):
            target_fname = "{}_{}_view{:03d}_full_spherical.png".format(synset_id, obj_id, idx)
            cv2.imwrite(os.path.join(target_path, target_fname), full_sph_img*255)

    else:
        raise NameError("wrong argument value for <output>")

def convert_GT(synset_id, obj_id):
    obj_gt_path = os.path.join(our_path, synset_id, obj_id) 
    
    target_path = os.path.join(target_dir, synset_id, obj_id)
    misc_utils.make_dir(target_path)
    # getting RGB data
    if 'rgb' in to_convert:

        img_path = os.path.join(obj_gt_path, 'image_output') 
        img_paths = sorted([os.path.join(img_path, x) for x in os.listdir(img_path)])

        for idx, img_path in enumerate(img_paths):
            target_fname = "{}_{}_view{:03d}_rgb.png".format(synset_id, obj_id, idx)

            if os.path.exists(target_fname):
                continue
            
            shutil.copy(img_path, os.path.join(target_path, target_fname))
    
    # getting segmentation data
    if 'sil' in to_convert:

        sil_path = os.path.join(obj_gt_path, 'segmentation') 
        sil_paths = sorted([os.path.join(sil_path, x) for x in os.listdir(sil_path)])

        for idx, sil_path in enumerate(sil_paths):
            target_fname = "{}_{}_view{:03d}_silhouette.png".format(synset_id, obj_id, idx)
            
            if os.path.exists(target_fname):
                continue
            shutil.copy(sil_path, os.path.join(target_path, target_fname))
    
    # getting surface normal data
    if 'normal' in to_convert:

        normal_path = os.path.join(obj_gt_path, 'normal_output') 
        normal_paths = sorted([os.path.join(normal_path, x) for x in os.listdir(normal_path)])

        for idx, normal_path in enumerate(normal_paths):
            target_fname = "{}_{}_view{:03d}_normal.png".format(synset_id, obj_id, idx)
            
            if os.path.exists(target_fname):
                continue
            
            shutil.copy(normal_path, os.path.join(target_path, target_fname))
    
    # transferring depth + depth min/max
    if 'depth' in to_convert:
        
        depth_path = os.path.join(obj_gt_path, 'depth_NPZ') 
        depth_paths = sorted([os.path.join(depth_path, x) for x in os.listdir(depth_path)])

        for idx, npz_path in enumerate(depth_paths):
            target_fname_1 = "{}_{}_view{:03d}_depth.png".format(synset_id, obj_id, idx)
            target_fname_2 = "{}_{}_view{:03d}.npy".format(synset_id, obj_id, idx)
            
            if os.path.exists(target_fname_1) and os.path.exists(target_fname_2):
                continue

            arr = np.load(npz_path, allow_pickle=True)
            depth_image = arr['img']
            min_max = arr['min_max']
            
            cv2.imwrite(os.path.join(target_path, target_fname_1), depth_image*255)
            np.save(os.path.join(target_path, target_fname_2), min_max)
    
    if 'full_spherical' in to_convert:    
        #generating full inpainted maps
        full_sph_imgs = generate_full_spherical(synset_id, obj_id)
        
        #saving inpainted maps
        for idx, full_sph in enumerate(full_sph_imgs):
            target_fname = "{}_{}_view{:03d}_spherical.npz".format(synset_id, obj_id, idx)
            np.savez(os.path.join(target_path, target_fname), 
                     obj_spherical=full_sph.astype(np.float32))
    
    if 'TuDF' in to_convert:
        generate_tudf_gt_metadata(synset_id, obj_id)

def job(args):
    synset_id, obj_id = args

    try: 
        convert_GT(synset_id, obj_id)
        return 
    except:
        return [synset_id, obj_id]

def main():

    parser = argparse.ArgumentParser(description='Range of Objects')
    parser.add_argument('-start', type=int, help='start point', required=True)
    parser.add_argument('-end', type=int, help='end point', required=True)

    args = parser.parse_args()

    with open('jsons/shapenet_info.json','r') as f:
        obj_dict = json.load(f)
     
    synset_list_13 = ['02691156', '02828884', '02933112',
                      '02958343', '03001627', '03211117',
                      '03636649', '03691459', '04090263',
                      '04256520', '04379243', '04401088',
                      '04530566']

    ls = [] 
    for key in sorted(obj_dict.keys()):
        if key not in synset_list_13:
            continue
        for obj in sorted(obj_dict[key]):
            
            ls.append([key, obj])

    ls = ls[args.start:args.end]

    '''
    to run in parallel uncomment this
    results = Parallel(n_jobs=n_jobs, verbose=1, backend='multiprocessing')(map(delayed(job), ls))
    np.save(os.path.join(target_dir, 'job_output.npy'),
           np.array([x for x in results if x != 1]))


    '''
    for idx, args in enumerate(ls):
        print(convert_GT(*args))
    


if __name__ == "__main__":
    main()
