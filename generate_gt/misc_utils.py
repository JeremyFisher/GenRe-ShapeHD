import os
import trimesh
import cv2
import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi

def make_dir(directory):
    if not os.path.exists(directory):
            os.makedirs(directory)

def save_meshes(path, meshes):
    
    make_dir(path)
    
    print("Saving meshes to: {}".format(path))
    count = 0
    for idx, mesh in enumerate(meshes):
        print("Saving mesh [{}/{}]".format(idx, len(meshes)), end='\r')
        try:
            mesh.export(file_obj = os.path.join(path,'{:03d}.obj'.format(idx)))
            count+=1
        except:
            raise Warning("Could not save mesh {} to {}".format(idx, path))
    
    print("Saved {} meshes.".format(count))

def save_depth_images(path, images):
    
    make_dir(path)

    for idx, img in enumerate(images):
        cv2.imwrite(os.path.join(path, '{:03d}.png'.format(idx)), img*255)

def load_obj(path):
    '''
        loads a mesh at path and strips the unncessary materials
        so that trimesh functions can be easily applied
    '''
    try:
        mesh = trimesh.load_mesh(path)
        mesh = scene2mesh(mesh)
    except AttributeError:
        pass

    return mesh

def scene2mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        return scene_or_mesh

    return mesh

def normalize_mesh(mesh):
    '''
        scales the mesh down uniformly in all directions
        so that it fits inside a 3D cube with side length 2
        around the origin
    ''' 
    max_0 = np.max(np.abs(mesh.vertices[:,0]))
    max_1 = np.max(np.abs(mesh.vertices[:,1]))
    max_2 = np.max(np.abs(mesh.vertices[:,2]))

    div_max = np.max([max_0, max_1, max_2])
    
    mesh.vertices /= div_max

    return mesh

def preprocess_shapenet_mesh(mesh):
    '''
        rotating and scaling the shapenet mesh so that it fits
        in a unit cube without changing the origin
    '''
    x_rot = 0
    y_rot = 90
    z_rot = 90

    x_mat = trimesh.transformations.rotation_matrix(
            np.radians(x_rot), 
            [1,0,0])

    y_mat = trimesh.transformations.rotation_matrix(
            np.radians(y_rot), 
            [0,1,0])
    
    z_mat = trimesh.transformations.rotation_matrix(
            np.radians(z_rot), 
            [0,0,1])
    
    mesh = normalize_mesh(mesh)

    mesh.apply_transform(x_mat)
    mesh.apply_transform(y_mat)
    mesh.apply_transform(z_mat)
    
    #make the mesh fit in a unit cube
    scale = trimesh.transformations.scale_matrix(1/np.max(mesh.extents), (0,0,0))
    mesh.apply_transform(scale)

    return mesh

### Borrowed from original GenRe repository ###

def _get_centeralized_mesh_grid(sx, sy, sz):
    x = np.arange(sx) - sx / 2.
    y = np.arange(sy) - sy / 2.
    z = np.arange(sz) - sz / 2.
    return np.meshgrid(x, y, z, indexing='ij')


def get_rotation_matrix(angles):
    # # legacy code
    # alpha, beta, gamma = angles
    # R_alpha = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
    # R_beta = np.array([[1, 0, 0], [0, np.cos(beta), -np.sin(beta)], [0, np.sin(beta), np.cos(beta)]])
    # R_gamma = np.array([[np.cos(gamma), 0, -np.sin(gamma)], [0, 1, 0], [np.sin(gamma), 0, np.cos(gamma)]])
    # R = np.dot(np.dot(R_alpha, R_beta), R_gamma)
    alpha, beta, gamma = angles
    R_alpha = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    R_beta = np.array([[np.cos(beta), 0, -np.sin(beta)], [0, 1, 0], [np.sin(beta), 0, np.cos(beta)]])
    R_gamma = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
    R = np.dot(np.dot(R_alpha, R_beta), R_gamma)
    return R


def get_scale_matrix(scales):
    return np.diag(scales)


def transform_by_matrix(voxel, matrix, offset):
    """
    transform a voxel by matrix, then apply an offset
    Note that the offset is applied after the transformation
    """
    sx, sy, sz = voxel.shape
    gridx, gridy, gridz = _get_centeralized_mesh_grid(sx, sy, sz)  # the coordinate grid of the new voxel
    mesh = np.array([gridx.reshape(-1), gridy.reshape(-1), gridz.reshape(-1)])
    mesh_rot = np.dot(np.linalg.inv(matrix), mesh) + np.array([sx / 2, sy / 2, sz / 2]).reshape(3, 1)
    mesh_rot = mesh_rot - np.array(offset).reshape(3, 1)    # grid for new_voxel should get a negative offset

    interp = rgi((np.arange(sx), np.arange(sy), np.arange(sz)), voxel,
                 method='linear', bounds_error=False, fill_value=0)
    new_voxel = interp(mesh_rot.T).reshape(sx, sy, sz)  # todo: move mesh to center
    return new_voxel


def transform(voxel, angles=(0, 0, 0), scales=(1, 1, 1), offset=(0, 0, 0), threshold=None, clamp=False):
    """
    transform a voxel by first rotate, then scale, then add offset.
    shortcut for transform_by_matrix
    """
    matrix = np.dot(get_rotation_matrix(angles), get_scale_matrix(scales))
    new_voxel = transform_by_matrix(voxel, matrix, offset)
    if clamp:
        new_voxel = np.clip(new_voxel, 0, 1)
    if threshold is not None:
        new_voxel = (new_voxel > threshold).astype(np.uint8)
    return new_voxel

