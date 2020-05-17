import numpy as np
import os
import copy
import trimesh

from mesh_gen_utils.libkdtree import KDTree

def eval_mesh_genre(mesh, pointcloud_gt, normals_gt,\
                num_fscore_thres=6, n_points=300000):

    if mesh is not None and type(mesh)==trimesh.base.Trimesh and len(mesh.vertices) != 0 and len(mesh.faces) != 0:
        pointcloud, idx = mesh.sample(n_points, return_index=True)
        pointcloud = pointcloud.astype(np.float32)
        normals = mesh.face_normals[idx]
    else:
        return {'cd': 2*np.sqrt(3), 'completeness': np.sqrt(3),\
                    'accuracy': np.sqrt(3), 'normals_completeness': -1,\
                    'normals_accuracy': -1, 'normals': -1, \
                    'fscore': np.zeros(6, dtype=np.float32), \
                    'precision': np.zeros(6, dtype=np.float32), \
                    'recall': np.zeros(6, dtype=np.float32)}
    # Eval pointcloud
    pointcloud = np.asarray(pointcloud)
    pointcloud_gt = np.asarray(pointcloud_gt)
    normals = np.asarray(normals)
    normals_gt = np.asarray(normals_gt)

    # Completeness: how far are the points of the target point cloud
    # from thre predicted point cloud
    completeness, normals_completeness = distance_p2p(
            pointcloud_gt, normals_gt, pointcloud, normals)

    # Accuracy: how far are th points of the predicted pointcloud
    # from the target pointcloud
    accuracy, normals_accuracy = distance_p2p(
        pointcloud, normals, pointcloud_gt, normals_gt
    )

    # Get fscore
    fscore_array, precision_array, recall_array = [], [], []
    for i, thres in enumerate([0.5, 1, 2, 5, 10, 20]):
        fscore, precision, recall = calculate_fscore(\
            accuracy, completeness, thres/100.)
        fscore_array.append(fscore)
        precision_array.append(precision)
        recall_array.append(recall)
    fscore_array = np.array(fscore_array, dtype=np.float32)
    precision_array = np.array(precision_array, dtype=np.float32)
    recall_array = np.array(recall_array, dtype=np.float32)

    accuracy = accuracy.mean()
    normals_accuracy = normals_accuracy.mean()

    completeness = completeness.mean()
    normals_completeness = normals_completeness.mean()

    cd = completeness + accuracy
    normals = 0.5*(normals_completeness+normals_accuracy)

    return {'cd': cd, 'completeness': completeness,\
                'accuracy': accuracy, \
                'normals_completeness': normals_completeness,\
                'normals_accuracy': normals_accuracy, 'normals': normals, \
                'fscore': fscore_array, 'precision': precision_array,\
                'recall': recall_array}

def calculate_fscore(accuracy, completeness, threshold):
    recall = np.sum(completeness < threshold)/len(completeness)
    precision = np.sum(accuracy < threshold)/len(accuracy)
    if precision + recall > 0:
        fscore = 2*recall*precision/(recall+precision)
    else:
        fscore = 0
    return fscore, precision, recall

def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product




