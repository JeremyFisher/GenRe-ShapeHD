import os
import numpy as np
import shutil

from joblib import Parallel, delayed

synsets = ['02691156', '02828884', '02933112',
           '02958343', '03001627', '03211117',
           '03636649', '03691459', '04090263',
           '04256520', '04379243', '04401088',
           '04530566']

def job(p):
    obj_npz = np.load(p)

    #string munging
    obj_rgb_path = obj_npz['rgb_path'][0].split('/')[-1]
    obj_rgb_path = obj_rgb_path.split('_')
    cat, obj, view = obj_rgb_path[0], obj_rgb_path[1], obj_rgb_path[2]
    view = int(view[-3:])
    
    dest_obj_dir = os.path.join(dest_data_dir, cat, obj)

    if not os.path.exists(dest_obj_dir):
        try:
            os.makedirs(dest_obj_dir)
        except:
            print(0)

    dest_npz_file = '{}_{}_view{:03d}_pred_vox.npz'.format(cat, obj, view)
    dest_npz_path = os.path.join(dest_obj_dir, dest_npz_file)

    np.savez(dest_npz_path, voxel=obj_npz['pred_voxel'])
    
for synset in synsets:
    dir_path = 'path/to/output/dir/test_genre_13_42/seen_test_{}_genre_full_model'.format(synset)
    dest_data_dir = '/path/to/output/dir/test_genre_13_42/{}_seen_vali'.format(synset)

    all_batches = []
    batches = [x for x in os.listdir(dir_path) if x.endswith('.npz')]
    batch_paths = [os.path.join(dir_path, x) for x in batches]

    all_batches.extend(batch_paths)

    count = 0
    total = len(all_batches)

    results = Parallel(n_jobs=24, verbose=10, backend="multiprocessing")(map(delayed(job), all_batches))
