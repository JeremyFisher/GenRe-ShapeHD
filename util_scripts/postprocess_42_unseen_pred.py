import os
import numpy as np
import shutil

from joblib import Parallel, delayed

synsets = ['02747177', '02801938', '02818832', '02871439', '02880940',
           '02942699', '02954340', '03046257', '03207941', '03325088',
           '03467517', '03593526', '03642806', '03759954', '03790512',
           '03928116', '03948459', '04004475', '04099429', '04330267',
           '04468005', '02773838', '02808440', '02843684', '02876657',
           '02924116', '02946921', '02992529', '03085013', '03261776',
           '03337140', '03513137', '03624134', '03710193', '03761084',
           '03797390', '03938244', '03991062', '04074963', '04225987',
           '04460130', '04554684']

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
    dir_path = '/path/to/output/dir/test_genre_13_42/unseen_test_{}_genre_full_model'.format(synset)
    dest_data_dir = '/path/to/output/dir/test_genre_13_42/{}_unseen_vali'.format(synset)

    all_batches = []
    batches = [x for x in os.listdir(dir_path) if x.endswith('.npz')]
    batch_paths = [os.path.join(dir_path, x) for x in batches]

    all_batches.extend(batch_paths)

    count = 0

    results = Parallel(n_jobs=24, verbose=10, backend="multiprocessing")(map(delayed(job), all_batches))
