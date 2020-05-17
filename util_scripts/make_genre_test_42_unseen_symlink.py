import os
import json
import numpy as np

from collections import defaultdict

src_path  = './downloads/data/shapenet'
dest_path = './downloads/data/test_13_42_v2/'

unseen_synsets = ['02747177', '02801938', '02818832', '02871439', '02880940',
                  '02942699', '02954340', '03046257', '03207941', '03325088',
                  '03467517', '03593526', '03642806', '03759954', '03790512',
                  '03928116', '03948459', '04004475', '04099429', '04330267',
                  '04468005', '02773838', '02808440', '02843684', '02876657',
                  '02924116', '02946921', '02992529', '03085013', '03261776',
                  '03337140', '03513137', '03624134', '03710193', '03761084',
                  '03797390', '03938244', '03991062', '04074963', '04225987',
                  '04460130', '04554684']

print("making symlinks for testing, this should be fast..")
for synset in unseen_synsets:
    syn_dest_path = os.path.join(dest_path, 'genre_test_42_unseen_{}/'.format(synset))

    if not os.path.exists(syn_dest_path):
        os.makedirs(syn_dest_path)
    
    syn_path = os.path.join(src_path, synset)

    objects = os.listdir(syn_path)

    for idx, obj in enumerate(objects):

        obj_path = os.path.join(syn_path, obj)

        view = np.random.randint(0,25)
        
        f = '{}_{}_view{:03d}'.format(synset, obj, view)

        append_str = '{}/{}/{}'.format(synset, obj, f)
        rgb_path = os.path.join(src_path, append_str)+'_rgb.png'
        sil_path = os.path.join(src_path, append_str)+'_silhouette.png'
        
        rgb_bool = os.path.exists(rgb_path)
        sil_bool = os.path.exists(sil_path)

        if not (rgb_bool and sil_bool):
            continue
        
        print(synset, obj, end='\r')
        src_path_img = rgb_path
        src_path_sil = sil_path

        rgb_fname = os.path.basename(rgb_path)
        sil_fname = os.path.basename(sil_path)

        dest_path_img = os.path.join(syn_dest_path, rgb_fname)
        dest_path_sil = os.path.join(syn_dest_path, sil_fname)

        os.symlink(src_path_img, dest_path_img)
        os.symlink(src_path_sil, dest_path_sil)
    
