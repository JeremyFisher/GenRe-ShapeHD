import os
import json
import numpy as np

from collections import defaultdict

src_path  = '../downloads/data/shapenet'
dest_path = '../downloads/data/test_13_42_v2/'

seen_synsets = ['02691156', '02828884', '02933112',
                '02958343', '03001627', '03211117',
                '03636649', '03691459', '04090263',
                '04256520', '04379243', '04401088',
                '04530566']

print("making symlinks for testing, this should be fast..")
for synset in seen_synsets:
    syn_dest_path = os.path.join(dest_path, 'genre_test_13_seen_{}/'.format(synset))

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
    
