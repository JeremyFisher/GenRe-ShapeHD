#!/usr/bin/env bash

export MKL_NUM_THREADS=2
export OMP_NUM_THREADS=2

synsets=("02747177"  "02801938"  "02818832"  "02871439"  "02880940"  "04554684" \
         "02942699"  "02954340"  "03046257"  "03207941"  "03325088"  "04460130" \
         "03467517"  "03593526"  "03642806"  "03759954"  "03790512"  "04225987" \
         "03928116"  "03948459"  "04004475"  "04099429"  "04330267"  "04074963" \
         "04468005"  "02773838"  "02808440"  "02843684"  "02876657"  "03991062" \
         "03085013"  "03261776"  "03938244"  "02924116"  "02946921"  "02992529" \
         "03337140"  "03513137"  "03624134"  "03710193"  "03761084"  "03797390")
gpu=0

for synset in "${synsets[@]}"; do

    out_dir="path/to/output/dir/test_genre_13_42/unseen_test_$synset"
    fullmodel="path/to/full/model/pt"
    rgb_pattern="./downloads/data/test_13_42/genre_test_42_unseen_$synset/*_rgb.*"
    mask_pattern="./downloads/data/test_13_42/genre_test_42_unseen_$synset/*_silhouette.*"
    
    source activate shaperecon

    python 'test.py' \
        --net genre_full_model \
        --net_file "$fullmodel" \
        --input_rgb "$rgb_pattern" \
        --input_mask "$mask_pattern" \
        --output_dir "$out_dir" \
        --suffix '{net}' \
        --overwrite \
        --workers 0 \
        --batch_size 1 \
        --vis_workers 4 \
        --gpu "$gpu" \
        $*

    source deactivate

    done
