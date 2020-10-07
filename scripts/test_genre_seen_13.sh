#!/usr/bin/env bash

export MKL_NUM_THREADS=2
export OMP_NUM_THREADS=2

gpu=0

synsets=("02691156" "02828884" "04530566" \
         "02933112" "02958343" "03211117" \
         "03636649" "03691459" "04090263" \
         "04256520" "04379243" "04401088" \
         "04530566")


for synset in "${synsets[@]}"; do

    out_dir="path/to/output/test_genre_13_42/seen_test_$synset"
    fullmodel="path/to/full/model"
    rgb_pattern="./downloads/data/test_13_42/genre_test_13_seen_$synset/*_rgb.*"
    mask_pattern="./downloads/data/test_13_42/genre_test_13_seen_$synset/*_silhouette.*"
    
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
