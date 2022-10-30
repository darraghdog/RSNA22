# Create vertebrae label where we have segmentations
python scripts/_make_seg_labels_part1.py

# Train a number of models learn the segmentation ratio. 
for _ in 1 2 3
do
    for FOLD in 0 1 2 3 4 # -1
    do
        for MODEL_NAME in cfg_dh_seg_02G cfg_dh_seg_04A cfg_dh_seg_04F
        do 
            echo "Train fold" $FOLD " and model " $MODEL_NAME
            python train.py -C $MODEL_NAME --fold $FOLD
        done
    done
done

# Run bounding box inference of all images. 
for FOLD in 0 1 2 3 4 # -1
do
    for MODEL_NAME in cfg_dh_seg_02G cfg_dh_seg_04A cfg_dh_seg_04F
        search_dir='weights/'$MODEL_NAME'/fold' 
        search_dir+=$FOLD
        for WEIGHTS_NAME in "$search_dir"/*
        do
            echo "Running inference on "$WEIGHTS_NAME
            python train.py -C $MODEL_NAME'_test' --fold $FOLD --pretrained_weights WEIGHTS_NAME
        done
    done
done

# Aggregate vertebrae predictions and make fracture label
python scripts/_make_seg_labels_part2.py

