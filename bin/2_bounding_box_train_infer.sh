# Bounding Box Label Creation
python scripts/_make_bbox_part1.py

# Train a bounding box model
mkdir weights
for _ in 1 2 3
do
    for FOLD in 0 1 2 3 4 -1
    do
        echo "Train fold" $FOLD
        python train.py -C cfg_loc_dh_01B --fold $FOLD
    done
done

# Run bounding box inference of all images. 
for FOLD in 0 1 2 3 4 # -1
do
    search_dir='weights/cfg_loc_dh_01B/fold'
    search_dir+=$FOLD
    for WEIGHTS_NAME in "$search_dir"/*
    do
        echo "Running inference on "$WEIGHTS_NAME
        python train.py -C cfg_loc_dh_01B_test --fold $FOLD --pretrained_weights WEIGHTS_NAME
    done
done

# Aggregate bounding box predictions. 
python scripts/_make_bbox_part2.py