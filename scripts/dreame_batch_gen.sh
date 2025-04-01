#!/bin/bash

# Number of iterations
N=5
# Starting seed
BASE_SEED=3
# Base output folder
#BASE_OUTPUT="outputs/dataset/dining_rooms"
#BASE_OUTPUT="outputs/dataset/bed_rooms"
BASE_OUTPUT="outputs/dataset/bath_rooms"
#BASE_OUTPUT="outputs/dataset/living_rooms"
for ((i=0; i<N; i++))
do
    # Calculate new seed
    SEED=$((BASE_SEED + i))
    # Create output folder name with suffix
    OUTPUT="${BASE_OUTPUT}_${i}"
    
    echo "Running iteration $i with seed $SEED and output folder $OUTPUT"
    #-g fast_solve.gin singleroom.gin \
    python -m infinigen_examples.generate_indoors \
        --seed $SEED \
        --task coarse \
        --output_folder $OUTPUT \
        -g   singleroom.gin \
        -p \
        compose_indoors.solve_medium_enabled=True \
        compose_indoors.solve_large_enabled=True \
        compose_indoors.terrain_enabled=False \
        compose_indoors.solve_small_enabled=False \
        restrict_solving.restrict_parent_rooms=\[\"Bathroom\"\] \
        restrict_solving.restrict_child_primary=\[\"Furniture\"\]
done