#!/bin/bash

for i in {1..5}
do
  python -u main.py --data_path ../datasets/TripAdvisor/reviews.pickle \
  --index_dir ../datasets/TripAdvisor/$i --outf generated_tripadvisor.txt \
  --checkpoint ./models/TripAdvisor/ --use_feature --cuda --peter_mask --additional_recommender | tee script_output_ta_peterr_$i.log
done