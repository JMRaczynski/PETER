#!/bin/bash

for i in {1..5}
do
  python -u main.py --data_path ../datasets/Yelp/reviews.pickle \
  --index_dir ../datasets/Yelp/$i --outf generated_yelp.txt \
  --checkpoint ./models/Yelp/ --use_feature --cuda --peter_mask --additional_recommender | tee script_output_yelp_peterr_$i.log
done