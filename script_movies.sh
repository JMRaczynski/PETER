#!/bin/bash

for i in {1..5}
do
  python -u main.py --data_path ../datasets/Amazon/MoviesAndTV/reviews.pickle \
  --index_dir ../datasets/Amazon/MoviesAndTV/$i --outf generated_amazonmovies.txt \
  --checkpoint ./models/AmazonMovies/ --use_feature --cuda --peter_mask --additional_recommender | tee script_output_movies_peterr_$i.log
done