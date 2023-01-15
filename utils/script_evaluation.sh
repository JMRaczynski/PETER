#!/bin/bash

for dataset in "$@"
do
  for i in {1..5}
  do
    python rating_predictor_from_explanation.py --dataset $dataset --split_number $i
  done
done